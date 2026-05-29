# steps/step_a.py
from __future__ import annotations

import os
import re
import json
import time
import tempfile
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd

# =============================================================================
# 外部サービス存在チェック（未移植でもアプリが落ちないように）
# =============================================================================
_HAS_LLM = False
_HAS_AWS = False

try:
    from services.llm import get_llm  # 例: Gemini など
    _HAS_LLM = True
except Exception:
    get_llm = None

try:
    from services.aws_bedrock import (
        get_aws_clients,
        upload_file_to_s3,
        create_batch_job,
        stop_batch_job,
        get_batch_job_status,
        download_file_from_s3,
    )
    _HAS_AWS = True
except Exception:
    get_aws_clients = None
    upload_file_to_s3 = None
    create_batch_job = None
    stop_batch_job = None
    get_batch_job_status = None
    download_file_from_s3 = None


# utils.io_helpers が無い環境でも最低限の読込を提供
try:
    from utils.io_helpers import read_file  # type: ignore
except Exception:
    def read_file(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            name = file.name.lower()
            if name.endswith(".csv"):
                return pd.read_csv(file, encoding="utf-8"), None
            if name.endswith((".xlsx", ".xls")):
                return pd.read_excel(file), None
            return None, "未対応のファイル形式"
        except Exception as e:
            return None, str(e)


# =============================================================================
# セッション初期化（KeyError防止：必ず setdefault）
# =============================================================================
def _init_state() -> None:
    defaults = {
        # Step A の業務データ
        "generated_categories": {},      # {"カテゴリ名": "説明文"}
        "exclusion_list": [],            # ["牛肉","ステーキ", ...]
        "analysis_prompt_A": "",         # 分析指針
        "current_master_df": pd.DataFrame(),
        "last_job_arn": "",

        # エディタ表示（value= を使わず state を唯一のソースにする）
        "cats_editor": "",
        "ex_editor": "",
        # "text_col_select": None,       # 必要に応じて列選択の既定
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# =============================================================================
# LLM 支援
# =============================================================================
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"

def _ai_generate_categories(analysis_prompt: str) -> Optional[Dict[str, str]]:
    """分析指針からカテゴリ定義（{カテゴリ名: 説明}）を生成。"""
    if not _HAS_LLM or get_llm is None:
        st.warning("LLMサービスが未移植のため、カテゴリ自動生成はスキップします。")
        return None

    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        st.error("Gemini クライアントの生成に失敗しました。")
        return None

    prompt = (
        "あなたはデータ分析のスキーマ設計者です。\n"
        "以下の分析指針を読み、テキストから抽出すべきカテゴリを考案してください。\n\n"
        "# 分析指針:\n"
        f"{analysis_prompt}\n\n"
        "# 指示:\n"
        "- 出力は JSON 辞書のみ\n"
        '- 形式: {"カテゴリ名": "説明文"}\n'
        "- 値は必ず「人が読める説明文」にする\n"
        "- 配列やサンプル値は含めない\n"
        "- 地名（市区町村）は含めない\n"
    )

    try:
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore
        text = str(raw)

        # ```json ...``` を優先抽出、無ければ最初の {...}
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            st.warning("AI応答からJSONを抽出できませんでした。")
            return None

        raw_json = json.loads(m.group(1) if m.lastindex else m.group(0))
        return {str(k): str(v) for k, v in raw_json.items()}
    except Exception as e:
        st.error(f"カテゴリ自動生成エラー: {e}")
        return None


def _ai_generate_exclusion_list(analysis_prompt: str) -> List[str]:
    """分析指針から除外語リストを生成。未移植なら既定候補を返す。"""
    default_ng = ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼", "言及なし", "該当なし"]
    if not _HAS_LLM or get_llm is None:
        return default_ng

    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        return default_ng

    prompt = (
        "あなたはデータ分析のノイズ除去担当です。以下の「分析指針」から、"
        "特性（形容詞）として抽出してはいけない「名詞」（品目/部位/料理名など）を20～30語、"
        "JSONの文字列配列で出力してください。\n"
        f"# 分析指針:\n{analysis_prompt}\n"
        '# 出力形式:\n["単語1", "単語2", ...]\n'
    )
    try:
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore
        text = str(raw)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return default_ng
        base = json.loads(m.group(0))
        return sorted(list(set([str(w).strip() for w in base] + default_ng)))
    except Exception:
        return default_ng


# =============================================================================
# Bedrock バッチ：入力JSONLの生成／結果の取得
# =============================================================================
def _make_jsonl_for_bedrock(
    df: pd.DataFrame,
    text_col: str,
    categories: Dict[str, str],
    exclusion_list: List[str],
    analysis_prompt: str,
    record_id_col: str = "record_id",
) -> str:
    cats_str = json.dumps(categories, ensure_ascii=False)
    excl_str = ", ".join(exclusion_list)

    tmp = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".jsonl")
    with tmp as f:
        for _, row in df.iterrows():
            rid = row[record_id_col]
            text = str(row[text_col])[:10000]
            prompt = f"""
あなたはデータ抽出のスペシャリストです。以下の「分析指針」と「ルール」に基づき、情報を抽出してください。
# 分析指針:
{analysis_prompt}
# 抽出ターゲット(カテゴリ定義):
{cats_str}
# 【重要】除外ルール(NGワード):
[{excl_str}]
# 対象テキスト:
{text}
# 出力フォーマット(JSONのみ):
{{"relevant": true, "categories": {{"カテゴリ名": "値"}}, "inferred_location": "市区町村名"}}
""".strip()

            body = {
                "recordId": str(rid),
                "modelInput": {
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"max_new_tokens": 2048, "temperature": 0.0},
                },
            }
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
    return tmp.name


def _submit_job_to_bedrock(local_jsonl_path: str) -> Optional[str]:
    if not _HAS_AWS or any(fn is None for fn in [upload_file_to_s3, create_batch_job, get_aws_clients]):
        st.error("AWSサービス（services.aws_bedrock）が未移植のため、ジョブ投入はスキップされました。")
        return None

    input_bucket = os.getenv("BEDROCK_S3_INPUT_BUCKET")
    output_bucket = os.getenv("BEDROCK_S3_OUTPUT_BUCKET")
    role_arn = os.getenv("BEDROCK_ROLE_ARN")
    if not input_bucket or not output_bucket or not role_arn:
        st.error("AWS設定が不足（BEDROCK_S3_INPUT_BUCKET / BEDROCK_S3_OUTPUT_BUCKET / BEDROCK_ROLE_ARN）。")
        return None

    ts = int(time.time())
    job_name = f"bedrock-job-{ts}"
    input_key = f"input/{job_name}.jsonl"
    output_prefix = f"output/{job_name}/"

    ok = upload_file_to_s3(local_jsonl_path, input_bucket, input_key)
    if not ok:
        st.error("S3 へのアップロードに失敗しました。")
        return None

    job_arn = create_batch_job(job_name, input_key, output_prefix)
    if not job_arn:
        st.error("Bedrock ジョブ作成に失敗しました。")
        return None

    st.success("✅ 分析ジョブがAWSで開始されました！")
    st.info(f"ジョブID(ARN): `{job_arn}`")
    st.session_state["last_job_arn"] = job_arn
    return job_arn


def _flatten_ai(ai_json: Dict[str, Any]) -> Dict[str, Any]:
    """AIのJSONをフラットに整形。"""
    out: Dict[str, Any] = {}
    is_rel = ai_json.get("relevant")
    if isinstance(is_rel, str):
        is_rel = is_rel.lower() == "true"
    out["relevant"] = bool(is_rel) if is_rel is not None else None

    cats = ai_json.get("categories", {})
    if isinstance(cats, dict):
        for k, v in cats.items():
            out[k] = ", ".join(map(str, v)) if isinstance(v, list) else v

    inferred = ai_json.get("inferred_location", "")
    out["市区町村キーワード"] = inferred if inferred not in ["該当なし", "不明"] else ""
    return out


def _collect_and_merge_results(job_arn: str) -> pd.DataFrame:
    """S3 の .out を読み取り → 元データと結合 → CSV ダウンロードボタン表示。"""
    if not _HAS_AWS or any(fn is None for fn in [get_aws_clients, download_file_from_s3]):
        st.error("AWSサービス（services.aws_bedrock）が未移植のため、結果取得はスキップされました。")
        return pd.DataFrame()

    s3, bedrock = get_aws_clients()
    try:
        job_resp = bedrock.get_model_invocation_job(jobIdentifier=job_arn)  # type: ignore
        s3_output_uri = job_resp.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri")
        if not s3_output_uri:
            st.error("ジョブ情報からS3出力パスを取得できませんでした。")
            return pd.DataFrame()
        bucket, _, prefix = s3_output_uri.replace("s3://", "").partition("/")
    except Exception as e:
        st.error(f"ジョブ詳細の取得に失敗しました: {e}")
        return pd.DataFrame()

    try:
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)  # type: ignore
    except Exception as e:
        st.error(f"S3オブジェクト一覧の取得に失敗しました: {e}")
        return pd.DataFrame()
    if "Contents" not in objects:
        st.error(f"指定パス({prefix})に .out ファイルが見つかりません。")
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for obj in objects["Contents"]:
            key = obj["Key"]
            if not key.endswith(".out"):
                continue
            local = os.path.join(tmpdir, os.path.basename(key))
            if not download_file_from_s3(bucket, key, local):
                continue
            try:
                with open(local, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            rid = int(data.get("recordId"))
                            content = (
                                data.get("modelOutput", {})
                                    .get("output", {})
                                    .get("message", {})
                                    .get("content", [])
                            )
                            ai_text = content[0].get("text", "") if content else "{}"
                            # JSON抽出
                            m = re.search(r"```json\s*(\{.*?\})\s*```", ai_text, re.DOTALL)
                            if not m:
                                m = re.search(r"\{.*\}", ai_text, re.DOTALL)
                            ai_json = json.loads(m.group(1) if (m and m.lastindex) else (m.group(0) if m else "{}"))

                            results.append({"record_id": rid, "raw_ai_output": ai_text[:1000], **_flatten_ai(ai_json)})
                        except Exception:
                            continue
            except Exception:
                continue

    if not results:
        st.warning("有効な結果データを抽出できませんでした。")
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values("record_id")
    master = st.session_state.get("current_master_df", pd.DataFrame())
    if not master.empty and "record_id" in master.columns:
        final_df = pd.merge(master, result_df, on="record_id", how="left")
    else:
        final_df = result_df

    csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結果CSVをダウンロード",
        csv_bytes,
        "bedrock_analysis_results.csv",
        "text/csv",
        key="download_final_csv_step_a",
    )
    return final_df


# =============================================================================
# 画面描画（UI 本体）
# =============================================================================
def render() -> None:
    _init_state()
    st.title("💫 Step A: AWS Bedrock バッチ分析（v2）")

    # ---------- 1-1. データのアップロード ----------
    with st.container(border=True):
        st.subheader("1-1. データのアップロード")

        uploaded_files = st.file_uploader(
            "分析データ (Excel/CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True
        )

        master_df: pd.DataFrame = st.session_state.get("current_master_df", pd.DataFrame())
        text_col: Optional[str] = None

        if uploaded_files:
            dfs = []
            for f in uploaded_files:
                df, err = read_file(f)
                if df is not None:
                    dfs.append(df)
                elif err:
                    st.error(f"{f.name}: {err}")

            if dfs:
                master_df = pd.concat(dfs, ignore_index=True)

            if not master_df.empty:
                if "record_id" not in master_df.columns:
                    master_df["record_id"] = range(len(master_df))
                st.session_state["current_master_df"] = master_df
                st.success(f"読み込み完了: 合計 {len(master_df)} 件")

        if not master_df.empty:
            text_col = st.selectbox("分析対象のテキスト列を選択:", master_df.columns, key="text_col_select")

    # ---------- 1-2. 分析ルールの設計（前処理） ----------
    with st.container(border=True):
        st.subheader("1-2. 分析ルールの設計（前処理）")
        st.info("Gemini（LLM）が利用可能であれば自動設計できます。未移植でも手動編集で進められます。", icon="ℹ️")

        analysis_prompt = st.text_area(
            "分析指針（目的）",
            key="analysis_prompt_A",   # ← value= を渡さず state を唯一のソースに
            height=100,
        )

        col_g, col_e = st.columns([1, 1])
        with col_g:
            if st.button("カテゴリをAIで自動生成", disabled=not _HAS_LLM, key="btn_gen_cats"):
                if not analysis_prompt:
                    st.warning("分析指針を入力してください。")
                else:
                    cats = _ai_generate_categories(analysis_prompt)
                    if cats:
                        st.session_state["generated_categories"] = cats
                        # ★ エディタ表示へ即時反映
                        st.session_state["cats_editor"] = json.dumps(cats, ensure_ascii=False, indent=2)
                        st.success("カテゴリを生成しました。")

        with col_e:
            if st.button("除外ワードをAIで自動生成", disabled=not _HAS_LLM, key="btn_gen_ex"):
                if not analysis_prompt:
                    st.warning("分析指針を入力してください。")
                else:
                    ex_list = _ai_generate_exclusion_list(analysis_prompt)
                    st.session_state["exclusion_list"] = ex_list
                    # ★ エディタ表示へ即時反映
                    st.session_state["ex_editor"] = ", ".join(ex_list)
                    st.success("除外ワードを生成しました。")

        # 既に state に値があり、エディタ state が空のときは同期（初回表示の抜けを補正）
        if st.session_state.get("generated_categories") and not st.session_state.get("cats_editor"):
            st.session_state["cats_editor"] = json.dumps(
                st.session_state["generated_categories"], ensure_ascii=False, indent=2
            )
        if st.session_state.get("exclusion_list") and not st.session_state.get("ex_editor"):
            st.session_state["ex_editor"] = ", ".join(st.session_state["exclusion_list"])

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**抽出カテゴリ（JSON）**")
            # ★ 警告対策：value を渡さず、key だけにする（表示は session_state['cats_editor'] のみ）
            st.text_area("カテゴリ定義（JSON）", key="cats_editor", height=220)
        with c2:
            st.markdown("**除外ワード（カンマ区切り）**")
            st.text_area("特性として抽出しない単語", key="ex_editor", height=220)

    # ---------- 1-3. AWSバッチ分析の実行 ----------
    with st.container(border=True):
        st.subheader("1-3. AWSバッチ分析の実行")

        btn_disabled = (not _HAS_AWS) or master_df.empty or (text_col is None)
        if st.button(
            "AWSで分析を開始 (ジョブ投入)",
            type="primary",
            help="上記の設定に基づき、AWS上で一括処理を行います。",
            disabled=btn_disabled,
            key="btn_start_batch",
        ):
            if master_df.empty:
                st.error("入力データがありません。")
                st.stop()
            if text_col is None:
                st.error("分析対象のテキスト列を選択してください。")
                st.stop()

            # エディタの state から確定値を取得
            cats_text = st.session_state.get("cats_editor", "").strip()
            ex_text   = st.session_state.get("ex_editor", "").strip()
            try:
                final_categories = json.loads(cats_text) if cats_text else {}
            except Exception as e:
                st.error(f"カテゴリ定義(JSON)の解析に失敗: {e}")
                final_categories = {}
            final_ex_list = [w.strip() for w in ex_text.split(",") if w.strip()]

            with st.spinner("Bedrock用JSONLを生成中..."):
                jsonl_path = _make_jsonl_for_bedrock(
                    df=master_df,
                    text_col=text_col,
                    categories=final_categories,
                    exclusion_list=final_ex_list,
                    analysis_prompt=analysis_prompt or "",
                )

            with st.spinner("S3へアップロードし、ジョブを作成しています..."):
                job_arn = _submit_job_to_bedrock(jsonl_path)

            # 一時ファイルの後片付け
            try:
                if os.path.exists(jsonl_path):
                    os.remove(jsonl_path)
            except Exception:
                pass

            if job_arn:
                st.info("タブ『2. 結果確認 & ダウンロード』でステータス確認・取得を行ってください。")

    # ---------- 2. 結果の確認とダウンロード ----------
    with st.container(border=True):
        st.subheader("2. 結果の確認とダウンロード")

        default_arn = st.session_state.get("last_job_arn", "")
        job_arn_input = st.text_input("ジョブID (ARN) を入力:", value=default_arn, key="job_arn_input")

        col_status, col_stop = st.columns([1, 1])
        with col_status:
            if st.button("ステータス確認", disabled=not _HAS_AWS, key="btn_status"):
                if not _HAS_AWS:
                    st.error("services.aws_bedrock が未移植のため、確認できません。")
                elif not job_arn_input:
                    st.warning("ジョブIDを入力してください。")
                else:
                    status = get_batch_job_status(job_arn_input) if get_batch_job_status else "Unknown"
                    st.info(f"ステータス: **{status}**")
                    if status == "Completed":
                        st.success("完了しました。下のボタンから結果を取得してください。")

        with col_stop:
            if st.button("ジョブを強制停止", type="secondary", disabled=not _HAS_AWS, key="btn_stop"):
                if not _HAS_AWS:
                    st.error("services.aws_bedrock が未移植のため、停止できません。")
                elif not job_arn_input:
                    st.warning("ジョブIDを入力してください。")
                elif stop_batch_job:
                    ok = stop_batch_job(job_arn_input)
                    st.success("停止リクエストを送信しました。" if ok else "停止機能が未実装です。")

    if st.button(
        "結果を取得・結合・クリーニング",
        type="primary",
        help="S3から結果をダウンロードし、元データと結合してCSV化します。",
        disabled=not _HAS_AWS,
        key="btn_collect",
    ):
        if not _HAS_AWS:
            st.error("services.aws_bedrock が未移植のため、結果取得はできません。")
        elif not st.session_state.get("job_arn_input"):
            st.warning("ジョブIDを入力してください。")
        else:
            with st.spinner("S3から結果をダウンロードし、結合しています..."):
                final_df = _collect_and_merge_results(st.session_state["job_arn_input"])
            if isinstance(final_df, pd.DataFrame) and not final_df.empty:
                st.subheader("抽出結果プレビュー（結合済み）")
                st.dataframe(final_df.head(10), use_container_width=True)


# 単体実行（開発用）
if __name__ == "__main__":
    render()