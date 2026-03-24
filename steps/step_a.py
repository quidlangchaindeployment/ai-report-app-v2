# steps/step_a.py
# --- Step A: AWS Bedrock バッチ分析（v2） ------------------------------------------
# 役割：
# - CSV/Excelアップロード → 結合 → record_id 付与
# - Gemini でカテゴリ定義/除外語の自動生成（services.llm.get_llm があれば使用）
# - JSONL を一時ファイルに書き出し → services.aws_bedrock で S3 アップロード & ジョブ作成
# - ジョブID保存、ステータス確認、S3 から .out をダウンロード → 元データと left join → CSV 出力
# -----------------------------------------------------------------------------

import os
import io
import json
import time
import tempfile
import re
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd

_HAS_LLM = False
_HAS_AWS = False
try:
    from services.llm import get_llm  # -> Gemini クライアント
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


try:
    from utils.io_helpers import read_file  # CSV/Excel 読み込み
except Exception:
    # 簡易フォールバック（本物の read_file が未移植でも最低限動く）
    def read_file(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            if file.name.lower().endswith(".csv"):
                return pd.read_csv(file, encoding="utf-8"), None
            elif file.name.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(file), None
            return None, "未対応のファイル形式"
        except Exception as e:
            return None, str(e)


# --- 定数（必要に応じて config.py へ移動） ----------------------------------------
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
BATCH_MODEL_ID = "amazon.nova-lite-v1:0"  # Bedrock のバッチ用
NOVA_LITE_INPUT_PRICE_PER_1M = 0.06
NOVA_LITE_OUTPUT_PRICE_PER_1M = 0.24


# --- UI/状態の初期化 --------------------------------------------------------------
def _init_state():
    defaults = {
        "generated_categories": {},          # AI 生成カテゴリ（JSON）
        "exclusion_list": [],                # AI or 手入力の除外語
        "analysis_prompt_A": "",             # 分析指針のテキスト
        "current_master_df": pd.DataFrame(), # アップロード結合済みデータ
        "last_job_arn": "",                  # 直近のBedrockジョブARN
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# --- 1) カテゴリ定義の自動生成（LLM有無で分岐） -----------------------------------
def _ai_generate_categories(analysis_prompt: str) -> Optional[Dict[str, str]]:
    """分析指針から抽出カテゴリのJSON辞書を作る（LLM未移植でも安全に戻る）"""
    if not _HAS_LLM or get_llm is None:
        st.warning("LLMサービス（services.llm）が未移植のため、カテゴリ自動生成はスキップします。")
        return None

    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        st.error("Gemini クライアントの生成に失敗しました（APIキー未設定など）。")
        return None

    prompt = (
        "あなたはデータ分析のスキーマ設計者です。「分析指針」を読み、"
        "テキストから抽出するべき「カテゴリ」（キー）と説明（値）を考案してください。\n"
        "# 分析指針:\n"
        f"{analysis_prompt}\n"
        "# 条件:\n"
        "- 地名（市区町村）は別途自動処理するため除外\n"
        "- 出力は厳格なJSON辞書のみ（例：{\"カテゴリ名\": \"説明\", ...}）\n"
    )
    try:
        # LangChain 風の .invoke を利用しない直呼び（get_llm の実装に依存）
        # ここではシンプルに .invoke / .predict のどちらにも対応
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore

        text = str(raw)
        # JSON抽出（```json ... ```にも耐える）
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            st.warning("AI応答からJSONを抽出できませんでした。")
            return None
        return json.loads(m.group(0).replace("'", '"'))
    except Exception as e:
        st.error(f"カテゴリ自動生成エラー: {e}")
        return None


def _ai_generate_exclusion_list(analysis_prompt: str) -> List[str]:
    """特性抽出で無視すべき名詞（品目/部位/料理名など）のリスト（LLM未移植でもデフォルト返却）"""
    if not _HAS_LLM or get_llm is None:
        # フォールバック（最小限）
        return ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼"]

    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        return ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼"]

    prompt = (
        "あなたはデータ分析のノイズ除去担当です。以下の「分析指針」から、"
        "特性（形容詞）として抽出してはいけない「名詞」（品目/部位/料理名など）を20～30語、JSONの文字列配列で出力してください。\n"
        f"# 分析指針:\n{analysis_prompt}\n"
        "# 出力形式:\n[\"単語1\", \"単語2\", ...]\n"
    )
    try:
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore
        text = str(raw)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼"]
        base = json.loads(m.group(0))
        defaults = ["言及なし", "該当なし"]
        return sorted(list(set(base + defaults)))
    except Exception:
        return ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼"]


# --- 2) Bedrockバッチ：JSONLの生成（行ごと） --------------------------------------
def _make_jsonl_for_bedrock(df: pd.DataFrame, text_col: str,
                            categories: Dict[str, str], exclusion_list: List[str],
                            analysis_prompt: str, record_id_col: str = "record_id") -> str:
    """
    Bedrock バッチ推論へ渡す JSONL ファイルを一時生成し、ファイルパスを返す。
    """
    cats_str = json.dumps(categories, ensure_ascii=False)
    excl_str = ", ".join(exclusion_list)

    tmp = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".jsonl")
    with tmp as f:
        for _, row in df.iterrows():
            rid = row[record_id_col]
            text = str(row[text_col])[:10000]  # 安全のために上限を設定
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


# --- 3) S3 へアップロード → Bedrock ジョブ作成 -----------------------------------
def _submit_job_to_bedrock(local_jsonl_path: str) -> Optional[str]:
    if not _HAS_AWS or any(fn is None for fn in [upload_file_to_s3, create_batch_job, get_aws_clients]):
        st.error("AWSサービス（services.aws_bedrock）が未移植のため、ジョブ投入はスキップされました。")
        return None

    # .env から読み込み（load_dotenv は app.py で実行している想定）
    input_bucket = os.getenv("BEDROCK_S3_INPUT_BUCKET")
    output_bucket = os.getenv("BEDROCK_S3_OUTPUT_BUCKET")
    role_arn = os.getenv("BEDROCK_ROLE_ARN")
    if not input_bucket or not output_bucket or not role_arn:
        st.error("AWS設定が不足しています（BEDROCK_S3_INPUT_BUCKET / BEDROCK_S3_OUTPUT_BUCKET / BEDROCK_ROLE_ARN）。")
        return None

    timestamp = int(time.time())
    job_name = f"bedrock-job-{timestamp}"
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


# --- 4) S3 から .out を取得 → 元データと結合 → CSV ダウンロード -------------------
def _collect_and_merge_results(job_arn: str) -> pd.DataFrame:
    """S3の .out を集めて DataFrame 化 → current_master_df と left join"""
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

    # ダウンロードボタン
    csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結果CSVをダウンロード",
        csv_bytes,
        "bedrock_analysis_results.csv",
        "text/csv",
        key="download_final_csv_step_a",
    )
    return final_df


def _flatten_ai(ai_json: Dict[str, Any]) -> Dict[str, Any]:
    """AI応答（JSON）から必要フィールドを平坦化"""
    out: Dict[str, Any] = {}
    is_rel = ai_json.get("relevant")
    if isinstance(is_rel, str):
        is_rel = is_rel.lower() == "true"
    out["relevant"] = bool(is_rel) if is_rel is not None else None
    cats = ai_json.get("categories", {})
    if isinstance(cats, dict):
        for k, v in cats.items():
            if isinstance(v, list):
                out[k] = ", ".join(map(str, v))
            else:
                out[k] = v
    inferred = ai_json.get("inferred_location", "")
    out["市区町村キーワード"] = inferred if inferred not in ["該当なし", "不明"] else ""
    return out


# --- 5) UI 本体 -------------------------------------------------------------------
def render():
    _init_state()
    st.title("🚀 Step A: AWS Bedrock バッチ分析（v2）")

    tab1, tab2 = st.tabs(["1. 設定生成 & 分析開始", "2. 結果確認 & ダウンロード"])

    # --- Tab1: 入力とジョブ投入 ---------------------------------------------------
    with tab1:
        st.header("1-1. データのアップロード")
        uploaded_files = st.file_uploader(
            "分析データ (Excel/CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True
        )

        master_df = pd.DataFrame()
        text_col = None

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

            # record_id 付与 & セッションへ保存
            if not master_df.empty:
                if "record_id" not in master_df.columns:
                    master_df["record_id"] = range(len(master_df))
                st.session_state["current_master_df"] = master_df
                st.success(f"読み込み完了: 合計 {len(master_df)} 件")
                text_col = st.selectbox("分析対象のテキスト列を選択:", master_df.columns)

        st.markdown("---")
        st.header("1-2. 分析ルールの設計 (前処理)")
        st.info("Gemini（LLM）が利用可能であれば自動設計できます。未移植でも手動編集で進められます。")

        analysis_prompt = st.text_area(
            "分析指針（目的）を入力:",
            value=st.session_state.get("analysis_prompt_A", ""),
            placeholder="例: 栃木県の農産物に関する分析。①農産品カテゴリ... ②農産品特性... を抽出したい。",
            height=100,
        )
        st.session_state["analysis_prompt_A"] = analysis_prompt

        col_g, col_e = st.columns(2)
        with col_g:
            if st.button("カテゴリをAIで自動生成", disabled=not _HAS_LLM):
                if not analysis_prompt:
                    st.warning("分析指針を入力してください。")
                else:
                    cats = _ai_generate_categories(analysis_prompt)
                    if cats:
                        st.session_state["generated_categories"] = cats
                        st.success("カテゴリを生成しました。")

        with col_e:
            if st.button("除外ワードをAIで自動生成", disabled=not _HAS_LLM):
                if not analysis_prompt:
                    st.warning("分析指針を入力してください。")
                else:
                    ex_list = _ai_generate_exclusion_list(analysis_prompt)
                    st.session_state["exclusion_list"] = ex_list
                    st.success("除外ワードを生成しました。")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("抽出カテゴリ（JSON）")
            edited_cats_str = st.text_area(
                "カテゴリ定義 (JSON)",
                value=json.dumps(st.session_state.get("generated_categories", {}), ensure_ascii=False, indent=2),
                height=220,
                key="cats_editor",
            )

        with c2:
            st.subheader("除外ワード (カンマ区切り)")
            edited_ex_str = st.text_area(
                "特性として抽出しない単語",
                value=", ".join(st.session_state.get("exclusion_list", [])),
                height=220,
                key="ex_editor",
            )

        st.markdown("---")
        st.header("1-3. AWSバッチ分析の実行")

        if st.button(
            "AWSで分析を開始 (ジョブ投入)",
            type="primary",
            help="上記の設定に基づき、AWS上で一括処理を行います。",
            disabled=not _HAS_AWS or master_df.empty or not text_col,
        ):
            if not _HAS_AWS:
                st.error("services.aws_bedrock が未移植のため、ジョブ投入はできません。")
            elif master_df.empty or not text_col:
                st.error("入力データとテキスト列の選択が必要です。")
            else:
                # 入力チェック
                try:
                    final_categories = json.loads(edited_cats_str) if edited_cats_str.strip() else {}
                except Exception as e:
                    st.error(f"カテゴリ定義(JSON)の解析に失敗: {e}")
                    final_categories = {}

                final_ex_list = [w.strip() for w in edited_ex_str.split(",") if w.strip()]

                # JSONL 生成
                with st.spinner("Bedrock用JSONLを生成中..."):
                    jsonl_path = _make_jsonl_for_bedrock(
                        df=master_df,
                        text_col=text_col,
                        categories=final_categories,
                        exclusion_list=final_ex_list,
                        analysis_prompt=analysis_prompt or "",
                    )

                # S3 へアップロード & ジョブ作成
                with st.spinner("S3へアップロードし、ジョブを作成しています..."):
                    job_arn = _submit_job_to_bedrock(jsonl_path)

                # 後始末（ローカル一時ファイル削除）
                try:
                    if os.path.exists(jsonl_path):
                        os.remove(jsonl_path)
                except Exception:
                    pass

                if job_arn:
                    st.info("タブ『2. 結果確認 & ダウンロード』でステータス確認・取得を行ってください。")

    # --- Tab2: ジョブステータス & 結果取得 ------------------------------------------
    with tab2:
        st.header("2. 結果の確認とダウンロード")

        default_arn = st.session_state.get("last_job_arn", "")
        job_arn_input = st.text_input("ジョブID (ARN) を入力:", value=default_arn)

        col_status, col_stop = st.columns(2)
        with col_status:
            if st.button("ステータス確認", disabled=not _HAS_AWS):
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
            if st.button("ジョブを強制停止", type="secondary", disabled=not _HAS_AWS):
                if not _HAS_AWS:
                    st.error("services.aws_bedrock が未移植のため、停止できません。")
                elif not job_arn_input:
                    st.warning("ジョブIDを入力してください。")
                elif stop_batch_job:
                    ok = stop_batch_job(job_arn_input)
                    if ok:
                        st.success("停止リクエストを送信しました。")
                else:
                    st.error("停止機能が未実装です。")

        st.markdown("---")
        if st.button(
            "結果を取得・結合・クリーニング",
            type="primary",
            help="S3から結果をダウンロードし、元データと結合してCSV化します。",
            disabled=not _HAS_AWS,
        ):
            if not _HAS_AWS:
                st.error("services.aws_bedrock が未移植のため、結果取得はできません。")
            elif not job_arn_input:
                st.warning("ジョブIDを入力してください。")
            else:
                with st.spinner("S3から結果をダウンロードし、結合しています..."):
                    final_df = _collect_and_merge_results(job_arn_input)
                if not final_df.empty:
                    st.subheader("抽出結果プレビュー（結合済み）")
                    st.dataframe(final_df.head(10), use_container_width=True)