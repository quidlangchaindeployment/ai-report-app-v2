# steps/step_2.py
# --- Step 2: 構造化 (AWS Bedrock) ---
import os
import re
import json
import time
import tempfile
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd

from utils.dependencies import (
    HAS_LLM, HAS_AWS, require_aws, require_llm,
    get_llm, get_aws_clients, upload_file_to_s3, create_batch_job,
    stop_batch_job, get_batch_job_status, download_file_from_s3, read_file
)

MODEL_FLASH_LITE = "gemini-2.5-flash-lite"

# =============================================================================
# LLM 支援 (カテゴリ・除外ワードの自動生成)
# =============================================================================
def _ai_generate_categories(analysis_prompt: str) -> Optional[Dict[str, str]]:
    if not HAS_LLM or get_llm is None: return None
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None: return None

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
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        text = str(raw)
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return None
        return {str(k): str(v) for k, v in json.loads(m.group(1) if m.lastindex else m.group(0)).items()}
    except Exception as e:
        st.error(f"カテゴリ自動生成エラー: {e}")
        return None

def _ai_generate_exclusion_list(analysis_prompt: str) -> List[str]:
    default_ng = ["料理", "ごはん", "ご飯", "ランチ", "ディナー", "肉", "牛肉", "和牛", "ステーキ", "丼", "言及なし", "該当なし"]
    if not HAS_LLM or get_llm is None: return default_ng
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None: return default_ng

    prompt = (
        "あなたはデータ分析のノイズ除去担当です。以下の「分析指針」から、"
        "特性（形容詞）として抽出してはいけない「名詞」（品目/部位/料理名など）を20～30語、"
        "JSONの文字列配列で出力してください。\n"
        f"# 分析指針:\n{analysis_prompt}\n"
        '# 出力形式:\n["単語1", "単語2", ...]\n'
    )
    try:
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        text = str(raw)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m: return default_ng
        return sorted(list(set([str(w).strip() for w in json.loads(m.group(0))] + default_ng)))
    except Exception:
        return default_ng

# =============================================================================
# Bedrock バッチ処理関連
# =============================================================================
def _make_jsonl_for_bedrock(df: pd.DataFrame, text_col: str, categories: Dict[str, str], exclusion_list: List[str], analysis_prompt: str) -> str:
    cats_str = json.dumps(categories, ensure_ascii=False)
    excl_str = ", ".join(exclusion_list)
    tmp = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".jsonl")
    with tmp as f:
        for _, row in df.iterrows():
            rid = row.get("record_id", 0)
            text = str(row.get(text_col, ""))[:10000]
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
    if not require_aws(): return None

    input_bucket = os.getenv("BEDROCK_S3_INPUT_BUCKET")
    output_bucket = os.getenv("BEDROCK_S3_OUTPUT_BUCKET")
    role_arn = os.getenv("BEDROCK_ROLE_ARN")
    if not all([input_bucket, output_bucket, role_arn]):
        st.error("AWS設定が不足しています（.envファイルを確認してください）。")
        return None

    job_name = f"bedrock-job-{int(time.time())}"
    input_key = f"input/{job_name}.jsonl"
    output_prefix = f"output/{job_name}/"

    if not upload_file_to_s3(local_jsonl_path, input_bucket, input_key):
        st.error("S3へのアップロードに失敗しました。")
        return None

    job_arn = create_batch_job(job_name, input_key, output_prefix)
    return job_arn

def _flatten_ai(ai_json: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    is_rel = ai_json.get("relevant")
    out["relevant"] = bool(is_rel.lower() == "true" if isinstance(is_rel, str) else is_rel)
    cats = ai_json.get("categories", {})
    if isinstance(cats, dict):
        for k, v in cats.items():
            out[k] = ", ".join(map(str, v)) if isinstance(v, list) else v
    inferred = ai_json.get("inferred_location", "")
    out["市区町村キーワード"] = inferred if inferred not in ["該当なし", "不明"] else ""
    return out

def _collect_and_merge_results(job_arn: str) -> pd.DataFrame:
    if not require_aws(): return pd.DataFrame()
    s3, bedrock = get_aws_clients()
    
    try:
        job_resp = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        s3_output_uri = job_resp.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri")
        if not s3_output_uri:
            st.error("ジョブ情報からS3出力パスを取得できませんでした。")
            return pd.DataFrame()
        bucket, _, prefix = s3_output_uri.replace("s3://", "").partition("/")
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        st.error(f"S3/Bedrock情報の取得に失敗: {e}")
        return pd.DataFrame()

    if "Contents" not in objects:
        st.warning(f"指定パス({prefix})に結果ファイルが見つかりません。")
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for obj in objects["Contents"]:
            key = obj["Key"]
            if not key.endswith(".out"): continue
            local = os.path.join(tmpdir, os.path.basename(key))
            if download_file_from_s3(bucket, key, local):
                with open(local, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            rid = int(data.get("recordId"))
                            ai_text = data.get("modelOutput", {}).get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")
                            m = re.search(r"```json\s*(\{.*?\})\s*```", ai_text, re.DOTALL) or re.search(r"\{.*\}", ai_text, re.DOTALL)
                            ai_json = json.loads(m.group(1) if (m and m.lastindex) else (m.group(0) if m else "{}"))
                            results.append({"record_id": rid, "raw_ai_output": ai_text[:1000], **_flatten_ai(ai_json)})
                        except Exception:
                            continue

    if not results:
        st.warning("有効な結果データを抽出できませんでした。")
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values("record_id")
    master = st.session_state.get("current_master_df", pd.DataFrame())
    final_df = pd.merge(master, result_df, on="record_id", how="left") if not master.empty and "record_id" in master.columns else result_df

    csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("結果CSVをダウンロード", csv_bytes, "bedrock_analysis_results.csv", "text/csv", key="download_final_csv_step_a")
    return final_df

# =============================================================================
# 画面描画（UI 本体）
# =============================================================================
def render() -> None:
    st.title("💫 Step 2: 構造化 (AWS Bedrock)")

    # ---------- 1-1. データのアップロード ----------
    with st.container(border=True):
        st.subheader("1) データのアップロード")
        uploaded_files = st.file_uploader("分析データ (Excel/CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
        master_df = st.session_state["current_master_df"]
        text_col: Optional[str] = None

        if uploaded_files:
            dfs = [df for df, err in (read_file(f) for f in uploaded_files) if df is not None]
            if dfs:
                master_df = pd.concat(dfs, ignore_index=True)
                if "record_id" not in master_df.columns:
                    master_df["record_id"] = range(len(master_df))
                st.session_state["current_master_df"] = master_df
                st.success(f"読み込み完了: 合計 {len(master_df)} 件")

        if not master_df.empty:
            text_col = st.selectbox("分析対象のテキスト列を選択:", master_df.columns, key="text_col_select")

    # ---------- 1-2. 分析ルールの設計（前処理） ----------
    with st.container(border=True):
        st.subheader("2) 分析ルールの設計（前処理）")
        analysis_prompt = st.text_area("分析指針（目的）", key="analysis_prompt_A", height=100)

        col_g, col_e = st.columns([1, 1])
        with col_g:
            if st.button("カテゴリをAIで自動生成", disabled=not HAS_LLM):
                if analysis_prompt:
                    cats = _ai_generate_categories(analysis_prompt)
                    if cats:
                        st.session_state["cats_editor"] = json.dumps(cats, ensure_ascii=False, indent=2)
                        st.success("カテゴリを生成しました。")
                else:
                    st.warning("分析指針を入力してください。")

        with col_e:
            if st.button("除外ワードをAIで自動生成", disabled=not HAS_LLM):
                if analysis_prompt:
                    ex_list = _ai_generate_exclusion_list(analysis_prompt)
                    st.session_state["ex_editor"] = ", ".join(ex_list)
                    st.success("除外ワードを生成しました。")
                else:
                    st.warning("分析指針を入力してください。")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.text_area("カテゴリ定義（JSON）", key="cats_editor", height=220)
        with c2:
            st.text_area("特性として抽出しない単語", key="ex_editor", height=220)

    # ---------- 1-3. AWSバッチ分析の実行とステータス追跡 ----------
    with st.container(border=True):
        st.subheader("3) AWSバッチ分析の実行と結果取得")
        
        current_arn = st.session_state.get("last_job_arn", "")
        
        # 新規ジョブの投入
        if not current_arn:
            if st.button("AWSで分析を開始 (ジョブ投入)", type="primary", disabled=not HAS_AWS or master_df.empty or text_col is None):
                cats_text = st.session_state.get("cats_editor", "").strip()
                final_categories = json.loads(cats_text) if cats_text else {}
                final_ex_list = [w.strip() for w in st.session_state.get("ex_editor", "").split(",") if w.strip()]

                with st.spinner("S3へアップロードし、ジョブを作成しています..."):
                    jsonl_path = _make_jsonl_for_bedrock(master_df, text_col, final_categories, final_ex_list, analysis_prompt)
                    job_arn = _submit_job_to_bedrock(jsonl_path)
                    if os.path.exists(jsonl_path): os.remove(jsonl_path)
                
                if job_arn:
                    st.session_state["last_job_arn"] = job_arn
                    st.rerun() # ステータス追跡UIへ切り替え
        
        # ジョブ実行中・完了時のUI (ARNが存在する場合)
        else:
            st.info(f"対象ジョブID: `{current_arn}`")
            status = get_batch_job_status(current_arn) if HAS_AWS else "Unknown"
            
            if status in ["Submitted", "InProgress"]:
                with st.status(f"⏳ ジョブ実行中... (ステータス: {status})", expanded=True):
                    st.write("AWS Bedrock で推論処理を行っています。データ量によりますが数分〜数十分かかります。")
                    
                    col_refresh, col_stop = st.columns(2)
                    with col_refresh:
                        if st.button("🔄 最新ステータスを確認"):
                            st.rerun()
                    with col_stop:
                        if st.button("🛑 ジョブを強制停止", type="secondary"):
                            stop_batch_job(current_arn)
                            st.warning("停止リクエストを送信しました。")
                            st.rerun()

            elif status == "Completed":
                st.success("✅ ジョブが正常に完了しました！")
                if st.button("📥 結果を取得・結合してダウンロード", type="primary"):
                    with st.spinner("S3から結果をダウンロードし、結合しています..."):
                        final_df = _collect_and_merge_results(current_arn)
                    if not final_df.empty:
                        st.subheader("抽出結果プレビュー（結合済み）")
                        st.dataframe(final_df.head(10), use_container_width=True)
                        
                        # 自動で次のステップへ進める導線
                        if st.button("次のステップ (深層分析) へ進む"):
                            st.session_state["current_step"] = 3
                            st.rerun()

            elif status in ["Failed", "Stopped", "Error"]:
                st.error(f"❌ ジョブが異常終了しました (ステータス: {status})")
                
            st.markdown("---")
            if st.button("クリアして新しいジョブを開始する"):
                st.session_state["last_job_arn"] = ""
                st.rerun()