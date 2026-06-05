# steps/step_0.py
# --- Step 0: 戦略設計 (Design) ---
import json
import re
from datetime import datetime, timedelta

import streamlit as st

# --- 共通ユーティリティへの依存 ---
from utils.dependencies import HAS_LLM, get_llm

def _robust_json_parser(text: str) -> dict:
    """不完全なJSONやMarkdown装飾を修復してパースする"""
    if not text: raise ValueError("AIからの回答が空です。")
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    start_idx = text.find('{')
    if start_idx == -1: raise ValueError("JSONの開始点が見つかりません。")
    text = text[start_idx:]
    if text.count('"') % 2 != 0: text += '"'
    open_braces, close_braces = text.count('{'), text.count('}')
    if open_braces > close_braces: text += '}' * (open_braces - close_braces)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            text = re.sub(r',\s*([}\]])', r'\1', text)
            return json.loads(text)
        except: raise ValueError("JSON復旧に失敗しました。AIの回答が長すぎた可能性があります。")

def render():
    st.title("📋 Step 0: 戦略・リサーチ設計")
    st.caption("QUID Monitor設定マニュアルに準拠し、高精度な抽出条件を策定します。")

    # --- Section 1: ユーザー指定項目 ---
    with st.container():
        st.subheader("1) 分析要件の設定")
        col_bg, col_pur = st.columns(2)
        background = col_bg.text_area("分析の背景", height=120, key="input_bg", placeholder="例：○○県の観光に関して、ソーシャル上の観光客の評価を取得できておらず...")
        purpose = col_pur.text_area("分析の目的", height=120, key="input_pur", placeholder="例：特定の観光エリアでの購買行動やルートを可視化し...")

        col_date, col_lang = st.columns(2)
        today = datetime.now()
        default_start = today - timedelta(days=30)
        date_range = col_date.date_input("分析期間の設定", value=(default_start, today))
        
        languages = col_lang.multiselect(
            "対象言語の選択",
            ["日本語", "英語", "中国語（簡体字）", "中国語（繁体字）", "韓国語", "フランス語", "ドイツ語", "スペイン語"],
            default=["日本語"]
        )

    st.markdown("---")
    st.subheader("2) AI生成オプション")
    model_mapping = {
        "Gemini 2.5 Flash (推奨)": "gemini-2.5-flash",
        "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite"
    }
    selected_label = st.selectbox("使用するモデル", list(model_mapping.keys()))
    valid_model_id = model_mapping[selected_label]

    if st.button("🪄 高精度な抽出条件を生成する", type="primary", disabled=not HAS_LLM):
        if not background or not purpose or len(date_range) < 2 or not languages:
            st.warning("全ての項目を入力・選択してください。")
            return

        start_date = date_range[0].strftime("%Y-%m-%d")
        end_date = date_range[1].strftime("%Y-%m-%d")
        lang_str = ", ".join(languages)

        llm = get_llm(model_name=valid_model_id, temperature=0.1, max_output_tokens=1000)
        
        prompt = f"""
        あなたはQUID Monitorを使いこなすSNSアナリストです。
        提供された要件に基づき、データのノイズを最小化し、分析精度を最大化する設定JSONを作成してください。

        【背景】: {background}
        【目的】: {purpose}
        【期間】: {start_date} から {end_date}
        【言語】: {lang_str}

        ## キーワード生成の厳格な役割分担:
        1. **主要キーワード (main_keywords)**: センチメント判定の核となる語。日英で5-10個に厳選。
        2. **含めるキーワード (include_keywords)**: 母集団を絞り込むための地域名や必須文脈語（AND条件）。
        3. **除外キーワード (exclude_keywords)**: 
           - **数は30個以内を厳守**。
           - 短すぎる語（例：イラン、タイ、アップル）は、部分一致により「ハイランド」等まで消すリスクがあるため禁止。
           - 代わりに「富士急ハイランド」「タイランド銀行」など、具体的で紛らわしい固有名詞やキャンペーン語（抽選、当たる、ギフト券）を指定すること。

        ## 出力形式 (純粋なJSONのみ):
        {{
          "topic_creation": {{
            "topic_name": "Project_YYYYMMDD",
            "start_date": "{start_date}",
            "end_date": "{end_date}",
            "language": "{lang_str}",
            "main_keywords": [],
            "include_keywords": [],
            "exclude_keywords": [],
            "search_range": "パラグラフ内での一致"
          }},
          "analysis_strategy": {{
            "methods": ["投稿量推移", "センチメント分析", "StoryScope属性分析"],
            "insight_hypothesis": "100文字程度の仮説"
          }}
        }}
        """

        with st.spinner("アナリストエージェントが戦略を策定中..."):
            try:
                res = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
                extracted_json = _robust_json_parser(res)
                st.session_state["research_design"] = extracted_json
                st.success("高精度な設計案が完成しました。下部で最終調整を行ってください。")
            except Exception as e:
                st.error(f"生成エラー: {str(e)}")

    # --- Section 3: プレビュー・調整 ---
    if st.session_state.get("research_design"):
        st.markdown("---")
        design = st.session_state["research_design"]
        tc = design.get("topic_creation", {})
        
        st.subheader("3) 最終調整（QUID設定画面へ転記）")
        with st.form("precision_form_v8"):
            topic_name = st.text_input("トピック名", value=tc.get("topic_name", ""))
            
            c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
            sd = c1.text_input("開始日", value=tc.get("start_date", ""))
            ed = c2.text_input("終了日", value=tc.get("end_date", ""))
            lg = c3.text_input("言語", value=tc.get("language", ""))
            sr = c4.selectbox("推奨検索範囲", ["文中での一致", "パラグラフ内での一致", "ドキュメント内での一致"], 
                              index=1 if "パラグラフ" in tc.get("search_range","") else 0)
            
            def to_str(l): return ", ".join(l) if isinstance(l, list) else str(l)
            mk = st.text_area("主要キーワード（ポジネガ判定の対象）", value=to_str(tc.get("main_keywords", [])))
            ik = st.text_area("含めるキーワード（母集団の定義/AND条件）", value=to_str(tc.get("include_keywords", [])))
            ek = st.text_area("除外キーワード（ノイズ排除・30個厳選）", value=to_str(tc.get("exclude_keywords", [])), height=180)

            st.write("**推奨分析アプローチ:**", to_str(design.get("analysis_strategy", {}).get("methods", [])))
            st.info(f"**分析仮説:** {design.get('analysis_strategy', {}).get('insight_hypothesis', '')}")

            if st.form_submit_button("✅ この条件で確定して次へ"):
                st.session_state["research_design"]["topic_creation"] = {
                    "topic_name": topic_name, "start_date": sd, "end_date": ed, "language": lg, "search_range": sr,
                    "main_keywords": [s.strip() for s in mk.split(",") if s.strip()],
                    "include_keywords": [s.strip() for s in ik.split(",") if s.strip()],
                    "exclude_keywords": [s.strip() for s in ek.split(",") if s.strip()]
                }
                # Step 1へ自動遷移
                st.session_state["current_step"] = 1
                st.rerun()