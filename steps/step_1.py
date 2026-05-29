# steps/step_1.py
import streamlit as st
import asyncio
import pandas as pd
from services.browser_agent import run_quid_metrics_agent

def render():
    st.title("🌐 Step 1: データ収集 (AIエージェント連携)")
    
    if not st.session_state.get("research_design"):
        st.error("先に Step 0 で分析設計を確定させてください。")
        return

    topic_name = st.session_state["research_design"]["topic_creation"].get("topic_name")

    # --- ユーザーへの事前指示 ---
    st.warning(f"""
    **⚠️ エージェント起動前の準備**
    1. 起動ボタンを押すと、背後でブラウザが立ち上がります。
    2. ログインが必要な場合は、表示されたブラウザで手動ログインを済ませてください。 [cite: 1076, 1111]
    3. ログイン完了後、エージェントが自動的に作業を再開します。
    """)

    col_action1, col_action2 = st.columns([2, 1])
    
    with col_action1:
        if st.button("🚀 エージェントを起動してメトリクスを取得", type="primary"):
            with st.spinner("エージェントがブラウザを操作中...（必要ならログインしてください）"):
                try:
                    result = asyncio.run(run_quid_metrics_agent(topic_name))
                    st.session_state["extracted_metrics"] = result
                    st.success("✅ 作業が完了しました。")
                    st.json(result)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

    with col_action2:
        # 作業が止まった際のリセット用
        if st.button("🔄 エージェントを再起動（リトライ）"):
            st.rerun()

    # --- 作業が止まることへの対応策：手動入力フォーム ---
    with st.expander("🛠️ エージェントが止まってしまった場合（手動補完）"):
        st.info("自動抽出に失敗した場合は、ブラウザで確認した数値を以下に直接入力してください。")
        manual_mentions = st.text_input("メンション数")
        manual_sentiment = st.text_input("センチメント傾向 (%)")
        if st.button("数値を手動で確定"):
            st.session_state["extracted_metrics"] = {"mentions": manual_mentions, "sentiment": manual_sentiment}
            st.success("手動入力値を保存しました。")

    st.markdown("---")
    # ... (CSVアップロード機能はそのまま維持)