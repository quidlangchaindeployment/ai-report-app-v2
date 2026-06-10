# steps/step_1.py
# --- Step 1: データ収集 (QUID連携) ---
import streamlit as st
import asyncio

# --- 共通ユーティリティへの依存 ---
from utils.dependencies import HAS_LLM
from services.browser_agent import run_quid_metrics_agent

def render():
    st.title("🌐 Step 1: データ収集 (QUID連携)")
    
    if not st.session_state.get("research_design"):
        st.error("先に Step 0 で分析設計を確定させてください。")
        if st.button("Step 0 に戻る"):
            st.session_state["current_step"] = 0
            st.rerun()
        return

    topic_name = st.session_state["research_design"]["topic_creation"].get("topic_name", "Unknown Topic")

    # --- ユーザーへの事前指示 ---
    st.warning(f"""
    **⚠️ QUID自動収集エージェント 起動前の準備**
    1. 起動ボタンを押すと、背後でブラウザが立ち上がります。
    2. ログインが必要な場合は、表示されたブラウザで手動ログインを済ませてください。
    3. ログイン完了後、エージェントが自動的に作業を再開し、「{topic_name}」のメトリクスを抽出します。
    """)

    col_action1, col_action2 = st.columns([2, 1])
    
    with col_action1:
        if st.button("🚀 エージェントを起動してメトリクスを取得", type="primary", disabled=not HAS_LLM):
            with st.spinner("エージェントがブラウザを操作中...（必要ならログインしてください）"):
                try:
                    result = asyncio.run(run_quid_metrics_agent(topic_name))
                    
                    # ★修正ポイント：エラー文字列が返ってきた場合はJSONとして扱わずエラー表示する
                    if isinstance(result, str) and result.startswith("エージェント実行エラー"):
                        st.error(result)
                    else:
                        st.session_state["extracted_metrics"] = result
                        st.success("✅ 作業が完了しました。")
                except Exception as e:
                    st.error(f"予期せぬエラーが発生しました: {e}")

    with col_action2:
        if st.button("🔄 エージェントを再起動（リトライ）"):
            st.rerun()
            
    # 結果の表示と次のステップへの導線
    metrics = st.session_state.get("extracted_metrics")
    if metrics:
        # 辞書などのデータ形式の場合のみJSONとして表示する（クラッシュ防止）
        if isinstance(metrics, (dict, list)):
            st.json(metrics)
        else:
            st.write(metrics)
            
        if st.button("✅ 次のステップ (AWS Bedrock構造化) へ進む"):
            st.session_state["current_step"] = 2
            st.rerun()

    # --- 作業が止まることへの対応策：手動入力フォーム ---
    st.markdown("---")
    with st.expander("🛠️ エージェントが止まってしまった場合（手動補完）"):
        st.info("自動抽出に失敗した場合は、ブラウザで確認した数値を以下に直接入力してください。")
        manual_mentions = st.text_input("メンション数")
        manual_sentiment = st.text_input("センチメント傾向 (%)")
        if st.button("数値を手動で確定"):
            st.session_state["extracted_metrics"] = {"mentions": manual_mentions, "sentiment": manual_sentiment}
            st.success("手動入力値を保存しました。")
            if st.button("次のステップ (AWS Bedrock構造化) へ進む", key="btn_manual_next"):
                st.session_state["current_step"] = 2
                st.rerun()