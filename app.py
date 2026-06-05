# app.py
import streamlit as st

# --- ページ設定 (必ず最初に呼び出す) ---
st.set_page_config(
    page_title="AI共創型分析プラットフォーム (for QUID Monitor)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 共通ユーティリティの読み込み ---
from utils.session_manager import init_session_state, reset_session

# --- 各ステップモジュールのインポート ---
from steps import step_0, step_1, step_2, step_3, step_4, step_5

def main() -> None:
    # 状態の初期化
    init_session_state()

    # --- カスタムCSS（UIの微調整） ---
    st.markdown("""
        <style>
        .stButton>button { width: 100%; border-radius: 5px; }
        .stSidebar { background-color: #f0f2f6; }
        </style>
    """, unsafe_allow_html=True)

    # --- サイドバー：ナビゲーション ---
    with st.sidebar:
        st.title("分析ワークフロー")
        st.caption("QUID Monitor × AI 連携分析システム")
        
        st.markdown("---")
        
        # 画面表示用のリスト
        step_labels = [
            "0. 戦略設計 (Design)",
            "1. データ収集 (QUID連携)",
            "2. 構造化 (AWS Bedrock)",
            "3. 深層分析 (Interactive)",
            "4. レポート生成 (Draft)",
            "5. PowerPoint出力 (Output)"
        ]

        selected_label = st.radio(
            "現在の工程を選択",
            step_labels,
            index=st.session_state["current_step"],
            key="nav_radio"
        )
        
        st.markdown("---")
        with st.expander("⚙️ 設定・トラブルシュート", expanded=False):
            if st.button("🧹 セッションを初期化", use_container_width=True):
                reset_session()
                st.rerun()
 
    # --- ルーティング ---
    if selected_label.startswith("0"):
        st.session_state["current_step"] = 0
        step_0.render()

    elif selected_label.startswith("1"):
        st.session_state["current_step"] = 1
        step_1.render()

    elif selected_label.startswith("2"):
        st.session_state["current_step"] = 2
        step_2.render()

    elif selected_label.startswith("3"):
        st.session_state["current_step"] = 3
        step_3.render()

    elif selected_label.startswith("4"):
        st.session_state["current_step"] = 4
        step_4.render()

    elif selected_label.startswith("5"):
        st.session_state["current_step"] = 5
        step_5.render()

# エントリポイント
if __name__ == "__main__":
    main()