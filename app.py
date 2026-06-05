# app.py
import streamlit as st
import time

# --- ページ設定 ---
st.set_page_config(
    page_title="AI共創型分析プラットフォーム",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 各ステップモジュールのインポート ---
from steps import step_0, step_1, step_a, step_b, step_d

def _init_state():
    """セッション状態の初期化"""
    defaults = {
        "research_design": None, # Step 0 で生成した設計データ
        "raw_data": None,        # Step 1 でアップロードしたCSV/Excel
        "log_messages": [],
        "current_step": 0,       # 現在の進捗ステップ（0-4）
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main() -> None:
    _init_state()

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
        st.caption("AIエージェント共創型システム")
        
        st.markdown("---")
        
        # 画面表示用のリスト
        step_labels = [
            "0. 戦略設計 (Design)",
            "1. データ収集 (LAM/QUID)",
            "2. 構造化 (AWS Bedrock)",
            "3. 深層分析 (Interactive)",
            "4. レポート (Output)"
        ]

        # ラジオボタンでページを選択
        # current_step に基づいて初期選択位置を決定
        selected_label = st.radio(
            "現在の工程を選択",
            step_labels,
            index=st.session_state.get("current_step", 0),
            key="nav_radio"
        )
        
        st.markdown("---")
        with st.expander("⚙️ 設定・トラブルシュート", expanded=False):
            if st.button("🧹 セッションを初期化", use_container_width=True):
                st.session_state.clear()
                st.rerun()
 
    # 0. 戦略設計
    if selected_label.startswith("0"):
        step_0.render()

    # 1. データ収集（QUID連携 & アップロード）
    elif selected_label.startswith("1"):
        step_1.render()

    # 2. 構造化 (AWS Bedrock)
    elif selected_label.startswith("2"):
        if st.session_state.get("raw_data") is not None:
            step_a.render()
        else:
            st.title("🚀 Step 2: 構造化 (AWS Bedrock)")
            st.error("先に Step 1 で分析データをアップロードしてください。")

    # 3. 深層分析 (Interactive)
    elif selected_label.startswith("3"):
        step_b.render()

    # 4. レポート (Output)
    elif selected_label.startswith("4"):
        step_d.render()

# エントリポイント
if __name__ == "__main__":
    main()