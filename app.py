import streamlit as st
st.set_page_config(page_title="AI Report App v2", layout="wide")  # ← 必ず最初の st.* & 1回だけ
from steps import step_a, step_b, step_c, step_d


# グローバル初期化（セッションのキーなど）
def _init_state():
    defaults = {
        "log_messages": [],
        "progress_text": "",
        "tips_list": [],
        "current_tip_index": 0,
        "last_tip_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    _init_state()

    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio(
        "Step を選択",
        ["Step A: バッチ抽出", "Step B: 分析", "Step C: レポート", "Step D: PowerPoint"],
        index=0
    )

    if page.startswith("Step A"):
        step_a.render()
    elif page.startswith("Step B"):
        step_b.render()
    elif page.startswith("Step C"):
        step_c.render()
    else:
        step_d.render()

if __name__ == "__main__":
    main()