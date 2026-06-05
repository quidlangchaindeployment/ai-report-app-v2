# utils/session_manager.py
# --- セッション状態の統合管理 ---
import streamlit as st
import pandas as pd

def init_session_state():
    """アプリケーション全体で必要なセッション変数を初期化する"""
    defaults = {
        # アプリ全体
        "current_step": 0,
        "log_messages": [],
        
        # Step 0/1 (戦略設計・QUIDデータ)
        "research_design": None,
        "extracted_metrics": None,
        "raw_data": None,
        
        # Step A (Bedrock構造化)
        "current_master_df": pd.DataFrame(),
        "generated_categories": {},
        "exclusion_list": [],
        "analysis_prompt_A": "",
        "cats_editor": "",
        "ex_editor": "",
        "last_job_arn": "",
        
        # Step B (深層分析)
        "df_flagged_B": pd.DataFrame(),
        "suggestions_B": {},
        "selected_tasks_B": set(),
        "step_b_results": {},
        "step_b_json_output": None,
        "suggestions_attempted_B": False,
        "progress_text": "",
        
        # Step C (AIレポート)
        "step_c_report_json": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session():
    """セッションを完全にクリアする"""
    st.session_state.clear()
    init_session_state()