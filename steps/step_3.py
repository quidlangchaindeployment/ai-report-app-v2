# steps/step_3.py
# --- Step 3: 深層分析 (Interactive) ---
import json
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

# --- 共通ユーティリティへの依存 ---
from utils.dependencies import HAS_LLM
# utils.session_manager 側で状態管理されているため、ここでは直接 st.session_state を参照

# --- 分析モジュールのインポート ---
# 冗長な try-except を廃止し、システムとして必要なモジュールは必ず存在するという前提で記述
from analysis.proposals import (
    suggest_analysis_techniques_py,
    suggest_analysis_techniques_ai,
)
from analysis.executors_py import (
    run_overall_metrics, run_simple_count, run_crosstab, run_timeseries,
    run_text_mining, run_cooccurrence_network_pyvis, run_generic_category_summary,
    run_generic_engagement_top5, run_ab_comparison
)
from analysis.executors_ai import run_ai_summary_batch
from analysis.export import convert_results_to_json_string
from utils.io_helpers import read_file

# =============================================================================
# 分析タスクのルーター
# =============================================================================
def _execute_analysis(task_name: str, df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """選択されたタスク名に応じて適切な実行関数を呼び出す"""
    try:
        # Pythonベースの分析
        if task_name == "全体のメトリクス": return run_overall_metrics(df, suggestion)
        if task_name.startswith("単純集計:"): return run_simple_count(df, suggestion)
        if task_name.startswith("クロス集計"): return run_crosstab(df, suggestion)
        if task_name == "時系列キーワード分析": return run_timeseries(df, suggestion)
        if task_name == "テキストマイニング（頻出単語）": return run_text_mining(df, suggestion)
        if task_name == "共起ネットワーク": return run_cooccurrence_network_pyvis(df, suggestion)
        if task_name == "カテゴリ列の集計と深掘り": return run_generic_category_summary(df, suggestion)
        if task_name == "カテゴリ別 数値列TOP5分析": return run_generic_engagement_top5(df, suggestion)
        if task_name == "A/B 比較分析": return run_ab_comparison(df, suggestion)
        
        # AI考察タスク
        if suggestion.get("type") == "ai":
            txt = run_ai_summary_batch(df, suggestion)
            return {"data": txt, "image_base64": None, "summary": (txt or "")[:100] + "..."}
            
        return {"data": None, "image_base64": None, "summary": f"未定義のタスク: {task_name}"}
    except Exception as e:
        return {"data": f"実行エラー: {e}", "image_base64": None, "summary": f"エラー: {e}"}

# =============================================================================
# 画面描画（UI 本体）
# =============================================================================
def render():
    st.title("📊 Step 3: 深層分析 (Interactive)")

    # --- Step 3-1: CSV アップロード ---
    st.header("1) 分析対象データの読み込み")
    st.info("Step 2 で構造化した結果CSV（または同等の形式のデータ）をアップロードしてください。")
    uploaded = st.file_uploader("フラグ付け済みCSVファイル", type=["csv"], key="step_b_uploader")

    if uploaded:
        df, err = read_file(uploaded)
        if err:
            st.error(f"ファイル読み込みエラー: {err}")
            st.session_state["df_flagged_B"] = pd.DataFrame()
            return
        st.session_state["df_flagged_B"] = df
        st.success(f"読込完了: {len(df)} 行")
        with st.expander("データプレビュー（先頭5行）", expanded=True):
            st.dataframe(df.head())
    elif not st.session_state["df_flagged_B"].empty:
        with st.expander("データプレビュー（先頭5行）"):
            st.dataframe(st.session_state["df_flagged_B"].head())
    else:
        st.warning("分析を続けるには CSV をアップロードしてください。")
        return

    df_B = st.session_state["df_flagged_B"]

    # --- Step 3-2: 分析手法の提案 ---
    st.header("2) 分析手法の自動提案")
    user_prompt = st.text_area(
        "（任意）AIに追加で指示したい分析タスクの要望を入力:",
        placeholder="例: QUIDの感情分析スコアと地域別トレンドの相関を見たい。",
        key="step_b_prompt",
    )

    if st.button("💡 データの構造から分析手法を提案させる", type="primary"):
        st.session_state["suggestions_attempted_B"] = True
        st.session_state["step_b_results"] = {}
        st.session_state["step_b_json_output"] = None

        # Pythonベースの基本提案
        base_suggestions = suggest_analysis_techniques_py(df_B)

        # ユーザー指示に基づくAI追加提案 (LLMが有効な場合のみ)
        ai_suggestions = []
        if user_prompt.strip() and HAS_LLM:
            ai_suggestions = suggest_analysis_techniques_ai(user_prompt, df_B, base_suggestions)

        # 重複排除してマージ
        base_names = {s["name"] for s in base_suggestions}
        filtered_ai = [s for s in ai_suggestions if s.get("name") not in base_names]
        all_suggestions = sorted(base_suggestions + filtered_ai, key=lambda x: x.get("priority", 99))

        st.session_state["suggestions_B"] = {s["name"]: s for s in all_suggestions}
        st.session_state["selected_tasks_B"] = set(st.session_state["suggestions_B"].keys())

        st.success(f"分析手法の提案が完了しました（計 {len(all_suggestions)} 件）。以下から実行対象を選んでください。")
        st.rerun()

    if not st.session_state["suggestions_attempted_B"]:
        return

    if not st.session_state["suggestions_B"]:
        st.warning("提案可能な分析タスクがありませんでした。CSVの列構成（カテゴリ列、数値列など）を確認してください。")
        return

    # --- Step 3-3: 実行タスクの選択 ---
    st.header("3) 実行する分析タスクの選択")
    suggestions_B: Dict[str, Dict[str, Any]] = st.session_state["suggestions_B"]
    selected: set[str] = st.session_state["selected_tasks_B"]

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("すべて選択"): selected = set(suggestions_B.keys())
    with col_b:
        if st.button("すべて解除"): selected = set()

    st.markdown("---")
    cols = st.columns(3)
    for i, (task_name, details) in enumerate(sorted(suggestions_B.items(), key=lambda kv: kv[1].get("priority", 99))):
        with cols[i % 3]:
            if st.checkbox(task_name, value=(task_name in selected), help=details.get("description", "")):
                selected.add(task_name)
            else:
                selected.discard(task_name)

    st.session_state["selected_tasks_B"] = selected

    # --- Step 3-4: 一括実行 ---
    st.markdown("---")
    if st.button(f"🏃 選択した {len(selected)} 件の分析を一括実行", type="primary", use_container_width=True):
        if not selected:
            st.warning("実行するタスクが選択されていません。")
        else:
            st.session_state["step_b_results"] = {}
            progress_bar = st.progress(0.0, text="一括実行を開始します...")
            
            for idx, task_name in enumerate(selected, start=1):
                progress_msg = f"({idx}/{len(selected)}) {task_name} を実行中..."
                st.session_state["progress_text"] = progress_msg
                progress_bar.progress(idx / len(selected), text=progress_msg)
                
                suggestion = suggestions_B.get(task_name, {})
                st.session_state["step_b_results"][task_name] = _execute_analysis(task_name, df_B, suggestion)
                
            progress_bar.progress(1.0, text="すべての分析が完了しました！")

    # --- Step 3-5: 結果のプレビュー ---
    if st.session_state["step_b_results"]:
        st.header("4) 分析結果のプレビュー")
        for task_name, details in sorted(suggestions_B.items(), key=lambda kv: kv[1].get("priority", 99)):
            if task_name not in st.session_state["step_b_results"]:
                continue
            
            res = st.session_state["step_b_results"][task_name]
            with st.expander(f"▼ {task_name}", expanded=False):
                st.write(res.get("summary") or "（サマリなし）")
                data = res.get("data")
                if isinstance(data, pd.DataFrame):
                    st.dataframe(data.head(100), use_container_width=True)
                elif isinstance(data, dict):
                    st.json(data)
                elif isinstance(data, str):
                    st.markdown(data)
                
                # Base64画像がある場合の描画
                img_b64 = res.get("image_base64")
                if img_b64:
                    st.image(f"data:image/png;base64,{img_b64}")

    # --- Step 3-6: JSONL へエクスポート ---
    st.header("5) 次のステップ（レポート生成）への引き継ぎ")
    st.info("生成した分析結果をまとめ、レポート作成AI（Step 4）へ渡すためのデータを生成します。")
    if st.button("📄 分析結果を保存して次へ (JSONL生成)", type="primary"):
        try:
            jsonl_str = convert_results_to_json_string(st.session_state["step_b_results"])
            st.session_state["step_b_json_output"] = jsonl_str
            st.success("エクスポート用データの生成が完了しました！")
        except Exception as e:
            st.error(f"データ変換エラー: {e}")

    if st.session_state["step_b_json_output"]:
        st.download_button(
            label="🔽 中間データ(JSONL)をダウンロード",
            data=st.session_state["step_b_json_output"].encode("utf-8"),
            file_name="analysis_output_step3.jsonl",
            mime="application/json",
        )