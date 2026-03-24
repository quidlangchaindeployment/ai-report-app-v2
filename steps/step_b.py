# steps/step_b.py
# --- Step B: インタラクティブ分析（v2） --------------------------------------------
# 役割：
# - キュレーション済みCSVのアップロード（Step A の結果CSVなど）
# - analysis.proposals で分析手法を提案（Python/AI）※未移植なら簡易提案で代替
# - 選択タスクの一括実行（analysis.executors_py / executors_ai → analysis.export で JSONL 化）
# - プレビュー、パラメータ再実行、ダウンロード
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import json
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pandas as pd

# --- 遅延インポート（未移植でも UI が落ちないように） ------------------------------
_HAS_PROPOSALS = False
_HAS_EXEC_PY = False
_HAS_EXEC_AI = False
_HAS_EXPORT = False

# proposals: suggest_analysis_techniques_py / suggest_analysis_techniques_ai
try:
    from analysis.proposals import (
        suggest_analysis_techniques_py,
        suggest_analysis_techniques_ai,
    )
    _HAS_PROPOSALS = True
except Exception:
    suggest_analysis_techniques_py = None
    suggest_analysis_techniques_ai = None

# executors: run_* をまとめて呼ぶルーター execute_analysis をこのファイル内で定義しても可
try:
    from analysis.executors_py import (
        run_overall_metrics,
        run_simple_count,
        run_crosstab,
        run_timeseries,
        run_text_mining,
        run_cooccurrence_network_pyvis,
        run_generic_category_summary,
        run_generic_engagement_top5,
        run_ab_comparison,
    )
    _HAS_EXEC_PY = True
except Exception:
    run_overall_metrics = None
    run_simple_count = None
    run_crosstab = None
    run_timeseries = None
    run_text_mining = None
    run_cooccurrence_network_pyvis = None
    run_generic_category_summary = None
    run_generic_engagement_top5 = None
    run_ab_comparison = None

try:
    from analysis.executors_ai import run_ai_summary_batch
    _HAS_EXEC_AI = True
except Exception:
    run_ai_summary_batch = None

# export: convert_results_to_json_string
try:
    from analysis.export import convert_results_to_json_string
    _HAS_EXPORT = True
except Exception:
    convert_results_to_json_string = None


# --- utils（未移植時のフォールバック付き） ----------------------------------------
try:
    from utils.io_helpers import read_file
except Exception:
    def read_file(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            if file.name.lower().endswith(".csv"):
                return pd.read_csv(file, encoding="utf-8"), None
            elif file.name.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(file), None
            return None, "未対応のファイル形式"
        except Exception as e:
            return None, str(e)

try:
    from utils.cols_detect import find_col, find_cols, find_engagement_cols
except Exception:
    import re
    import numpy as np

    def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        cols = df.columns
        for pattern in patterns:
            try:
                for c in cols:
                    if c.lower() == pattern.lower():
                        return c
                for c in cols:
                    if re.search(pattern, c, re.IGNORECASE):
                        return c
            except re.error:
                continue
        return None

    def find_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
        cols = df.columns
        found = set()
        import re as _re
        for pattern in patterns:
            try:
                for c in cols:
                    if _re.search(pattern, c, _re.IGNORECASE):
                        found.add(c)
            except _re.error:
                continue
        return sorted(list(found))

    def find_engagement_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
        import numpy as _np
        numeric_cols = df.select_dtypes(include=_np.number).columns
        found = set()
        import re as _re
        for pattern in patterns:
            try:
                for c in numeric_cols:
                    if _re.search(pattern, c, _re.IGNORECASE):
                        found.add(c)
            except _re.error:
                continue
        return sorted(list(found))


# --- このファイル内の簡易ルーター（analysis.executors_* が無い時の保険） -----------
def _execute_analysis_local_router(
    analysis_name: str,
    df: pd.DataFrame,
    suggestion: Dict[str, Any],
) -> Dict[str, Any]:
    """
    analysis.executors_* が未移植でも最低限動かすためのローカル・フォールバック。
    既存 executors_* が使える場合は、そちらを優先（render() 内で切替）。
    """
    # 1) 既存の executors_py/ai がある場合は優先
    if _HAS_EXEC_PY or _HAS_EXEC_AI:
        # 本来のルーター（analysis 側）に合わせた分岐
        try:
            if analysis_name == "全体のメトリクス" and run_overall_metrics:
                return run_overall_metrics(df, suggestion)
            elif analysis_name.startswith("単純集計:") and run_simple_count:
                return run_simple_count(df, suggestion)
            elif analysis_name.startswith("クロス集計") and run_crosstab:
                return run_crosstab(df, suggestion)
            elif analysis_name == "時系列キーワード分析" and run_timeseries:
                return run_timeseries(df, suggestion)
            elif analysis_name == "テキストマイニング（頻出単語）" and run_text_mining:
                return run_text_mining(df, suggestion)
            elif analysis_name == "共起ネットワーク" and run_cooccurrence_network_pyvis:
                return run_cooccurrence_network_pyvis(df, suggestion)
            elif analysis_name == "カテゴリ列の集計と深掘り" and run_generic_category_summary:
                return run_generic_category_summary(df, suggestion)
            elif analysis_name == "カテゴリ別 数値列TOP5分析" and run_generic_engagement_top5:
                return run_generic_engagement_top5(df, suggestion)
            elif analysis_name == "A/B 比較分析" and run_ab_comparison:
                return run_ab_comparison(df, suggestion)
            else:
                # AIタスク（考察）
                if suggestion.get("type") == "ai" and run_ai_summary_batch:
                    txt = run_ai_summary_batch(df, suggestion)
                    return {"data": txt, "image_base64": None, "summary": (txt or "")[:100] + "..."}
        except Exception as e:
            return {"data": f"実行エラー: {e}", "image_base64": None, "summary": f"エラー: {e}"}

    # 2) フォールバック（最低限の可視化/要約を返す）
    try:
        if df is None or df.empty:
            return {"data": pd.DataFrame(), "image_base64": None, "summary": "データが空でした。"}
        # 簡易メトリクス
        metrics = {
            "rows": f"{len(df):,}",
            "columns": f"{len(df.columns):,}",
        }
        return {"data": metrics, "image_base64": None, "summary": "簡易メトリクスを返しました。"}
    except Exception as e:
        return {"data": f"実行エラー: {e}", "image_base64": None, "summary": f"エラー: {e}"}


# --- フォールバック提案（proposals 未移植時用の簡易案） ---------------------------
def _fallback_proposals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return suggestions

    text_col = find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])
    location_col = find_col(df, ["市区町村キーワード", "location", "city", "地域"])
    engagement_cols = find_engagement_cols(df, ["eng", "like", "いいね", "エンゲージメント"])

    # 最低限の 3 本
    suggestions.append({
        "priority": 1,
        "name": "全体のメトリクス",
        "description": "投稿総数や簡易傾向を表示します。",
        "reason": "全体像の把握に必須",
        "suitable_cols": [],
        "type": "python",
    })
    if text_col:
        suggestions.append({
            "priority": 3,
            "name": "テキストマイニング（頻出単語）",
            "description": f"テキスト列 '{text_col}' の頻出語を算出します。",
            "reason": "原文の傾向把握",
            "suitable_cols": [text_col],
            "type": "python",
        })
    if location_col and engagement_cols:
        suggestions.append({
            "priority": 4,
            "name": "カテゴリ別 数値列TOP5分析",
            "description": "カテゴリ（例: 市区町村）×エンゲージメント高スコア投稿の概要",
            "reason": "バズ分析",
            "suitable_cols": {"category_cols": [location_col], "numeric_cols": engagement_cols},
            "type": "python",
        })
    return suggestions


# --- JSONL ダウンロードヘルパ -----------------------------------------------
def _download_jsonl(label: str, jsonl_str: str, filename: str):
    st.download_button(
        label,
        data=jsonl_str.encode("utf-8"),
        file_name=filename,
        mime="application/json",
        key=f"dl_{label}_{filename}",
    )


# --- UI 本体 ----------------------------------------------------------------------
def render():
    st.title("📊 Step B: インタラクティブ分析（v2）")

    # セッションステート初期化
    defaults = {
        "df_flagged_B": pd.DataFrame(),      # アップロードしたCSV
        "suggestions_B": {},                 # 提案名 -> 詳細dict
        "selected_tasks_B": set(),           # チェック済みタスク名
        "step_b_results": {},                # 実行結果（タスク名 -> 結果dict）
        "step_b_json_output": None,          # JSONL 文字列（エクスポート用）
        "suggestions_attempted_B": False,    # 提案を実行したか
        "progress_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Step 1: CSV アップロード -------------------------------------------------
    st.header("Step 1: CSV のアップロード")
    st.info("Step A でキュレーションした CSV（または同等の列構成のCSV）をアップロードしてください。")
    uploaded = st.file_uploader("フラグ付け済みCSVファイル", type=["csv"], key="step_b_uploader")

    if uploaded:
        try:
            df, err = read_file(uploaded)
            if err:
                st.error(f"ファイル読み込みエラー: {err}")
                st.session_state["df_flagged_B"] = pd.DataFrame()
                return
            st.session_state["df_flagged_B"] = df
            st.success(f"読込完了: {len(df)} 行")
            with st.expander("データプレビュー（先頭5行）", expanded=True):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"ファイル読み込み中にエラー: {e}")
            st.session_state["df_flagged_B"] = pd.DataFrame()
            return
    else:
        if st.session_state["df_flagged_B"].empty:
            st.warning("分析を続けるには CSV をアップロードしてください。")
            return
        else:
            with st.expander("データプレビュー（先頭5行）"):
                st.dataframe(st.session_state["df_flagged_B"].head())

    df_B = st.session_state["df_flagged_B"]

    # --- Step 2: 分析手法の提案 ---------------------------------------------------
    st.header("Step 2: 分析手法の提案")
    st.caption("(analysis.proposals が未移植の場合は簡易提案で代替します)")

    user_prompt = st.text_area(
        "（任意）AIに追加で指示したい分析タスクを入力:",
        placeholder="例: グルメ投稿と自然投稿の傾向を比較したい。",
        key="step_b_prompt",
    )

    if st.button("💡 分析手法を提案させる", type="primary"):
        st.session_state["suggestions_attempted_B"] = True
        st.session_state["step_b_results"] = {}
        st.session_state["step_b_json_output"] = None

        base_suggestions: List[Dict[str, Any]] = []
        if _HAS_PROPOSALS and suggest_analysis_techniques_py:
            try:
                base_suggestions = suggest_analysis_techniques_py(df_B)  # type: ignore
            except Exception as e:
                st.warning(f"Pythonベース提案の生成に失敗: {e}")
                base_suggestions = _fallback_proposals(df_B)
        else:
            base_suggestions = _fallback_proposals(df_B)

        ai_suggestions: List[Dict[str, Any]] = []
        if user_prompt.strip() and _HAS_PROPOSALS and suggest_analysis_techniques_ai:
            try:
                ai_suggestions = suggest_analysis_techniques_ai(user_prompt, df_B, base_suggestions)  # type: ignore
            except Exception as e:
                st.warning(f"AI追加提案の生成に失敗: {e}")
                ai_suggestions = []

        # 重複排除してマージ
        base_names = {s["name"] for s in base_suggestions}
        filtered_ai = [s for s in ai_suggestions if s.get("name") not in base_names]
        all_suggestions = sorted(base_suggestions + filtered_ai, key=lambda x: x.get("priority", 99))

        st.session_state["suggestions_B"] = {s["name"]: s for s in all_suggestions}
        st.session_state["selected_tasks_B"] = set(st.session_state["suggestions_B"].keys())

        st.success(f"分析手法の提案が完了しました（{len(all_suggestions)} 件）。Step 3 で実行対象を選んでください。")
        st.experimental_rerun()

    if not st.session_state["suggestions_attempted_B"]:
        st.info("上のボタンを押して、分析手法の提案を実行してください。")
        return

    if not st.session_state["suggestions_B"]:
        st.warning("提案が 0 件でした。CSV の列構成を確認してください。")
        return

    # --- Step 3: 実行タスクの選択 -------------------------------------------------
    st.header("Step 3: 実行する分析タスクの選択")
    suggestions_B: Dict[str, Dict[str, Any]] = st.session_state["suggestions_B"]
    selected: set[str] = set(st.session_state["selected_tasks_B"])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("すべて選択"):
            selected = set(suggestions_B.keys())
    with col_b:
        if st.button("すべて解除"):
            selected = set()

    st.markdown("---")
    cols = st.columns(3)
    i = 0
    for task_name, details in sorted(suggestions_B.items(), key=lambda kv: kv[1].get("priority", 99)):
        with cols[i % 3]:
            checked = st.checkbox(task_name, value=(task_name in selected), help=details.get("description", ""))
            if checked:
                selected.add(task_name)
            else:
                if task_name in selected:
                    selected.remove(task_name)
            i += 1

    st.session_state["selected_tasks_B"] = selected

    # --- Step 4: 一括実行 ----------------------------------------------------------
    if st.button(f"🏃 選択した {len(selected)} 件の分析を実行", type="primary", use_container_width=True):
        if len(selected) == 0:
            st.warning("実行するタスクが選択されていません。")
        else:
            st.session_state["step_b_results"] = {}
            progress_bar = st.progress(0.0, text="一括実行を開始します...")
            total = len(selected)
            for idx, task_name in enumerate(selected, start=1):
                st.session_state["progress_text"] = f"({idx}/{total}) {task_name} を実行中..."
                progress_bar.progress(idx / total, text=st.session_state["progress_text"])
                suggestion = suggestions_B.get(task_name, {})
                try:
                    # 既存 executors がある場合はそちらを優先、無ければローカル・フォールバック
                    result = _execute_analysis_local_router(task_name, df_B, suggestion)
                    st.session_state["step_b_results"][task_name] = result
                except Exception as e:
                    st.session_state["step_b_results"][task_name] = {
                        "data": f"一括実行中にエラーが発生しました: {e}",
                        "image_base64": None,
                        "summary": f"エラー: {e}",
                    }
            progress_bar.progress(1.0, text="完了")
            st.success("選択された分析の実行が完了しました。Step 5 で結果を確認してください。")

    # --- Step 5: プレビュー & 個別再実行 ------------------------------------------
    st.header("Step 5: 結果のプレビューとパラメータ再実行")
    if not st.session_state["step_b_results"]:
        st.info("Step 4 で分析を実行すると、ここに結果が表示されます。")
        return

    # 優先度順に表示
    for task_name, details in sorted(suggestions_B.items(), key=lambda kv: kv[1].get("priority", 99)):
        if task_name not in st.session_state["step_b_results"]:
            continue
        res: Dict[str, Any] = st.session_state["step_b_results"][task_name]
        with st.expander(f"▼ {task_name}", expanded=False):
            # サマリ
            st.write(res.get("summary") or "（サマリなし）")

            # DataFrame / dict / str のいずれにも対応
            data = res.get("data")
            if isinstance(data, pd.DataFrame):
                st.dataframe(data.head(100), use_container_width=True)
            elif isinstance(data, dict):
                st.json(data)
            elif isinstance(data, str):
                st.markdown(data)

            # 画像（Base64）や HTML（pyvis）への拡張があればここで差し込み可能

    # --- Step 6: JSONL へエクスポート ---------------------------------------------
    st.header("Step 6: 分析結果を JSONL へエクスポート（Step C 用）")
    if st.button("JSONL を生成", type="primary"):
        results_dict = st.session_state["step_b_results"]
        jsonl_str = None
        if _HAS_EXPORT and convert_results_to_json_string:
            try:
                jsonl_str = convert_results_to_json_string(results_dict)  # type: ignore
            except Exception as e:
                st.error(f"JSONL 変換でエラー: {e}")
                jsonl_str = None
        else:
            # 簡易フォールバック：全結果を JSON にして 1 行で返す
            try:
                jsonl_str = json.dumps({"fallback_results": results_dict}, ensure_ascii=False)
            except Exception as e:
                st.error(f"フォールバックJSON生成エラー: {e}")
                jsonl_str = None

        if jsonl_str:
            st.session_state["step_b_json_output"] = jsonl_str
            st.success("JSONL を生成しました。下のボタンからダウンロードできます。")

    if st.session_state.get("step_b_json_output"):
        _download_jsonl("JSONL をダウンロード", st.session_state["step_b_json_output"], "analysis_output_stepB.jsonl")