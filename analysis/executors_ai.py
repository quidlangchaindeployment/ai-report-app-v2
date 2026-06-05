# analysis/executors_ai.py
import pandas as pd
from typing import Any, Dict

from utils.dependencies import HAS_LLM, get_llm
from analysis.prompts import get_ai_summary_batch_prompt, get_ai_category_insight_prompt
from analysis.config import COLUMN_ALIASES

def run_ai_summary_batch(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    if not HAS_LLM or get_llm is None:
        return _fallback_summary(df)

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.2, timeout_seconds=60)
    if llm is None: return _fallback_summary(df)

    head_rows = df.head(8).to_dict(orient="records")
    colnames = list(df.columns)
    prompt = get_ai_summary_batch_prompt(colnames, head_rows)

    try:
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        return str(raw).strip()
    except Exception:
        return _fallback_summary(df)

def run_ai_category_insight(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    if not HAS_LLM or get_llm is None:
        return _fallback_summary(df)

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.2, timeout_seconds=60)
    if llm is None: return _fallback_summary(df)

    cat_cols = suggestion.get("suitable_cols", {}).get("category_cols", []) if isinstance(suggestion, dict) else []
    if not cat_cols:
        cat_cols = [c for c in df.columns if any(alias in c.lower() for alias in COLUMN_ALIASES["category"])]

    head_rows = df.head(8).to_dict(orient="records")
    prompt = get_ai_category_insight_prompt(cat_cols, head_rows)

    try:
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        return str(raw).strip()
    except Exception:
        return _fallback_summary(df)

def _fallback_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "データが空です。"
    cols = list(df.columns)
    return (
        f"- 全 {len(df):,} 件のデータがあります。\n"
        f"- 列数: {len(cols)}（{', '.join(cols[:8])} ...）\n"
        "- LLM未使用のため簡易サマリです。"
    )