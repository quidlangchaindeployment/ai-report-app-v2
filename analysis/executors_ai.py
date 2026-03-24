# analysis/executors_ai.py
# --- AI 考察タスク（LLMを利用した分析） ------------------------------------------
# 役割：
# - Step B から呼び出される「AI考察」系タスクを実装
# - services.llm.get_llm が実装されていれば Gemini で本格推論
# - 未実装または APIキー未設定でも落ちないフェイルセーフ

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# LLM（遅延インポート：未移植でも落ちない）
_HAS_LLM = False
try:
    from services.llm import get_llm
    _HAS_LLM = True
except Exception:
    get_llm = None


# ----------------------------------------------------------------------
# AI考察：全体サマリ
# ----------------------------------------------------------------------
def run_ai_summary_batch(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    """
    Step B 用の AI 考察タスク。
    DataFrame 全体を要約し、主な傾向/仮説を文章として返す。

    返り値：
        str（文章）… Step B 側で {"data": <str>, "summary": ...} に組み込む
    """
    # --- 1) LLMが無ければフォールバック -----------------------------------------
    if not _HAS_LLM or get_llm is None:
        return _fallback_summary(df)

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.2, timeout_seconds=60)
    if llm is None:
        return _fallback_summary(df)

    # --- 2) データ圧縮：列名 & 先頭数行だけを LLM に渡す ---------------------------
    head_rows = df.head(8).to_dict(orient="records")
    colnames = list(df.columns)

    prompt = (
        "以下は分析対象データのサンプルです。\n"
        "このデータ全体の傾向を、要点3〜7点の bullet 形式で簡潔にまとめてください。\n"
        "・バズしている投稿傾向\n"
        "・地名/カテゴリの偏り\n"
        "・特性語のパターン\n"
        "など、気づきにつながる仮説も歓迎します。\n\n"
        "【列名】\n"
        f"{json.dumps(colnames, ensure_ascii=False)}\n\n"
        "【先頭サンプル（最大8件）】\n"
        f"{json.dumps(head_rows, ensure_ascii=False)}\n\n"
        "【出力形式】\n"
        "- 箇条書きで簡潔に\n"
        "- 事実と仮説を分けてもよい\n"
        "- JSON ではなくテキスト"
    )

    try:
        # LangChain 互換の invoke/predict の両方に対応
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore

        text = str(raw)
        return text.strip()
    except Exception:
        return _fallback_summary(df)


# ----------------------------------------------------------------------
# AI考察：カテゴリ別深掘り（例として提供）
# ----------------------------------------------------------------------
def run_ai_category_insight(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    """
    任意のカテゴリ列ごとに「そのカテゴリの特徴」「バズ条件」などを考察するタスク。
    Step B の提案ロジックに組み込むと利用可能。
    """
    if not _HAS_LLM or get_llm is None:
        return _fallback_summary(df)

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.2, timeout_seconds=60)
    if llm is None:
        return _fallback_summary(df)

    # suggestion から列情報を取る（なければフォールバック）
    cat_cols = []
    if isinstance(suggestion, dict):
        cat_cols = suggestion.get("suitable_cols", {}).get("category_cols", [])

    if not cat_cols:
        cat_cols = [c for c in df.columns if "カテゴリ" in c or "category" in c.lower()]

    head_rows = df.head(8).to_dict(orient="records")

    prompt = (
        "与えられたデータのカテゴリ列ごとに、特徴的な傾向や仮説を bullet で述べてください。\n"
        "特に『高いエンゲージメントをもたらす要因』『地域差』『テーマ別の強さ』などに注目してください。\n\n"
        f"【対象カテゴリ列】{cat_cols}\n"
        "【先頭サンプル（最大8件）】\n"
        f"{json.dumps(head_rows, ensure_ascii=False)}\n\n"
        "【出力形式】テキスト（箇条書き）"
    )

    try:
        try:
            raw = llm.invoke(prompt)
        except Exception:
            raw = llm.predict(prompt)
        return str(raw).strip()
    except Exception:
        return _fallback_summary(df)


# ----------------------------------------------------------------------
# フォールバック要約（LLM が使えない場合）
# ----------------------------------------------------------------------
def _fallback_summary(df: pd.DataFrame) -> str:
    """
    LLM が未導入/失敗時の簡易サマリ。
    """
    if df is None or df.empty:
        return "データが空です。"

    cols = list(df.columns)
    n_rows = len(df)

    parts = [
        f"- 全 {n_rows:,} 件のデータがあります。",
        f"- 列数: {len(cols)}（{', '.join(cols[:8])} ...）",
        "- LLM未使用のため、簡易サマリを返しています。",
        "- Step B の Python 分析と組み合わせて全体傾向を把握してください。",
    ]
    return "\n".join(parts)