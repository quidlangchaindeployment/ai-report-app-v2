# analysis/proposals.py
# --- 分析手法の提案ロジック -------------------------------------------------------
# 役割：
# - Step B: 「データ構造（列）＋任意のユーザー指示」から実行可能な分析タスク候補を返す
# - Python提案（確実に実行できる基本セット）＋ AI提案（ユーザ指示に基づく追加アイデア）
# - 返却形式（各要素の例）：
#   {
#     "priority": 1,
#     "name": "全体のメトリクス",
#     "description": "・・・（実行関数の説明/目的）",
#     "reason": "・・・（提案理由）",
#     "suitable_cols": [... または dict],
#     "type": "python"  # or "ai"
#   }

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --- 列名検出（未移植でもフォールバックで動作） ---------------------------------
try:
    from utils.cols_detect import find_col, find_cols, find_engagement_cols
except Exception:
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
        for pattern in patterns:
            try:
                for c in cols:
                    if re.search(pattern, c, re.IGNORECASE):
                        found.add(c)
            except re.error:
                continue
        return sorted(list(found))

    def find_engagement_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
        numeric_cols = df.select_dtypes(include=np.number).columns
        found = set()
        for pattern in patterns:
            try:
                for c in numeric_cols:
                    if re.search(pattern, c, re.IGNORECASE):
                        found.add(c)
            except re.error:
                continue
        return sorted(list(found))


# --- LLM（任意）：未移植でも落ちないように ----------------------------------------
_HAS_LLM = False
try:
    from services.llm import get_llm
    _HAS_LLM = True
except Exception:
    get_llm = None


# =============================================================================
# 1) Python 提案（基本セット）
# =============================================================================
def suggest_analysis_techniques_py(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    データフレームの列構成を見て、実行可能な「Pythonタスク」を提案する。
    - 全体のメトリクス（常に）
    - 単純集計（カテゴリ列や ...キーワード があれば）
    - クロス集計（カテゴリ列が複数あれば）
    - 時系列キーワード分析（日時列＋カテゴリ/キーワード列があれば）
    - テキストマイニング（テキスト列があれば）
    - 共起ネットワーク（テキスト列＋代表カテゴリがあれば）
    - 汎用カテゴリ深掘り / 数値列TOP5 / A/B比較の雛形
    """
    suggestions: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return suggestions

    # 1) 基本的な列候補を特定
    text_col = find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])
    location_col = find_col(df, ["市区町村キーワード", "location", "city", "地域"])
    # 日付候補（object列から parse 可能性をチェック）
    date_col = None
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    date_patterns = ["date", "time", "日付", "日時"]
    for c in obj_cols:
        if any(re.search(p, c, re.IGNORECASE) for p in date_patterns):
            try:
                if pd.to_datetime(df[c].dropna().head(5), errors="coerce").notna().any():
                    date_col = c
                    break
            except Exception:
                pass

    # エンゲージメント数値列
    engagement_cols = find_engagement_cols(df, ["eng", "like", "いいね", "エンゲージメント"])

    # カテゴリ類（柔軟）
    base_flag_cols = find_cols(df, ["key", "keyword", "キーワード", "カテゴリ", "topic", "ハッシュタグ"])
    flag_cols = sorted(set([c for c in base_flag_cols if c != location_col]))

    # 2) 提案ロジック

    # 全体メトリクス（常に提案）
    suggestions.append({
        "priority": 1,
        "name": "全体のメトリクス",
        "description": "投稿数、エンゲージメント、センチメント傾向など全体の基本指標を算出します。",
        "reason": "データ全体の状況把握に必須。",
        "suitable_cols": [],
        "type": "python",
    })

    # 単純集計（カテゴリ/キーワード列がある場合）
    for col in flag_cols + ([location_col] if location_col else []):
        suggestions.append({
            "priority": 1,
            "name": f"単純集計: {col}",
            "description": f"「{col}」列の出現頻度（TOP50）を分析します。",
            "reason": f"カテゴリ列（{col}）の基本指標。",
            "suitable_cols": [col],
            "type": "python",
        })

    # クロス集計（カテゴリ列が2つ以上）
    all_categorical = flag_cols + ([location_col] if location_col else [])
    if len(all_categorical) >= 2:
        suggestions.append({
            "priority": 2,
            "name": "クロス集計（カテゴリ間）",
            "description": "2つのカテゴリ列（例: '話題カテゴリ' vs '市区町村'）の組み合わせを集計します。",
            "reason": "カテゴリ間の関係を把握。",
            "suitable_cols": all_categorical,
            "type": "python",
        })

    # 時系列
    if date_col and (flag_cols or location_col):
        suggestions.append({
            "priority": 3,
            "name": "時系列キーワード分析",
            "description": f"{date_col} と任意のキーワード列で、出現数の推移を算出し可視化します。",
            "reason": "季節性・キャンペーン影響などを把握。",
            "suitable_cols": {"datetime": [date_col], "keywords": (flag_cols or [location_col])},
            "type": "python",
        })

    # テキストマイニング
    if text_col:
        suggestions.append({
            "priority": 4,
            "name": "テキストマイニング（頻出単語）",
            "description": f"原文テキスト（{text_col}）から頻出語を抽出し、ワードクラウドを生成します（spaCyがあれば高精度）。",
            "reason": "原文から主要トピックを抽出。",
            "suitable_cols": [text_col],
            "type": "python",
        })

    # 共起ネットワーク（簡易）
    if text_col and (flag_cols or location_col):
        suggestions.append({
            "priority": 4,
            "name": "共起ネットワーク",
            "description": "フィルタ列（例: 市区町村/カテゴリ）で絞った投稿から、単語の共起関係を抽出します（簡易）。",
            "reason": "単語間の関係性を可視化。",
            "suitable_cols": [text_col],
            "type": "python",
        })

    # 汎用カテゴリ深掘り / 数値列TOP5
    if flag_cols and text_col:
        suggestions.append({
            "priority": 4,
            "name": "カテゴリ列の集計と深掘り",
            "description": "指定カテゴリ列の上位カテゴリTOP10について、投稿数や上位キーワードを算出します。",
            "reason": "カテゴリ観点の全体把握。",
            "suitable_cols": {'category_cols': flag_cols, 'text_col': [text_col]},
            "type": "python",
        })

    if flag_cols and engagement_cols:
        suggestions.append({
            "priority": 4,
            "name": "カテゴリ別 数値列TOP5分析",
            "description": f"指定カテゴリ列ごとに、数値列（例: {engagement_cols[0]}）が高いTOP5投稿を抽出します。",
            "reason": "「バズった」投稿の内容把握。",
            "suitable_cols": {'category_cols': flag_cols, 'numeric_cols': engagement_cols},
            "type": "python",
        })

    # A/B 比較（雛形）
    if (flag_cols or location_col):
        suggestions.append({
            "priority": 5,
            "name": "A/B 比較分析",
            "description": "2つのグループ（例：カテゴリA vs B、またはエリアA vs B）を選んで比較します。",
            "reason": "群間差の明確化。",
            "suitable_cols": {'category_cols': (flag_cols + ([location_col] if location_col else []))},
            "type": "python",
        })

    # priority重複の簡易解消（名前で重複除去）
    final: List[Dict[str, Any]] = []
    seen = set()
    for s in sorted(suggestions, key=lambda x: x.get("priority", 99)):
        if s["name"] not in seen:
            final.append(s)
            seen.add(s["name"])
    return final


# =============================================================================
# 2) AI 提案（ユーザー指示ベースの追加）
# =============================================================================
def suggest_analysis_techniques_ai(
    user_prompt: str,
    df: pd.DataFrame,
    existing_suggestions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    ユーザーの自由記述（user_prompt）に基づき、
    既にある Python 提案と重複しない「AIタスク（考察系）」を追加提案する。
    - services.llm.get_llm が使えなければ空リストを返す（フォールバック）
    """
    if not user_prompt or not user_prompt.strip():
        return []
    if not _HAS_LLM or get_llm is None:
        # LLM が未移植/未設定なら AI 提案はなし
        return []

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.1, timeout_seconds=60)
    if llm is None:
        return []

    # 既存名（重複回避）
    existing_names = {s.get("name") for s in (existing_suggestions or [])}
    # 典型的に重複しがちな汎用タスク名を明示的に禁止
    forbidden = {
        "全体のメトリクス", "単純集計", "クロス集計", "時系列キーワード分析", "共起ネットワーク",
        "テキストマイニング（頻出単語）", "カテゴリ列の集計と深掘り", "カテゴリ別 数値列TOP5分析", "A/B 比較分析",
        "市区町村別投稿数集計", "全体のセンチメント分析"
    }
    existing_block = ", ".join(sorted(existing_names.union(forbidden)))

    # 列情報のスニペット（重すぎない範囲で）
    col_info = []
    for c in df.columns[:20]:
        example = None
        try:
            series = df[c].dropna()
            example = str(series.iloc[0])[:60] if not series.empty else "N/A"
        except Exception:
            example = "N/A"
        col_info.append(f"- {c}（型: {str(df[c].dtype)} / 例: {example}）")
    col_info_str = "\n".join(col_info)

    prompt = (
        "あなたはデータ分析の専門家です。ユーザーの『分析指示』と『データ構造』を読み、"
        "実行可能な『AI考察タスク』を JSON リストで提案してください。\n\n"
        "【データ構造（列と例）】\n"
        f"{col_info_str}\n\n"
        "【既に提案済み/禁止タスク（これらは提案しない）】\n"
        f"{existing_block}\n\n"
        "【ユーザーの分析指示】\n"
        f"{user_prompt}\n\n"
        "【出力（厳格なJSON配列）】\n"
        "[\n"
        "  {\n"
        '    "priority": 5,\n'
        '    "name": "（タスク名）",\n'
        '    "description": "（このタスクでAIに実行させる具体指示＝プロンプト）",\n'
        '    "reason": "ユーザー指示に基づく",\n'
        '    "suitable_cols": [],\n'
        '    "type": "ai"\n'
        "  }\n"
        "]\n"
        "説明文や前置きは不要。JSONのみを返してください。"
    )

    try:
        # invoke/predict どちらにも対応
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore
        text = str(raw)

        # JSON抽出（```json ...``` にも対応）
        m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []

        ai_list = json.loads(m.group(1) if (m and m.lastindex) else m.group(0))
        # 最小バリデーション＋重複除去
        out: List[Dict[str, Any]] = []
        for s in ai_list:
            name = s.get("name")
            if not name or name in existing_names or name in forbidden:
                continue
            s["type"] = "ai"
            if "priority" not in s:
                s["priority"] = 5
            out.append(s)
        return out
    except Exception:
        return []