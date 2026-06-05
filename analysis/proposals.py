# analysis/proposals.py
import json
import re
from typing import Any, Dict, List
import pandas as pd

from utils.cols_detect import find_col, find_cols, find_engagement_cols
from utils.dependencies import HAS_LLM, get_llm
from analysis.prompts import get_ai_proposal_prompt
from analysis.config import COLUMN_ALIASES

def suggest_analysis_techniques_py(df: pd.DataFrame) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if df is None or df.empty: return suggestions

    # configのエイリアスを使って対象列を特定する (汎用化)
    text_col = find_col(df, COLUMN_ALIASES["text"])
    location_col = find_col(df, COLUMN_ALIASES["location"])
    engagement_cols = find_engagement_cols(df, COLUMN_ALIASES["engagement"])
    base_flag_cols = find_cols(df, COLUMN_ALIASES["category"])
    
    flag_cols = sorted(set([c for c in base_flag_cols if c != location_col]))

    # 日付候補（object列から parse 可能性をチェック）
    date_col = None
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        if any(re.search(p, c, re.IGNORECASE) for p in COLUMN_ALIASES["date"]):
            try:
                if pd.to_datetime(df[c].dropna().head(5), errors="coerce").notna().any():
                    date_col = c
                    break
            except Exception:
                pass

    # 提案ロジックの構築
    suggestions.append({
        "priority": 1, "name": "全体のメトリクス", "type": "python",
        "description": "投稿数、エンゲージメント、センチメント傾向など全体の基本指標を算出します。",
        "reason": "データ全体の状況把握に必須。", "suitable_cols": []
    })

    for col in flag_cols + ([location_col] if location_col else []):
        suggestions.append({
            "priority": 1, "name": f"単純集計: {col}", "type": "python",
            "description": f"「{col}」列の出現頻度（TOP50）を分析します。",
            "reason": f"カテゴリ列（{col}）の基本指標。", "suitable_cols": [col]
        })

    all_categorical = flag_cols + ([location_col] if location_col else [])
    if len(all_categorical) >= 2:
        suggestions.append({
            "priority": 2, "name": "クロス集計（カテゴリ間）", "type": "python",
            "description": "2つのカテゴリ列の組み合わせを集計します。",
            "reason": "カテゴリ間の関係を把握。", "suitable_cols": all_categorical
        })

    if date_col and (flag_cols or location_col):
        suggestions.append({
            "priority": 3, "name": "時系列キーワード分析", "type": "python",
            "description": f"{date_col} と任意のキーワード列で、出現数の推移を算出し可視化します。",
            "reason": "季節性・キャンペーン影響などを把握。", 
            "suitable_cols": {"datetime": [date_col], "keywords": (flag_cols or [location_col])}
        })

    if text_col:
        suggestions.append({
            "priority": 4, "name": "テキストマイニング（頻出単語）", "type": "python",
            "description": f"原文テキスト（{text_col}）から頻出語を抽出し、ワードクラウドを生成します。",
            "reason": "原文から主要トピックを抽出。", "suitable_cols": [text_col]
        })

    if text_col and (flag_cols or location_col):
        suggestions.append({
            "priority": 4, "name": "共起ネットワーク", "type": "python",
            "description": "フィルタ列で絞った投稿から、単語の共起関係を抽出します。",
            "reason": "単語間の関係性を可視化。", "suitable_cols": [text_col]
        })

    if flag_cols and text_col:
        suggestions.append({
            "priority": 4, "name": "カテゴリ列の集計と深掘り", "type": "python",
            "description": "指定カテゴリ列の上位カテゴリTOP10について、投稿数や上位キーワードを算出します。",
            "reason": "カテゴリ観点の全体把握。", "suitable_cols": {'category_cols': flag_cols, 'text_col': [text_col]}
        })

    if flag_cols and engagement_cols:
        suggestions.append({
            "priority": 4, "name": "カテゴリ別 数値列TOP5分析", "type": "python",
            "description": f"指定カテゴリ列ごとに、数値列が高いTOP5投稿を抽出します。",
            "reason": "「バズった」投稿の内容把握。", "suitable_cols": {'category_cols': flag_cols, 'numeric_cols': engagement_cols}
        })

    if flag_cols or location_col:
        suggestions.append({
            "priority": 5, "name": "A/B 比較分析", "type": "python",
            "description": "2つのグループ（例：カテゴリA vs B、またはエリアA vs B）を選んで比較します。",
            "reason": "群間差の明確化。", "suitable_cols": {'category_cols': (flag_cols + ([location_col] if location_col else []))}
        })

    final: List[Dict[str, Any]] = []
    seen = set()
    for s in sorted(suggestions, key=lambda x: x.get("priority", 99)):
        if s["name"] not in seen:
            final.append(s)
            seen.add(s["name"])
    return final

def suggest_analysis_techniques_ai(user_prompt: str, df: pd.DataFrame, existing_suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not user_prompt or not user_prompt.strip(): return []
    if not HAS_LLM or get_llm is None: return []

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.1, timeout_seconds=60)
    if llm is None: return []

    existing_names = {s.get("name") for s in (existing_suggestions or [])}
    forbidden = {
        "全体のメトリクス", "単純集計", "クロス集計", "時系列キーワード分析", "共起ネットワーク",
        "テキストマイニング（頻出単語）", "カテゴリ列の集計と深掘り", "カテゴリ別 数値列TOP5分析", "A/B 比較分析",
        "市区町村別投稿数集計", "全体のセンチメント分析"
    }
    existing_block = ", ".join(sorted(existing_names.union(forbidden)))

    col_info = []
    for c in df.columns[:20]:
        try:
            series = df[c].dropna()
            example = str(series.iloc[0])[:60] if not series.empty else "N/A"
        except Exception:
            example = "N/A"
        col_info.append(f"- {c}（型: {str(df[c].dtype)} / 例: {example}）")
    col_info_str = "\n".join(col_info)

    prompt = get_ai_proposal_prompt(col_info_str, existing_block, user_prompt)

    try:
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        text = str(raw)
        m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL) or re.search(r"\[.*\]", text, re.DOTALL)
        if not m: return []

        ai_list = json.loads(m.group(1) if (m and m.lastindex) else m.group(0))
        out: List[Dict[str, Any]] = []
        for s in ai_list:
            name = s.get("name")
            if not name or name in existing_names or name in forbidden: continue
            s["type"] = "ai"
            s.setdefault("priority", 5)
            out.append(s)
        return out
    except Exception:
        return []