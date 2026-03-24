# utils/cols_detect.py
# -----------------------------------------------------------------------------
# 役割：
# - 列名の自動検出ユーティリティ（Step B の提案/実行や各種分析で利用）
# - find_col        : 最初にマッチした 1 列を返す
# - find_cols       : すべてのマッチ列を返す（DFの列順を維持）
# - find_engagement_cols : エンゲージメント系の数値列を返す
#
# 照合優先度：
#   1) 厳密一致（大文字小文字/全半角/空白・記号差を正規化して比較）
#   2) サブストリング一致（正規化後の包含）
#   3) 正規表現一致（パターンに/正規表現/が含まれる or re.Pattern 指定）
#
# パターン指定：
#   - "text" や "本文" のようなプレーン文字列
#   - re.compile(r"...") のような正規表現
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# =============== 正規化ユーティリティ ==============================================

_PUNCT_SPACES = re.compile(r"[\s_\-／／・|｜・、，,\.。／/\\]+")

def _normalize_name(name: str) -> str:
    """
    列名の正規化：
    - 前後空白除去
    - 空白・アンダースコア・記号をまとめて削除
    - 英字は小文字化
    """
    s = str(name).strip()
    s = _PUNCT_SPACES.sub("", s)
    s = s.lower()
    return s


def _is_regex_pattern(p: Union[str, re.Pattern]) -> bool:
    if isinstance(p, re.Pattern):
        return True
    # 文字列が正規表現らしいか（先頭/末尾に /.../ を持つ場合など）
    return bool(isinstance(p, str) and len(p) >= 2 and p.startswith("/") and p.endswith("/"))


def _to_regex(p: Union[str, re.Pattern]) -> re.Pattern:
    if isinstance(p, re.Pattern):
        return p
    if _is_regex_pattern(p):
        return re.compile(p.strip("/"), re.IGNORECASE)
    # プレーン文字列 → 正規表現にエスケープ
    return re.compile(re.escape(p), re.IGNORECASE)


# =============== メインAPI：列検出 =================================================

PatternLike = Union[str, re.Pattern]

def find_col(df: pd.DataFrame, patterns: Sequence[PatternLike]) -> Optional[str]:
    """
    最初にマッチした 1 列を返す（見つからなければ None）。
    優先：厳密一致（正規化）→ 部分一致（正規化）→ 正規表現一致。

    例：
        find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])
    """
    if df is None or df.empty or not patterns:
        return None

    # 事前に正規化マップ作成
    norm_to_original = { _normalize_name(c): c for c in df.columns }

    # 1) 厳密一致（正規化後に完全一致）
    for p in patterns:
        if _is_regex_pattern(p):
            continue
        norm_pat = _normalize_name(str(p))
        for norm_col, orig in norm_to_original.items():
            if norm_col == norm_pat:
                return orig

    # 2) 部分一致（正規化後の包含）
    for p in patterns:
        if _is_regex_pattern(p):
            continue
        norm_pat = _normalize_name(str(p))
        for norm_col, orig in norm_to_original.items():
            if norm_pat and norm_pat in norm_col:
                return orig

    # 3) 正規表現一致（元の列名で評価）
    for p in patterns:
        rgx = _to_regex(p)
        for col in df.columns:
            if rgx.search(str(col)):
                return col

    return None


def find_cols(df: pd.DataFrame, patterns: Sequence[PatternLike]) -> List[str]:
    """
    すべてのマッチ列を **DataFrame の列順で**返す（重複・空を除く）。
    """
    if df is None or df.empty or not patterns:
        return []

    found: List[str] = []
    seen = set()

    # 1) 厳密一致（正規化）
    norm_cols = { _normalize_name(c): c for c in df.columns }
    for p in patterns:
        if _is_regex_pattern(p):
            continue
        norm_pat = _normalize_name(str(p))
        for norm_col, orig in norm_cols.items():
            if norm_col == norm_pat and orig not in seen:
                found.append(orig); seen.add(orig)

    # 2) 部分一致（正規化）
    for p in patterns:
        if _is_regex_pattern(p):
            continue
        norm_pat = _normalize_name(str(p))
        for col in df.columns:
            if col in seen:
                continue
            if norm_pat and norm_pat in _normalize_name(col):
                found.append(col); seen.add(col)

    # 3) 正規表現一致（元の列名）
    for p in patterns:
        rgx = _to_regex(p)
        for col in df.columns:
            if col in seen:
                continue
            if rgx.search(str(col)):
                found.append(col); seen.add(col)

    return found


# =============== エンゲージメント列の検出 =========================================

# よくあるエンゲージメント指標のキーワード（数値列を想定）
_DEFAULT_ENG_PATTERNS: List[PatternLike] = [
    "engagement", "エンゲージメント",
    "like", "いいね", "お気に入り", "fav", "favs",
    "retweet", "rt", "リツイート", "repost", "シェア", "share",
    "reply", "replies", "コメント", "comment",
    "view", "views", "impression", "impressions", "インプレッション", "表示",
    # 指標系
    "/score|スコア|rating|評価|反応度|反応数|クリック|clicks?/",
]

def find_engagement_cols(df: pd.DataFrame, patterns: Optional[Sequence[PatternLike]] = None) -> List[str]:
    """
    「エンゲージメント系の数値列」を検出して返す（列順維持）。
    patterns 未指定時は _DEFAULT_ENG_PATTERNS を使用。
    - 数値型（または数値に変換可能）を優先
    """
    if df is None or df.empty:
        return []

    pats = list(patterns) if patterns else list(_DEFAULT_ENG_PATTERNS)

    # まずは名前で候補抽出
    name_matched = find_cols(df, pats)
    if not name_matched:
        return []

    # 数値列を優先（float/int または 数値に変換可能性が高い列）
    numeric_cols = set(df.select_dtypes(include=np.number).columns.tolist())
    ordered_numeric = [c for c in name_matched if c in numeric_cols]
    # それ以外：数値への変換が多く成功する列を追加（軽く判定）
    others = [c for c in name_matched if c not in numeric_cols]

    def _is_mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
        try:
            s = pd.to_numeric(series, errors="coerce")
            ratio = float(s.notna().mean())
            return ratio >= threshold
        except Exception:
            return False

    mostly_numeric = [c for c in others if _is_mostly_numeric(df[c])]
    return ordered_numeric + mostly_numeric


# =============== おまけ：便利候補検出（任意で利用可能） ===========================

def guess_text_col(df: pd.DataFrame) -> Optional[str]:
    """
    それっぽいテキスト列を推定する（任意の補助関数）。
    - 代表候補: "ANALYSIS_TEXT_COLUMN", "text", "content", "本文", "caption", "body"
    - object かつ 平均文字長がある程度以上
    """
    candidates = find_cols(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文", "caption", "body"])
    for c in candidates:
        try:
            if df[c].dtype == "object":
                s = df[c].dropna().astype(str)
                if not s.empty and s.str.len().mean() >= 20:
                    return c
        except Exception:
            continue
    return candidates[0] if candidates else None


def guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """
    それっぽい日時列を推定する（任意の補助関数）。
    - 代表候補: "date", "日時", "created", "time", "posted_at", "published_at"
    """
    cands = find_cols(df, ["date", "日時", "created", "time", "posted_at", "published_at"])
    for c in cands:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return c
        except Exception:
            continue
    return cands[0] if cands else None


__all__ = [
    "find_col",
    "find_cols",
    "find_engagement_cols",
    "guess_text_col",
    "guess_datetime_col",
]