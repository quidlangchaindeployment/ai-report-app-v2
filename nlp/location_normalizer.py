# nlp/location_normalizer.py
# -----------------------------------------------------------------------------
# 役割：
# - 地名辞書（例：JAPAN_GEOGRAPHY_DB = {"広島市": ["中区", "南区", ...], "栃木県": ["宇都宮市", ...], ...}）から
#   1) エイリアス辞書（"中" -> "中区" / "中区" -> "広島市 中区" など）
#   2) 曖昧語セット（"広島" / "東京" / "中央" など市/区/都道府県を取り除いた語や短すぎる区名）
#   3) 既知の正式地名セット（"広島市" / "中区" / "宇都宮市" など）
#   を生成し、地名推論/正規化をサポートします。
#
# - Streamlit 未導入でも動くように実装（存在すれば @st.cache_data を適用）。
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional
import re

# （任意）地名辞書。アプリ側で from geography_db import JAPAN_GEOGRAPHY_DB を使う前提なので、
# ここでは参照しません（関数引数として受け取ります）。

# Streamlit のキャッシュは存在時のみ利用する
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


# ============ 内部ユーティリティ ===================================================

_KEN_TO_FU_TO_SUFFIX = ("県", "都", "府", "道")
_SHI_SUFFIX = "市"
_KU_SUFFIX = "区"
_CHO_SUFFIX = "町"
_SON_SUFFIX = "村"

_SUFFIXES = _KEN_TO_FU_TO_SUFFIX + (_SHI_SUFFIX, _KU_SUFFIX, _CHO_SUFFIX, _SON_SUFFIX)


def _strip_suffix(name: str) -> str:
    """末尾の行政区画接尾辞を 1 つだけ取り除く（例：'広島市' -> '広島'、'中区' -> '中'）。"""
    for suf in _SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _normalize_alias(raw: str) -> str:
    """
    汎用エイリアスを作る：
    - 市/区/町/村 のサフィックスを落とす
    - 全角/半角の空白や記号を軽く正規化
    """
    s = str(raw).strip()
    s = re.sub(r"\s+", "", s)  # 空白除去
    # 一旦サフィックスを除去
    s = _strip_suffix(s)
    return s


def _add_prompt_bias(
    db: Dict[str, List[str]],
    analysis_prompt_str: str,
    alias_to_city_map: Dict[str, str],
    ambiguous_keys: Set[str],
) -> None:
    """
    分析指針に含まれる政令市名があれば、その市の区エイリアスを
    より強く「<市> <区>」の形にマッピングして誤判定を減らす。
    """
    prompt_lower = analysis_prompt_str.lower()
    for city_key, wards in db.items():
        if not isinstance(wards, list) or not wards:
            continue
        # 例： "広島市" -> "広島" を含むか
        if _SHI_SUFFIX in city_key and _KU_SUFFIX in wards[0]:
            base = city_key.replace(_SHI_SUFFIX, "")
            if base and base in prompt_lower:
                for ward in wards:
                    ward_alias = ward.replace(_KU_SUFFIX, "")
                    # 「中」など短すぎる語は曖昧→ただし強制的にこの市の区に紐づけておく
                    alias_to_city_map[ward] = f"{city_key} {ward}"
                    alias_to_city_map[ward_alias] = f"{city_key} {ward}"
                    ambiguous_keys.discard(ward_alias)  # この市にバイアス


# ============ 1) 地名正規化辞書の生成 ==============================================

def _cache_data(ttl: int = 3600):
    """streamlit.cache_data 相当（Streamlit が無ければ no-op デコレータ）。"""
    if _HAS_ST and hasattr(st, "cache_data"):
        return st.cache_data(ttl=ttl)  # type: ignore
    # no-op
    def _decorator(fn):
        return fn
    return _decorator


@_cache_data(ttl=3600)
def get_location_normalization_maps(
    db: Dict[str, List[str]],
    analysis_prompt_str: str,
) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """
    地名辞書から「正規化」に使う 3 つの構造を生成する。

    返り値:
        alias_to_city_map : Dict[str, str]
          - エイリアス（接尾辞を落としたもの、または短縮形）→ 正式地名（"市/区/..." を含む）
          - 例: "中" -> "中区", "中区" -> "広島市 中区", "尾道" -> "尾道市"
        ambiguous_keys    : Set[str]
          - 単独では曖昧で採用しにくい語（"広島" / "東京" / "中" など）
        all_cities_wards  : Set[str]
          - 正式な市/区/町/村のフル表記（照合用）

    備考:
      - analysis_prompt_str（分析指針）に含まれる政令市名を優先して、区エイリアスの曖昧さを解消。
    """
    if not db:
        return {}, set(), set()

    alias_to_city_map: Dict[str, str] = {}
    ambiguous_keys: Set[str] = set()
    prefectures: Set[str] = set()
    all_cities_wards: Set[str] = set()

    # 1) DB全体をスキャン
    for key, values in db.items():
        if not isinstance(values, list):
            continue

        # 1a. キー側の処理（都道府県 or 政令市など）
        key_normalized = _normalize_alias(key)
        if any(suf in key for suf in _KEN_TO_FU_TO_SUFFIX):  # 都道府県
            prefectures.add(key)
            ambiguous_keys.add(key_normalized)  # "東京" など
        elif _SHI_SUFFIX in key and values and _KU_SUFFIX in values[0]:  # 政令市
            ambiguous_keys.add(key_normalized)  # "広島" "札幌" などベース名は曖昧
            all_cities_wards.add(key)          # "広島市"
        else:
            # ふつうの「○○市」など
            all_cities_wards.add(key)
            # 「尾道」→「尾道市」 のエイリアス救済
            alias = _normalize_alias(key)
            if alias and alias != key:
                # 他の同名が無ければ採用、衝突時は曖昧へ
                if alias not in alias_to_city_map:
                    alias_to_city_map[alias] = key
                elif alias_to_city_map.get(alias) != key:
                    alias_to_city_map.pop(alias, None)
                    ambiguous_keys.add(alias)

        # 1b. 値リスト（市/区/町/村）の処理
        for city_or_ward in values:
            all_cities_wards.add(city_or_ward)  # "函館市", "中央区" など
            alias = _normalize_alias(city_or_ward)  # "中央"
            # 短すぎるエイリアス（例："中"）は曖昧扱い
            if _KU_SUFFIX in city_or_ward and len(alias) <= 2:
                ambiguous_keys.add(alias)
            else:
                if alias and alias != city_or_ward:
                    if alias not in alias_to_city_map:
                        alias_to_city_map[alias] = city_or_ward
                    elif alias_to_city_map.get(alias) != city_or_ward:
                        # 衝突 → 曖昧へ
                        alias_to_city_map.pop(alias, None)
                        ambiguous_keys.add(alias)

    # 2) 分析指針に含まれる政令市の Ward を強化（バイアス付け）
    _add_prompt_bias(db, analysis_prompt_str, alias_to_city_map, ambiguous_keys)

    # 3) 曖昧キーと都道府県をまとめて「曖昧集合」に
    final_ambiguous = ambiguous_keys.union(prefectures)

    return alias_to_city_map, final_ambiguous, all_cities_wards


# ============ 2) 単語の曖昧性チェック ===============================================

def is_ambiguous_term(term: str, ambiguous_keys: Set[str]) -> bool:
    """
    語（接尾辞を取った形でも可）が「曖昧集合」に含まれるかを判定。
    """
    if not term:
        return True
    base = _normalize_alias(term)
    return (term in ambiguous_keys) or (base in ambiguous_keys)


# ============ 3) 単語 -> 正式地名への正規化 =========================================

def normalize_keyword(
    keyword: str,
    alias_to_city_map: Dict[str, str],
    ambiguous_keys: Set[str],
    all_cities_wards: Set[str],
) -> str:
    """
    キーワード（"広島" / "中" / "中区" / "尾道" など）を正式地名に正規化して返す。
    返り値が空文字のときは「正規化不能 or 曖昧」と判断。

    優先度：
      1) そのまま正式地名に一致（"○○市" "○○区" ...）
      2) エイリアス辞書で解決（"尾道" -> "尾道市", "中" -> "中区" or "広島市 中区"）
      3) 曖昧語なら棄却
      4) "市/区" を含む複合（"札幌市 中央区" のような形）は許容
    """
    if not keyword or not keyword.strip():
        return ""

    k = keyword.strip()
    # 1) 完全一致（正式地名セット）
    if k in all_cities_wards:
        return k

    # 2) エイリアス解決（"中" -> "中区"、"尾道" -> "尾道市" など）
    base = _normalize_alias(k)
    if base in alias_to_city_map:
        return alias_to_city_map[base]

    # 3) 曖昧語は棄却
    if is_ambiguous_term(k, ambiguous_keys):
        return ""

    # 4) "札幌市 中央区" などはそのまま許容（市/区が含まれる）
    if " " in k and (("市" in k) or ("区" in k) or ("町" in k) or ("村" in k)):
        return k

    return ""


# ============ 4) テキストからの簡易抽出 ============================================

def detect_locations_in_text(
    text: str,
    alias_to_city_map: Dict[str, str],
    ambiguous_keys: Set[str],
    all_cities_wards: Set[str],
    top_k: int = 3,
) -> List[str]:
    """
    テキストから「市/区/町/村」を含む語、またはエイリアス語を検出して正規化。
    - まず「正式表記」を優先的に拾い、見つからない場合はエイリアスで補う
    - 曖昧語は自動除外
    - 重複を排除して上位 top_k を返す
    """
    if not text:
        return []

    candidates: List[str] = []

    # 1) 正式表記（市/区/町/村が末尾）の拾い上げ
    # 例："広島市", "中区", "那須町", "名寄市"
    for m in re.finditer(r"[一-龥ぁ-んァ-ンA-Za-z0-9]+(?:市|区|町|村)", text):
        word = m.group(0)
        if word in all_cities_wards:
            candidates.append(word)
        else:
            # "○○市○○区" のような連結語を分割して試す
            if "市" in word and "区" in word:
                parts = re.split(r"(市|区)", word)
                # 例: ["札幌", "市", "中央", "区", ""]
                try:
                    shi = "".join(parts[:2])         # 札幌市
                    ku = "".join(parts[2:4])         # 中央区
                    if shi in all_cities_wards and ku in all_cities_wards:
                        candidates.append(f"{shi} {ku}")
                    elif shi in all_cities_wards:
                        candidates.append(shi)
                    elif ku in all_cities_wards:
                        candidates.append(ku)
                except Exception:
                    pass

    # 2) エイリアス候補（"広島", "尾道", "中" など；短語は曖昧排除）
    for m in re.finditer(r"[一-龥ぁ-んァ-ンA-Za-z0-9]+", text):
        base = _normalize_alias(m.group(0))
        if not base or len(base) < 2:
            continue
        if base in alias_to_city_map and (not is_ambiguous_term(base, ambiguous_keys)):
            normalized = alias_to_city_map[base]
            if normalized:
                candidates.append(normalized)

    # 重複排除 & 上位抽出
    uniq: List[str] = []
    for c in candidates:
        if c and c not in uniq:
            uniq.append(c)
        if len(uniq) >= top_k:
            break
    return uniq