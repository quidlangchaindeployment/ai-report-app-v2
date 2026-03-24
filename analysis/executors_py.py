# analysis/executors_py.py
# --- Pythonベース分析 実装集 -------------------------------------------------------
# 役割：
# - Step B から呼ばれる各種 Python 分析関数を提供
# - 返却形式は {"data": <pd.DataFrame|dict|str>, "image_base64": <str|None>, "summary": <str>} に統一
# - 依存（spaCy, plotting, cols_detect）が未移植でもフォールバックで動作

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- utils: 列検出 ---------------------------------------------------------------
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

# --- utils: グラフ描画（Base64） ---------------------------------------------------
try:
    from utils.plotting import generate_graph_image
except Exception:
    # 画像が未対応でも落ちないよう、ダミー関数を用意
    def generate_graph_image(
        df: pd.DataFrame, plot_type: str, x_col: Optional[str] = None, y_col: Optional[str] = None, title: str = ""
    ) -> Optional[str]:
        return None

# --- NLP: spaCy ローダ ------------------------------------------------------------
try:
    from nlp.spacy_loader import load_spacy_model
except Exception:
    def load_spacy_model():
        return None


# ================================================================================
# 1) 全体メトリクス
# ================================================================================
def run_overall_metrics(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    データセット全体の基本メトリクスを返す。
    - 総件数
    - 総エンゲージメント（候補列の合計）
    - センチメント列（あれば）から簡易傾向
    """
    metrics: Dict[str, Any] = {}
    try:
        metrics["total_posts"] = f"{len(df):,}件"

        # エンゲージメント候補
        eng_cols = [c for c in df.columns if any(k in c.lower() for k in ["いいね", "like", "engagement", "エンゲージメント", "retweet", "リツイート"])]
        total_eng = 0
        for c in eng_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                total_eng += int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        metrics["total_engagement"] = f"{total_eng:,}件" if eng_cols else "N/A"

        # センチメント（あれば）
        sent_col = None
        if "センチメント" in df.columns:
            sent_col = "センチメント"
        else:
            alt = find_col(df, ["sent", "センチメント"])
            if alt:
                sent_col = alt
        if sent_col:
            s = df[sent_col].astype(str)
            pos = int(s.str.contains("ポジティブ|Positive", case=False, na=False).sum())
            neg = int(s.str.contains("ネガティブ|Negative", case=False, na=False).sum())
            metrics["positive_posts"] = f"{pos:,}件"
            metrics["negative_posts"] = f"{neg:,}件"
            tot = pos + neg
            metrics["sentiment_tendency_percent"] = f"{int(((pos - neg) / tot) * 100) if tot else 0}%"
        else:
            metrics["positive_posts"] = "N/A"
            metrics["negative_posts"] = "N/A"
            metrics["sentiment_tendency_percent"] = "N/A"

        summary = f"全体のメトリクスを計算。総投稿数: {metrics['total_posts']}, 総エンゲージメント: {metrics['total_engagement']}。"
        return {"data": metrics, "image_base64": None, "summary": summary}
    except Exception as e:
        return {"data": {"error": str(e)}, "image_base64": None, "summary": f"エラー: {e}"}


# ================================================================================
# 2) 単純集計（頻度）
# ================================================================================
def run_simple_count(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    カテゴリ列（カンマ区切り想定）を頻度集計して TOP20 の棒グラフを返す。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    flag_cols = suggestion.get("suitable_cols", [])
    if not flag_cols:
        results["summary"] = "集計対象の列が見つかりません。"
        return results

    col = suggestion.get("ui_selected_col", flag_cols[0])
    if col not in df.columns:
        results["summary"] = f"列 '{col}' がDFに存在しません。"
        return results

    try:
        s = df[col].astype(str).str.split(",").explode().str.strip()
        s = s[~s.isin(["", "nan", "None", "N/A", "該当なし"])]
        if s.empty:
            results["summary"] = "集計対象のキーワードがありませんでした。"
            return results

        counts = s.value_counts().head(50)
        counts_df = counts.reset_index()
        counts_df.columns = [col, "count"]
        results["data"] = counts_df
        results["image_base64"] = generate_graph_image(
            df=counts_df, plot_type="bar", x_col=col, y_col="count", title=f"「{col}」 頻出TOP20"
        )
        top2 = ", ".join([f"{counts_df.iloc[i,0]} ({counts_df.iloc[i,1]}件)" for i in range(min(2, len(counts_df)))])
        results["summary"] = f"「{col}」の単純集計を実行。上位: {top2}。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 3) クロス集計
# ================================================================================
def run_crosstab(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    2 つのカテゴリ列のクロス集計（出現回数）を返す。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    cols = suggestion.get("suitable_cols", [])
    if len(cols) < 2:
        results["summary"] = "クロス集計には2列以上必要です。"
        return results

    col1 = suggestion.get("ui_selected_col1", cols[0])
    col2 = suggestion.get("ui_selected_col2", cols[1])
    if col1 not in df.columns or col2 not in df.columns:
        results["summary"] = f"選択された列 ({col1}, {col2}) がDFに存在しません。"
        return results

    try:
        df1 = df.assign(**{col1: df[col1].astype(str).str.split(",")}).explode(col1)
        df2 = df1.assign(**{col2: df1[col2].astype(str).str.split(",")}).explode(col2)
        df2[col1] = df2[col1].str.strip()
        df2[col2] = df2[col2].str.strip()
        df2 = df2.replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna(subset=[col1, col2])

        ct = pd.crosstab(df2[col1], df2[col2])
        long_df = ct.stack().reset_index()
        long_df.columns = [col1, col2, "count"]
        long_df = long_df[long_df["count"] > 0].sort_values(by="count", ascending=False)
        results["data"] = long_df.head(100)
        results["summary"] = f"「{col1}」×「{col2}」のクロス集計を実行。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 4) 時系列キーワード分析
# ================================================================================
def run_timeseries(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    日時列 × キーワード列（カンマ区切り）で日次推移を作成し、上位キーワードの折れ線を返す。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    cols_dict = suggestion.get("suitable_cols", {})
    if not isinstance(cols_dict, dict) or "datetime" not in cols_dict or "keywords" not in cols_dict:
        results["summary"] = "列情報（datetime, keywords）が不十分です。"
        return results

    dt_col = suggestion.get("ui_selected_dt_col", cols_dict["datetime"][0])
    kw_col = suggestion.get("ui_selected_kw_col", cols_dict["keywords"][0])
    if dt_col not in df.columns:
        results["summary"] = f"日時列 '{dt_col}' が見つかりません。"
        return results
    if kw_col not in df.columns:
        results["summary"] = f"キーワード列 '{kw_col}' が見つかりません。"
        return results

    try:
        work = df[[dt_col, kw_col]].copy()
        work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
        work = work.dropna(subset=[dt_col])

        ex = work.assign(**{kw_col: work[kw_col].astype(str).str.split(",")}).explode(kw_col)
        ex[kw_col] = ex[kw_col].str.strip()
        ex = ex[~ex[kw_col].isin(["", "nan", "None", "N/A", "該当なし"])]

        if ex.empty:
            results["summary"] = "有効な日時/キーワードデータがありません。"
            return results

        ts = ex.groupby([pd.Grouper(key=dt_col, freq="D"), kw_col]).size().rename("count").reset_index()
        ts.columns = ["date", "keyword", "count"]

        # 上位キーワードのみに絞る（視認性）
        top_keywords = ex[kw_col].value_counts().head(50).index
        ts_f = ts[ts["keyword"].isin(top_keywords)]

        # 表示用（JSON/CSV連携）
        data_for_json = ts_f.sort_values(by=["keyword", "date"]).copy()
        data_for_json["date"] = data_for_json["date"].dt.strftime("%Y-%m-%d")
        results["data"] = data_for_json

        # グラフ（Base64）
        results["image_base64"] = generate_graph_image(
            df=ts_f, plot_type="timeseries", x_col="date", y_col="count", title=f"「{kw_col}」別 時系列トレンド (TOP5)"
        )
        results["summary"] = f"『{dt_col}』×『{kw_col}』の時系列分析を実行。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 5) テキストマイニング（頻出語）
# ================================================================================
def run_text_mining(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    spaCy（ja_core_news_sm）があれば名詞/固有名詞/形容詞などから頻出語TOP100を抽出し、ワードクラウド画像も返す。
    spaCy が未導入でも簡易トークナイズで頻出語を返す。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    text_col = suggestion.get("ui_selected_text_col", suggestion.get("suitable_cols", ["ANALYSIS_TEXT_COLUMN"])[0])
    if text_col not in df.columns or df[text_col].empty:
        results["summary"] = f"テキスト列 '{text_col}' がないか、空です。"
        return results

    nlp = load_spacy_model()
    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            results["summary"] = "テキストデータが空です。"
            return results

        words: List[str] = []
        if nlp is not None:
            target_pos = {"NOUN", "PROPN", "ADJ"}
            stop_words = {"の", "に", "は", "を", "が", "で", "て", "です", "ます", "こと", "もの", "それ", "これ", "ため", "いる",
                          "する", "ある", "ない", "いう", "よう", "など"}
            for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=50):
                for t in doc:
                    if (t.pos_ in target_pos) and (not t.is_stop) and (t.lemma_ not in stop_words) and (len(t.lemma_) > 1):
                        words.append(t.lemma_)
        else:
            # フォールバック：空白/句読点で分割、短語/汎用ストップ語を除去
            sw = {"の", "に", "は", "を", "が", "で", "て", "です", "ます", "こと", "もの", "それ", "これ", "ため"}
            for tx in texts:
                for w in re.split(r"[、。\s,./!?:;（）()「」【】『』\[\]\-]+", tx):
                    w = w.strip()
                    if len(w) > 1 and w not in sw:
                        words.append(w)

        if not words:
            results["summary"] = "抽出可能な単語が見つかりませんでした。"
            return results

        counts = pd.Series(words).value_counts().head(100)
        wc_df = counts.reset_index()
        wc_df.columns = ["word", "count"]
        results["data"] = wc_df
        results["image_base64"] = generate_graph_image(
            df=wc_df, plot_type="wordcloud", title=f"「{text_col}」 頻出単語 ワードクラウド (TOP100)"
        )
        results["summary"] = f"『{text_col}』に対するテキストマイニングを実行。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 6) 共起ネットワーク（pyvis HTML は Step B 側で扱う想定 / ここでは表データのみ）
# ================================================================================
def run_cooccurrence_network_pyvis(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    フィルタ列（カテゴリ）で絞り込み → 文書内の単語共起を算出 → 上位エッジリストを返す。
    * pyvis HTML の生成自体は utils.plotting では扱わないため、ここではエッジ表のみ返却。
    """
    # 依存が重いのでフォールバック的に実装（簡易）
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    flag_col = suggestion.get("ui_selected_flag_col") or find_col(df, ["市区町村キーワード", "location", "city", "地域"]) \
               or find_col(df, ["話題カテゴリ", "topic", "category"])
    text_col = suggestion.get("ui_selected_text_col") or find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])

    if not flag_col or not text_col or flag_col not in df.columns or text_col not in df.columns:
        results["summary"] = "対象列が見つかりません（フィルタ列/テキスト列）。"
        return results

    try:
        # デフォルトで Top10 キーワード抽出
        s = df[flag_col].dropna().astype(str).str.split(",").explode().str.strip()
        s = s[~s.isin(["", "nan", "None", "N/A"])]
        selected = s.value_counts().index.tolist()[:10]
        if not selected:
            results["summary"] = "絞り込みキーワードが取得できません。"
            return results

        # 選択語を含む投稿のみ
        patt = "|".join([re.escape(k) for k in selected])
        df_f = df[df[flag_col].astype(str).str.contains(patt, na=False)]
        texts = df_f[text_col].dropna().astype(str)

        # 超簡易共起：1文書内のユニーク語集合から組合せをカウント（名詞/形容詞のみ抽出できればより良い）
        def tokenize(t: str) -> List[str]:
            return [w for w in re.split(r"[、。\s,./!?:;（）()「」【】『』\[\]\-]+", t) if len(w) > 1][:80]

        from collections import Counter
        from itertools import combinations

        edge_counter = Counter()
        for tx in texts:
            ws = set(tokenize(tx))
            for a, b in combinations(sorted(ws), 2):
                edge_counter[(a, b)] += 1

        if not edge_counter:
            results["summary"] = "共起エッジが得られませんでした。"
            return results

        # 上位100エッジ
        rows = [{"source": s, "target": t, "weight": w} for (s, t), w in edge_counter.most_common(100)]
        edge_df = pd.DataFrame(rows)
        results["data"] = edge_df
        results["summary"] = f"共起ネットワーク（簡易）を生成。{len(edge_df)} エッジ。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 7) 汎用：カテゴリ列ごとの深掘り
# ================================================================================
def run_generic_category_summary(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    指定カテゴリ列の上位カテゴリ TOP10 について、投稿数と上位キーワード（...キーワード列）を算出し、
    AIサマリ（別タスク）と組み合わせて使いやすくするための表を返す。
    ここでは Python 側で投稿数と上位キーワードまで。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    topic_col = suggestion.get("ui_selected_category_col") or find_col(df, ["話題カテゴリ", "topic", "category"])
    text_col = find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])
    if not topic_col or not text_col:
        results["summary"] = "分析に必要な列（カテゴリ列/テキスト列）が見つかりません。"
        return results
    if topic_col not in df.columns or text_col not in df.columns:
        results["summary"] = f"指定列がDFに存在しません: '{topic_col}', '{text_col}'"
        return results

    try:
        s = df[topic_col].astype(str).str.split(", ").explode().str.strip()
        s = s[~s.isin(["", "nan", "None", "N/A", "該当なし"])]
        if s.empty:
            results["summary"] = f"カテゴリ列 '{topic_col}' に有効なデータがありません。"
            return results
        targets = s.value_counts().head(10).index.tolist()

        flag_cols = [c for c in df.columns if c.endswith("キーワード")]
        loc_col = find_col(df, ["市区町村キーワード", "location", "city", "地域"])
        cols_for_kw = [c for c in flag_cols if c not in {loc_col, topic_col}]

        rows = []
        for cat in targets:
            df_f = df[df[topic_col].astype(str).str.contains(re.escape(cat), na=False)]
            post_count = len(df_f)
            top_keywords: List[str] = []
            if cols_for_kw:
                ss = []
                for c in cols_for_kw:
                    s_ = df_f[c].astype(str).str.split(", ").explode().str.strip()
                    s_ = s_[~s_.isin(["", "nan", "None", "N/A", "該当なし"])]
                    if not s_.empty:
                        ss.append(s_)
                if ss:
                    comb = pd.concat(ss)
                    top_keywords = comb.value_counts().head(5).index.tolist()
            rows.append({"category": cat, "post_count": post_count, "top_keywords": top_keywords})

        out_df = pd.DataFrame(rows)
        img = generate_graph_image(
            df=out_df, plot_type="bar", x_col="category", y_col="post_count", title=f"「{topic_col}」別 投稿数 (Top 10)"
        )
        results["data"] = out_df
        results["image_base64"] = img
        results["summary"] = f"「{topic_col}」別の投稿数と上位キーワード（簡易）を算出。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 8) 汎用：カテゴリ別 数値列TOP5（バズ投稿）
# ================================================================================
def run_generic_engagement_top5(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    指定カテゴリ列ごとに、指定の数値列（エンゲージメント等）が高い TOP5 投稿を抽出し、
    概要（テキストの冒頭）と数値を表で返す。
    """
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    topic_col = suggestion.get("ui_selected_category_col") or find_col(df, ["話題カテゴリ", "topic", "category"])
    text_col = suggestion.get("ui_selected_text_col") or find_col(df, ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文"])
    eng_col = suggestion.get("ui_selected_numeric_col") or (find_engagement_cols(df, ["eng", "like", "いいね", "エンゲージメント"]) or [None])[0]

    if not topic_col or not text_col or not eng_col:
        results["summary"] = "カテゴリ列/テキスト列/数値列 が見つかりません。"
        return results
    if topic_col not in df.columns or text_col not in df.columns or eng_col not in df.columns:
        results["summary"] = "指定された列がDFに存在しません。"
        return results
    if not pd.api.types.is_numeric_dtype(df[eng_col]):
        results["summary"] = f"数値列 '{eng_col}' が数値として扱えません。"
        return results

    try:
        ex = df.assign(**{topic_col: df[topic_col].astype(str).str.split(",")}).explode(topic_col)
        ex[topic_col] = ex[topic_col].str.strip()
        s = ex[topic_col]
        s = s[~s.isin(["", "nan", "None", "N/A", "該当なし"])]
        if s.empty:
            results["summary"] = f"カテゴリ列 '{topic_col}' に有効なデータがありません。"
            return results
        targets = s.value_counts().head(10).index.tolist()

        out_rows = []
        for cat in targets:
            df_f = ex[ex[topic_col] == cat]
            if df_f.empty:
                continue
            top5 = df_f.nlargest(5, eng_col, keep="first")
            top_posts = []
            for _, r in top5.iterrows():
                snippet = str(r[text_col])[:100]
                val = int(r[eng_col]) if pd.notna(r[eng_col]) else 0
                top_posts.append({"engagement": val, "original_text_snippet": snippet})
            out_rows.append({"category": cat, "post_count": len(df_f), "top_posts": top_posts})

        results["data"] = pd.DataFrame(out_rows)
        results["summary"] = f"「{topic_col}」別の高「{eng_col}」投稿TOP5を抽出しました。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results


# ================================================================================
# 9) A/B 比較
# ================================================================================
def run_ab_comparison(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    2つのグループ（AとB）に分けてカテゴリ別/地域別の出現数と順位差を比較する。
    suggestion["ui_ab_params"] に {a_col, a_val, b_col, b_val} を期待。
    """
    results = {"data": {}, "image_base64": None, "summary": ""}

    try:
        ab = suggestion.get("ui_ab_params", {}) if isinstance(suggestion, dict) else {}
        a_col, a_val = ab.get("a_col"), ab.get("a_val")
        b_col, b_val = ab.get("b_col"), ab.get("b_val")

        loc_col = find_col(df, ["市区町村キーワード", "location", "city", "地域"])
        topic_col = find_col(df, ["話題カテゴリ", "topic", "category"])

        if not all([a_col, a_val, b_col, b_val, loc_col, topic_col]):
            results["summary"] = "A/B比較のパラメータ（A/B列/値、地域列、トピック列）が不足しています。"
            return results

        df_A = df[df[a_col].astype(str).str.contains(re.escape(str(a_val)), na=False)]
        df_B = df[df[b_col].astype(str).str.contains(re.escape(str(b_val)), na=False)]
        if df_A.empty or df_B.empty:
            results["summary"] = f"グループA ({a_val}: {len(df_A)}件) または グループB ({b_val}: {len(df_B)}件) が 0 件です。"
            return results

        # カテゴリ別
        cats_A = df_A[topic_col].astype(str).str.split(", ").explode().value_counts().rename(f"Count (A: {a_val})")
        cats_B = df_B[topic_col].astype(str).str.split(", ").explode().value_counts().rename(f"Count (B: {b_val})")
        cat_cmp = pd.concat([cats_A, cats_B], axis=1).fillna(0).astype(int)
        cat_cmp["Total"] = cat_cmp.sum(axis=1)
        cat_cmp.sort_values(by="Total", ascending=False, inplace=True)
        sum_A = cat_cmp[cats_A.name].sum()
        sum_B = cat_cmp[cats_B.name].sum()
        cat_cmp[f"Share (A: {a_val})"] = (cat_cmp[cats_A.name] / sum_A).map("{:.1%}".format) if sum_A > 0 else "0%"
        cat_cmp[f"Share (B: {b_val})"] = (cat_cmp[cats_B.name] / sum_B).map("{:.1%}".format) if sum_B > 0 else "0%"

        # 地域別順位差
        loc_A = df_A[loc_col].astype(str).value_counts().rename(f"Count (A: {a_val})")
        loc_B = df_B[loc_col].astype(str).value_counts().rename(f"Count (B: {b_val})")
        rank_cmp = pd.concat([loc_A, loc_B], axis=1).fillna(0).astype(int)
        rank_cmp[f"Rank (A: {a_val})"] = rank_cmp[loc_A.name].rank(ascending=False, method="min").astype(int)
        rank_cmp[f"Rank (B: {b_val})"] = rank_cmp[loc_B.name].rank(ascending=False, method="min").astype(int)
        rank_cmp["Rank Change (A vs B)"] = (rank_cmp[f"Rank (B: {b_val})"] - rank_cmp[f"Rank (A: {a_val})"]).astype(int)
        rank_cmp.sort_values(by=f"Count (B: {b_val})", ascending=False, inplace=True)

        results["data"] = {
            "category_comparison": cat_cmp.reset_index().rename(columns={"index": topic_col}).to_dict(orient="records"),
            "ranking_comparison": rank_cmp.reset_index().rename(columns={"index": loc_col}).head(20).to_dict(orient="records"),
        }
        results["summary"] = f"A/B比較: 「{a_val}」(A:{len(df_A)}件) vs 「{b_val}」(B:{len(df_B)}件)。"
        return results
    except Exception as e:
        results["summary"] = f"エラー: {e}"
        return results