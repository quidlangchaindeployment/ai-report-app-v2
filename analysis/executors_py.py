# analysis/executors_py.py
# --- Pythonベース分析 実装集 ---
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# --- 共通ユーティリティへの依存 ---
from utils.cols_detect import find_col, find_cols, find_engagement_cols
from utils.plotting import generate_graph_image
from nlp.spacy_loader import load_spacy_model

# --- ドメイン知識（列名エイリアス）のインポート ---
from analysis.config import COLUMN_ALIASES

# ================================================================================
# 1) 全体メトリクス
# ================================================================================
def run_overall_metrics(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    try:
        metrics["total_posts"] = f"{len(df):,}件"

        # エンゲージメント候補
        eng_cols = find_engagement_cols(df, COLUMN_ALIASES["engagement"])
        total_eng = 0
        for c in eng_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                total_eng += int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        metrics["total_engagement"] = f"{total_eng:,}件" if eng_cols else "N/A"

        # センチメント傾向
        sent_col = find_col(df, COLUMN_ALIASES["sentiment"])
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
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    flag_cols = suggestion.get("suitable_cols", [])
    if not flag_cols:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "集計対象の列が見つかりません。"}

    col = suggestion.get("ui_selected_col", flag_cols[0])
    if col not in df.columns:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"列 '{col}' が存在しません。"}

    try:
        s = df[col].astype(str).str.split(",").explode().str.strip()
        s = s[~s.isin(["", "nan", "None", "N/A", "該当なし"])]
        if s.empty:
            return {"data": pd.DataFrame(), "image_base64": None, "summary": "集計対象のキーワードがありません。"}

        counts_df = s.value_counts().head(50).reset_index()
        counts_df.columns = [col, "count"]
        
        results["data"] = counts_df
        results["image_base64"] = generate_graph_image(df=counts_df, plot_type="bar", x_col=col, y_col="count", title=f"「{col}」 頻出TOP20")
        top2 = ", ".join([f"{counts_df.iloc[i,0]} ({counts_df.iloc[i,1]}件)" for i in range(min(2, len(counts_df)))])
        results["summary"] = f"「{col}」の単純集計を実行。上位: {top2}。"
        return results
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 3) クロス集計
# ================================================================================
def run_crosstab(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    cols = suggestion.get("suitable_cols", [])
    if len(cols) < 2: return {"data": pd.DataFrame(), "image_base64": None, "summary": "クロス集計には2列以上必要です。"}

    col1 = suggestion.get("ui_selected_col1", cols[0])
    col2 = suggestion.get("ui_selected_col2", cols[1])

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
        
        return {"data": long_df.head(100), "image_base64": None, "summary": f"「{col1}」×「{col2}」のクロス集計を実行。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 4) 時系列キーワード分析
# ================================================================================
def run_timeseries(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    cols_dict = suggestion.get("suitable_cols", {})
    if not isinstance(cols_dict, dict) or "datetime" not in cols_dict or "keywords" not in cols_dict:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "列情報が不十分です。"}

    dt_col = suggestion.get("ui_selected_dt_col", cols_dict["datetime"][0])
    kw_col = suggestion.get("ui_selected_kw_col", cols_dict["keywords"][0])

    try:
        work = df[[dt_col, kw_col]].copy()
        work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
        work = work.dropna(subset=[dt_col])

        ex = work.assign(**{kw_col: work[kw_col].astype(str).str.split(",")}).explode(kw_col)
        ex[kw_col] = ex[kw_col].str.strip()
        ex = ex[~ex[kw_col].isin(["", "nan", "None", "N/A", "該当なし"])]

        ts = ex.groupby([pd.Grouper(key=dt_col, freq="D"), kw_col]).size().rename("count").reset_index()
        ts.columns = ["date", "keyword", "count"]
        top_keywords = ex[kw_col].value_counts().head(50).index
        ts_f = ts[ts["keyword"].isin(top_keywords)]

        data_for_json = ts_f.sort_values(by=["keyword", "date"]).copy()
        data_for_json["date"] = data_for_json["date"].dt.strftime("%Y-%m-%d")

        img = generate_graph_image(df=ts_f, plot_type="timeseries", x_col="date", y_col="count", title=f"「{kw_col}」別 時系列トレンド")
        return {"data": data_for_json, "image_base64": img, "summary": f"『{dt_col}』×『{kw_col}』の時系列分析を実行。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 5) テキストマイニング（頻出語）
# ================================================================================
def run_text_mining(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    text_col = suggestion.get("ui_selected_text_col", find_col(df, COLUMN_ALIASES["text"]))
    if not text_col or text_col not in df.columns:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "テキスト列が見つかりません。"}

    nlp = load_spacy_model()
    try:
        texts = df[text_col].dropna().astype(str)
        words: List[str] = []
        if nlp is not None:
            target_pos = {"NOUN", "PROPN", "ADJ"}
            stop_words = {"の", "に", "は", "を", "が", "で", "て", "です", "ます", "こと", "もの", "それ", "これ", "ため", "いる", "する", "ある", "ない"}
            for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=50):
                for t in doc:
                    if (t.pos_ in target_pos) and (not t.is_stop) and (t.lemma_ not in stop_words) and (len(t.lemma_) > 1):
                        words.append(t.lemma_)
        else:
            sw = {"の", "に", "は", "を", "が", "で", "て", "です", "ます", "こと", "もの", "それ", "これ", "ため"}
            for tx in texts:
                for w in re.split(r"[、。\s,./!?:;（）()「」【】『』\[\]\-]+", tx):
                    w = w.strip()
                    if len(w) > 1 and w not in sw: words.append(w)

        counts = pd.Series(words).value_counts().head(100).reset_index()
        counts.columns = ["word", "count"]
        img = generate_graph_image(df=counts, plot_type="wordcloud", title=f"「{text_col}」頻出単語")
        return {"data": counts, "image_base64": img, "summary": f"『{text_col}』のテキストマイニングを実行。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 6) 共起ネットワーク（簡易エッジ抽出）
# ================================================================================
def run_cooccurrence_network_pyvis(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    flag_col = suggestion.get("ui_selected_flag_col") or find_col(df, COLUMN_ALIASES["location"]) or find_col(df, COLUMN_ALIASES["category"])
    text_col = suggestion.get("ui_selected_text_col") or find_col(df, COLUMN_ALIASES["text"])

    if not flag_col or not text_col:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "対象列が見つかりません。"}

    try:
        s = df[flag_col].dropna().astype(str).str.split(",").explode().str.strip()
        s = s[~s.isin(["", "nan", "None", "N/A"])]
        selected = s.value_counts().index.tolist()[:10]
        if not selected: return {"data": pd.DataFrame(), "image_base64": None, "summary": "絞り込みキーワードが取得できません。"}

        patt = "|".join([re.escape(k) for k in selected])
        df_f = df[df[flag_col].astype(str).str.contains(patt, na=False)]
        texts = df_f[text_col].dropna().astype(str)

        from collections import Counter
        from itertools import combinations
        edge_counter = Counter()
        
        for tx in texts:
            ws = set([w for w in re.split(r"[、。\s,./!?:;（）()「」【】『』\[\]\-]+", tx) if len(w) > 1][:80])
            for a, b in combinations(sorted(ws), 2):
                edge_counter[(a, b)] += 1

        rows = [{"source": s, "target": t, "weight": w} for (s, t), w in edge_counter.most_common(100)]
        return {"data": pd.DataFrame(rows), "image_base64": None, "summary": f"共起ネットワーク（簡易）を生成。{len(rows)} エッジ。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 7) 汎用：カテゴリ列ごとの深掘り
# ================================================================================
def run_generic_category_summary(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    topic_col = suggestion.get("ui_selected_category_col") or find_col(df, COLUMN_ALIASES["category"])
    text_col = find_col(df, COLUMN_ALIASES["text"])
    
    if not topic_col or not text_col:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "分析に必要な列が見つかりません。"}

    try:
        s = df[topic_col].astype(str).str.split(", ").explode().str.strip()
        targets = s[~s.isin(["", "nan", "None", "N/A", "該当なし"])].value_counts().head(10).index.tolist()

        flag_cols = [c for c in df.columns if c.endswith("キーワード")]
        loc_col = find_col(df, COLUMN_ALIASES["location"])
        cols_for_kw = [c for c in flag_cols if c not in {loc_col, topic_col}]

        rows = []
        for cat in targets:
            df_f = df[df[topic_col].astype(str).str.contains(re.escape(cat), na=False)]
            top_keywords: List[str] = []
            if cols_for_kw:
                comb = pd.concat([df_f[c].astype(str).str.split(", ").explode().str.strip() for c in cols_for_kw])
                top_keywords = comb[~comb.isin(["", "nan", "None", "N/A", "該当なし"])].value_counts().head(5).index.tolist()
            rows.append({"category": cat, "post_count": len(df_f), "top_keywords": top_keywords})

        out_df = pd.DataFrame(rows)
        img = generate_graph_image(df=out_df, plot_type="bar", x_col="category", y_col="post_count", title=f"「{topic_col}」別 投稿数 (Top 10)")
        return {"data": out_df, "image_base64": img, "summary": f"「{topic_col}」別の投稿数と上位キーワードを算出。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 8) 汎用：カテゴリ別 数値列TOP5（バズ投稿）
# ================================================================================
def run_generic_engagement_top5(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    topic_col = suggestion.get("ui_selected_category_col") or find_col(df, COLUMN_ALIASES["category"])
    text_col = suggestion.get("ui_selected_text_col") or find_col(df, COLUMN_ALIASES["text"])
    eng_cols = find_engagement_cols(df, COLUMN_ALIASES["engagement"])
    eng_col = suggestion.get("ui_selected_numeric_col") or (eng_cols[0] if eng_cols else None)

    if not all([topic_col, text_col, eng_col]):
        return {"data": pd.DataFrame(), "image_base64": None, "summary": "必要な列が見つかりません。"}

    try:
        ex = df.assign(**{topic_col: df[topic_col].astype(str).str.split(",")}).explode(topic_col)
        ex[topic_col] = ex[topic_col].str.strip()
        targets = ex[topic_col][~ex[topic_col].isin(["", "nan", "None", "N/A"])].value_counts().head(10).index.tolist()

        out_rows = []
        for cat in targets:
            df_f = ex[ex[topic_col] == cat]
            if df_f.empty: continue
            top5 = df_f.nlargest(5, eng_col, keep="first")
            top_posts = [{"engagement": int(r[eng_col]) if pd.notna(r[eng_col]) else 0, "text": str(r[text_col])[:100]} for _, r in top5.iterrows()]
            out_rows.append({"category": cat, "post_count": len(df_f), "top_posts": top_posts})

        return {"data": pd.DataFrame(out_rows), "image_base64": None, "summary": f"「{topic_col}」別の高「{eng_col}」投稿TOP5を抽出。"}
    except Exception as e:
        return {"data": pd.DataFrame(), "image_base64": None, "summary": f"エラー: {e}"}

# ================================================================================
# 9) A/B 比較
# ================================================================================
def run_ab_comparison(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    try:
        ab = suggestion.get("ui_ab_params", {}) if isinstance(suggestion, dict) else {}
        a_col, a_val, b_col, b_val = ab.get("a_col"), ab.get("a_val"), ab.get("b_col"), ab.get("b_val")
        loc_col, topic_col = find_col(df, COLUMN_ALIASES["location"]), find_col(df, COLUMN_ALIASES["category"])

        if not all([a_col, a_val, b_col, b_val, loc_col, topic_col]):
            return {"data": {}, "image_base64": None, "summary": "A/B比較のパラメータが不足しています。"}

        df_A = df[df[a_col].astype(str).str.contains(re.escape(str(a_val)), na=False)]
        df_B = df[df[b_col].astype(str).str.contains(re.escape(str(b_val)), na=False)]
        
        if df_A.empty or df_B.empty:
            return {"data": {}, "image_base64": None, "summary": f"比較対象のデータが0件です。"}

        # カテゴリ比較
        cats_A = df_A[topic_col].astype(str).str.split(", ").explode().value_counts().rename(f"Count (A)")
        cats_B = df_B[topic_col].astype(str).str.split(", ").explode().value_counts().rename(f"Count (B)")
        cat_cmp = pd.concat([cats_A, cats_B], axis=1).fillna(0).astype(int)
        
        # 順位比較
        loc_A = df_A[loc_col].astype(str).value_counts().rename(f"Count (A)")
        loc_B = df_B[loc_col].astype(str).value_counts().rename(f"Count (B)")
        rank_cmp = pd.concat([loc_A, loc_B], axis=1).fillna(0).astype(int)

        return {
            "data": {
                "category_comparison": cat_cmp.reset_index().to_dict(orient="records"),
                "ranking_comparison": rank_cmp.reset_index().head(20).to_dict(orient="records"),
            },
            "image_base64": None,
            "summary": f"A/B比較: 「{a_val}」(A:{len(df_A)}件) vs 「{b_val}」(B:{len(df_B)}件)。"
        }
    except Exception as e:
        return {"data": {}, "image_base64": None, "summary": f"エラー: {e}"}