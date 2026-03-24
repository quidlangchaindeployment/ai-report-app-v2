# utils/plotting.py
# -----------------------------------------------------------------------------
# 役割：
# - 分析結果を簡潔に可視化し、Base64(PNG) で返すユーティリティ
# - 対応プロット：
#     * "bar"        : 棒グラフ（x_col × y_col）
#     * "timeseries" : 折れ線（x_col=日時、y_col=値、"keyword" 列があれば系列分け）
#     * "wordcloud"  : ワードクラウド（df["word"], df["count"] を想定）
# - 戻り値：str(Base64) / 失敗時 None
#
# 注意：
# - 余計なスタイルは適用しません（実行環境のデフォルトを尊重）。
# - 日本語フォントが無い環境では文字化けする場合があります（その場合は OS に日本語フォントを導入してください）。
# -----------------------------------------------------------------------------

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# WordCloud は任意依存（無ければ wordcloud プロットは None を返す）
try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False


# ===== 共通：Figure -> Base64 =====================================================
def _fig_to_base64(fig, *, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ===== 個別プロット ===============================================================

def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Optional[str]:
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    try:
        # 可読性のため上位 20 に制限
        work = df[[x_col, y_col]].dropna()
        if not pd.api.types.is_numeric_dtype(work[y_col]):
            work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[y_col]).sort_values(y_col, ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 5))  # スタイルは環境デフォルト
        ax.bar(work[x_col].astype(str), work[y_col].astype(float))
        ax.set_title(title or "")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        # ラベルが長い場合の回転
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        plt.close("all")
        return None


def _plot_timeseries(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Optional[str]:
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    try:
        work = df[[x_col, y_col] + ([c for c in ["keyword"] if c in df.columns])].copy()

        # 日付変換
        work[x_col] = pd.to_datetime(work[x_col], errors="coerce")
        work = work.dropna(subset=[x_col, y_col])
        if work.empty:
            return None

        # 値は数値化
        if not pd.api.types.is_numeric_dtype(work[y_col]):
            work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[y_col])

        # 系列分け（keyword があれば）
        fig, ax = plt.subplots(figsize=(8, 5))
        if "keyword" in work.columns:
            # 系列が多いと見づらいので上位 5 に絞る
            top_keys = (
                work.groupby("keyword")[y_col]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .index
            )
            for kw in top_keys:
                sub = work[work["keyword"] == kw].sort_values(x_col)
                ax.plot(sub[x_col], sub[y_col], marker="", label=str(kw))
            ax.legend(loc="best", fontsize=9)
        else:
            work = work.sort_values(x_col)
            ax.plot(work[x_col], work[y_col], marker="")

        ax.set_title(title or "")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.autofmt_xdate()
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception:
        plt.close("all")
        return None


def _plot_wordcloud(df: pd.DataFrame, title: str) -> Optional[str]:
    if not _HAS_WORDCLOUD:
        return None
    # 期待：df に "word" と "count" 列
    if df is None or df.empty or not {"word", "count"}.issubset(df.columns):
        return None
    try:
        # 頻度辞書を作成（上位 200 まで）
        wc_df = df[["word", "count"]].dropna()
        if not pd.api.types.is_numeric_dtype(wc_df["count"]):
            wc_df["count"] = pd.to_numeric(wc_df["count"], errors="coerce")
        wc_df = wc_df.dropna(subset=["count"]).sort_values("count", ascending=False).head(200)
        freqs = {str(r["word"]): float(r["count"]) for _, r in wc_df.iterrows() if str(r["word"]).strip()}

        # 日本語フォントは環境依存のため、指定しない（導入済みなら自動で使われます）
        wc = WordCloud(
            width=900,
            height=600,
            background_color="white",
            collocations=False,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return _fig_to_base64(fig, dpi=110)
    except Exception:
        plt.close("all")
        return None


# ===== パブリック API =============================================================

def generate_graph_image(
    df: pd.DataFrame,
    plot_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "",
) -> Optional[str]:
    """
    可視化を生成して Base64(PNG) を返します。失敗時は None。
    - plot_type:
        * "bar"        : x_col × y_col
        * "timeseries" : x_col=日時, y_col=値（"keyword" 列があれば系列別）
        * "wordcloud"  : df["word"], df["count"] を使用
    """
    try:
        kind = (plot_type or "").lower()
        if kind == "bar":
            if not x_col or not y_col:
                return None
            return _plot_bar(df, x_col, y_col, title)
        elif kind == "timeseries":
            if not x_col or not y_col:
                return None
            return _plot_timeseries(df, x_col, y_col, title)
        elif kind == "wordcloud":
            return _plot_wordcloud(df, title)
        else:
            # 未対応の種類
            return None
    except Exception:
        return None