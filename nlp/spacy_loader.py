# nlp/spacy_loader.py
# -----------------------------------------------------------------------------
# 役割：
# - 日本語 spaCy モデル（ja_core_news_sm）をロードし、Streamlit のリソースキャッシュに載せる。
# - モデル未インストール時はアプリを落とさず None を返す（UI側でフォールバック可能）。
# - 便利ヘルパ tokenize_texts() も提供。
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, List, Optional

# Streamlit は任意（存在すればキャッシュを使う）
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


def _cache_resource():
    """
    Streamlit の cache_resource を返す。無ければ no-op デコレータ。
    """
    if _HAS_ST and hasattr(st, "cache_resource"):
        return st.cache_resource  # type: ignore

    # no-op decorator
    def _decorator(fn):
        return fn
    return _decorator


@_cache_resource()
def _load_spacy() -> Optional["Language"]:
    """
    内部実装：spaCy と ja_core_news_sm を import して返す。
    失敗時は None を返す（呼び出し側でフォールバック）。
    """
    try:
        import spacy  # type: ignore
    except Exception as e:
        if _HAS_ST:
            st.warning(f"spaCy が見つかりません（{e}）。'pip install spacy' を実行してください。")
        return None

    # 既にロード済みなら再利用
    try:
        nlp = spacy.get_pipe("ja_core_news_sm")  # type: ignore[attr-defined]
        # ↑ get_pipe ではなく既存インスタンスの取得APIは無いので例外になりやすい。try/exceptで続行。
    except Exception:
        nlp = None  # type: ignore

    if nlp is not None:
        return nlp  # type: ignore[return-value]

    # モデルを直接 import（インストール済みチェック）
    try:
        # インストール済みなら通常は "spacy download ja_core_news_sm" 済み
        # ここでは `spacy.load` で読み込みを試みる
        nlp = spacy.load("ja_core_news_sm")  # type: ignore
        return nlp  # type: ignore[return-value]
    except OSError as e:
        # モデル未インストール（典型：OSError: [E050] ...）
        if _HAS_ST:
            st.warning(
                "spaCy 日本語モデル 'ja_core_news_sm' が未インストールのため、高精度解析を無効化します。\n"
                "次のコマンドで事前に導入してください：\n\n"
                "    python -m spacy download ja_core_news_sm\n\n"
                f"詳細: {e}"
            )
        return None
    except Exception as e:
        if _HAS_ST:
            st.error(f"spaCy モデルのロード中にエラーが発生しました: {e}")
        return None


def load_spacy_model() -> Optional["Language"]:
    """
    公開API：日本語モデル（ja_core_news_sm）をロードして返す。
    失敗時は None（UI 側でフォールバック可能）。
    """
    return _load_spacy()


def tokenize_texts(texts: Iterable[str], batch_size: int = 50) -> List[List[str]]:
    """
    ヘルパ：テキスト反復を受け取り、spaCy があれば形態素ごとの lemma を返す。
    spaCy が無い場合は簡易分割（正規表現ベース）にフォールバック。

    戻り値例： [["東京", "観光"], ["牡蠣", "広島", "名物"], ...]
    """
    nlp = load_spacy_model()
    out: List[List[str]] = []

    # spaCy あり → 品詞とストップワードを考慮（NOUN/PROPN/ADJ など）
    if nlp is not None:
        target_pos = {"NOUN", "PROPN", "ADJ"}
        # 日本語の一般的なストップ語（最低限）
        stop_words = {"の", "に", "は", "を", "が", "で", "て", "です", "ます", "こと", "もの", "それ", "これ", "ため", "いる", "する"}

        try:
            for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=max(1, batch_size)):
                toks: List[str] = []
                for t in doc:
                    # t.is_stop は英語寄りの辞書が強いので、自前の stop_words も併用
                    if (t.pos_ in target_pos) and (not t.is_stop) and (t.lemma_ not in stop_words) and (len(t.lemma_) > 1):
                        toks.append(t.lemma_)
                out.append(toks)
            return out
        except Exception as e:
            if _HAS_ST:
                st.warning(f"spaCy によるトークナイズでエラーが発生したため、簡易分割にフォールバックします: {e}")
            # フォールバックに続ける

    # spaCy なし → 簡易分割（句読点・空白）
    import re
    for tx in texts:
        words = [
            w for w in re.split(r"[、。\s,./!?:;（）()「」【】『』\[\]\-]+", str(tx))
            if len(w.strip()) > 1
        ]
        out.append(words)
    return out