# services/llm.py
# -----------------------------------------------------------------------------
# 役割：
# - Google Generative AI (Gemini) を LangChain 経由で呼び出す共通クライアント
# - get_llm(...) でキャッシュ済みのラッパーを返し、.invoke/.predict のどちらでも "文字列" を返す
# - GOOGLE_API_KEY は .env / st.secrets / OS 環境から取得（未設定でもアプリが落ちないよう配慮）
# 依存：
#   - langchain-google-genai==1.0.3
#   - python-dotenv（任意）
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import os

# --- .env ロード（親で済んでいても重複無害） ---------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Streamlit は任意。存在すればメッセージとキャッシュに利用 --------------------
try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

# --- LangChain: Google Generative AI (Gemini) --------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "****"
    return k[:4] + "..." + k[-4:]


def _resolve_google_api_key() -> Optional[str]:
    """
    GOOGLE_API_KEY を以下の優先度で取得：
      1) st.secrets["GOOGLE_API_KEY"]
      2) 環境変数 GOOGLE_API_KEY
      3) .env から読み込まれた os.environ
    未設定なら None を返す。
    """
    # 1) Streamlit secrets
    if _HAS_ST:
        try:
            if "GOOGLE_API_KEY" in st.secrets:  # type: ignore[attr-defined]
                key = st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
                if isinstance(key, str) and key.strip():
                    return key.strip()
        except Exception:
            pass
    # 2) OS 環境
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    return v or None


def _cache_resource():
    """Streamlit の cache_resource を返す。無ければ no-op デコレータ。"""
    if _HAS_ST and hasattr(st, "cache_resource"):
        return st.cache_resource  # type: ignore
    def _decorator(fn):
        return fn
    return _decorator


@dataclass(frozen=True)
class _LLMParams:
    model_name: str
    temperature: float
    timeout_seconds: int
    max_output_tokens: int
    system_prompt: Optional[str]


class _TextOnlyLLMWrapper:
    """
    LangChain の Chat* モデルを受け取り、.invoke/.predict の返り値を
    '文字列' に正規化して返す薄いラッパー。
    """
    def __init__(self, llm_obj: Any, system_prompt: Optional[str] = None):
        self._llm = llm_obj
        self._system_prompt = system_prompt

    def _call(self, prompt: str) -> str:
        if not prompt:
            return ""

        # System prompt を付与（必要ならテンプレ化可能）
        final_prompt = prompt if not self._system_prompt else f"{self._system_prompt}\n\n{prompt}"

        try:
            # LangChain 0.1+ では .invoke がベース
            out = self._llm.invoke(final_prompt)
        except Exception:
            # 旧API/互換
            out = self._llm.predict(final_prompt)

        # 返却を string に正規化
        try:
            # ChatMessage の content -> str
            content = getattr(out, "content", None)
            if content is not None:
                if isinstance(content, list):
                    # content が複合（text/chunk）の場合
                    texts = []
                    for part in content:
                        t = part.get("text") if isinstance(part, dict) else str(part)
                        if t:
                            texts.append(str(t))
                    return "\n".join(texts).strip()
                return str(content).strip()
            # すでに str の場合
            return str(out).strip()
        except Exception:
            return str(out)

    # 互換：step_* 側で invoke/predict を混用しても OK
    def invoke(self, prompt: str) -> str:
        return self._call(prompt)

    def predict(self, prompt: str) -> str:
        return self._call(prompt)


@_cache_resource()
def _create_llm_cached(params: _LLMParams) -> Optional[_TextOnlyLLMWrapper]:
    """
    指定パラメータで ChatGoogleGenerativeAI を構築し、_TextOnlyLLMWrapper で包んで返す。
    Streamlit があればリソースキャッシュされる。
    """
    if not _HAS_GEMINI:
        if _HAS_ST:
            st.error("langchain-google-genai が見つかりません。`pip install langchain-google-genai` を実行してください。")
        return None

    api_key = _resolve_google_api_key()
    if not api_key:
        if _HAS_ST:
            st.error(
                "GOOGLE_API_KEY が未設定です。`.env` あるいは `st.secrets` に GOOGLE_API_KEY を設定してください。\n"
                "例）.env:\nGOOGLE_API_KEY=xxxxxxxxxxxxxxxx\n"
            )
        return None

    # LangChain 側の引数にマップ
    # - `max_output_tokens` は `max_output_tokens` で渡す
    # - `timeout` は requests のタイムアウト（秒）
    try:
        llm = ChatGoogleGenerativeAI(
            model=params.model_name,
            temperature=params.temperature,
            max_output_tokens=params.max_output_tokens,
            google_api_key=api_key,
            # request_timeout はバージョンにより `http_client` 側で扱われる場合あり。ここでは防御的に設定。
            # langchain-google-genai 1.0.3 時点では `client_options` を経由せずとも基本利用可。
        )
        if _HAS_ST:
            st.info(
                f"Gemini クライアント初期化に成功: model={params.model_name}, temp={params.temperature}, "
                f"max_tokens={params.max_output_tokens}, key={_mask_key(api_key)}"
            )
        return _TextOnlyLLMWrapper(llm, system_prompt=params.system_prompt)
    except Exception as e:
        if _HAS_ST:
            st.error(f"Gemini クライアント初期化エラー: {e}")
        return None


def get_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    timeout_seconds: int = 60,
    max_output_tokens: int = 2048,
    system_prompt: Optional[str] = None,
) -> Optional[_TextOnlyLLMWrapper]:
    """
    アプリ共通の LLM 取得関数。
    - Streamlit の @cache_resource により、同一パラメータでは再生成されません。
    - 返却オブジェクトは .invoke/.predict のどちらでも "文字列" を返します。

    Parameters
    ----------
    model_name : str
        例: "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
    temperature : float
        0.0〜1.0（低いほど決定論的）
    timeout_seconds : int
        （将来の拡張で HTTP タイムアウトに反映）
    max_output_tokens : int
        モデルの最大出力トークン数
    system_prompt : Optional[str]
        すべてのプロンプトに前置する共通文言

    Returns
    -------
    Optional[_TextOnlyLLMWrapper]
        文字列を返す `.invoke()` / `.predict()` を持つ簡易LLM。API未設定時は None。
    """
    params = _LLMParams(
        model_name=model_name,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        system_prompt=system_prompt,
    )
    return _create_llm_cached(params)