# services/llm.py
# -----------------------------------------------------------------------------
# 役割：
# - Google Generative AI (Gemini) を LangChain 経由で呼び出す共通クライアント
# - get_llm(...) でキャッシュ済みのラッパーを返し、.invoke/.predict のどちらでも "文字列" を返す
# - 安全設定（Safety Settings）を調整し、分析業務での不必要な拒否反応を抑制
# - GOOGLE_API_KEY は .env / st.secrets / OS 環境から取得
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import os

# --- .env ロード ---------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Streamlit ----------------------------------------------------------------
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None
    _HAS_ST = False

# --- LangChain & Google Generative AI -----------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    # 安全設定用の型をインポート
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
    """GOOGLE_API_KEY を取得（優先順位：secrets > 環境変数）"""
    if _HAS_ST:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                key = st.secrets["GOOGLE_API_KEY"]
                if isinstance(key, str) and key.strip():
                    return key.strip()
        except Exception:
            pass
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    return v or None


def _cache_resource():
    """Streamlit の cache_resource を返す"""
    if _HAS_ST and hasattr(st, "cache_resource"):
        return st.cache_resource
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
    """Chat* モデルの返り値を '文字列' に正規化する薄いラッパー"""
    def __init__(self, llm_obj: Any, system_prompt: Optional[str] = None):
        self._llm = llm_obj
        self._system_prompt = system_prompt

    def _call(self, prompt: str) -> str:
        if not prompt:
            return ""
        final_prompt = prompt if not self._system_prompt else f"{self._system_prompt}\n\n{prompt}"
        try:
            out = self._llm.invoke(final_prompt)
            content = getattr(out, "content", None)
            if content is not None:
                if isinstance(content, list):
                    texts = [str(part.get("text")) if isinstance(part, dict) else str(part) for part in content]
                    return "\n".join(filter(None, texts)).strip()
                return str(content).strip()
            return str(out).strip()
        except Exception as e:
            return f"Error from Gemini: {str(e)}"

    def invoke(self, prompt: str) -> str:
        return self._call(prompt)

    def predict(self, prompt: str) -> str:
        return self._call(prompt)


@_cache_resource()
def _create_llm_cached(params: _LLMParams) -> Optional[_TextOnlyLLMWrapper]:
    """ChatGoogleGenerativeAI を構築し、ラッパーで包んで返す"""
    if not _HAS_GEMINI:
        if _HAS_ST:
            st.error("langchain-google-genai が見つかりません。")
        return None

    api_key = _resolve_google_api_key()
    if not api_key:
        if _HAS_ST:
            st.error("GOOGLE_API_KEY が未設定です。")
        return None

    # --- 安全設定（Safety Settings）の定義 ---
    # 政治や紛争の分析において、AIが過剰にガードを固めないよう設定を緩和
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    try:
        llm = ChatGoogleGenerativeAI(
            model=params.model_name,
            temperature=params.temperature,
            max_output_tokens=params.max_output_tokens,
            google_api_key=api_key,
            safety_settings=safety_settings, # 安全設定を適用
        )
        if _HAS_ST:
            st.info(f"Gemini 初期化成功: model={params.model_name}, key={_mask_key(api_key)}")
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
    アプリ共通の LLM 取得関数（キャッシュ対応）。
    """
    params = _LLMParams(
        model_name=model_name,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        system_prompt=system_prompt,
    )
    return _create_llm_cached(params)