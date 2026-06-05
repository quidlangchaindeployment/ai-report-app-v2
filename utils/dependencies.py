# utils/dependencies.py
# --- 依存関係・外部サービスの統合管理モジュール ---
import streamlit as st
import pandas as pd
from typing import Optional, Any, Callable

# --- 1. LLM (Gemini) ---
try:
    from services.llm import get_llm
    HAS_LLM = True
except ImportError:
    get_llm = None
    HAS_LLM = False

# --- 2. AWS Bedrock ---
try:
    from services.aws_bedrock import (
        get_aws_clients,
        upload_file_to_s3,
        create_batch_job,
        stop_batch_job,
        get_batch_job_status,
        download_file_from_s3,
    )
    HAS_AWS = True
except ImportError:
    get_aws_clients = None
    upload_file_to_s3 = None
    create_batch_job = None
    stop_batch_job = None
    get_batch_job_status = None
    download_file_from_s3 = None
    HAS_AWS = False

# --- 3. Python-pptx ---
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    HAS_PPTX = True
except ImportError:
    Presentation = None
    Inches = None
    Pt = None
    HAS_PPTX = False

# --- 4. IO / 分析関連ユーティリティ ---
# (各ファイルで定義されていたダミー関数などを集約)
try:
    from utils.io_helpers import read_file
except ImportError:
    def read_file(file: Any, **kwargs) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            name = file.name.lower()
            if name.endswith(".csv"):
                return pd.read_csv(file, encoding="utf-8"), None
            if name.endswith((".xlsx", ".xls")):
                return pd.read_excel(file), None
            return None, "未対応のファイル形式"
        except Exception as e:
            return None, str(e)

# --- 警告表示ヘルパー ---
def require_llm():
    """LLM機能が必要な箇所で呼び出し、未導入なら警告を出す"""
    if not HAS_LLM:
        st.warning("LLMサービス（Gemini）が初期化されていません。機能が制限されます。")
    return HAS_LLM

def require_aws():
    """AWS機能が必要な箇所で呼び出し、未導入なら警告を出す"""
    if not HAS_AWS:
        st.warning("AWSサービス（Bedrock/S3）が初期化されていません。機能が制限されます。")
    return HAS_AWS