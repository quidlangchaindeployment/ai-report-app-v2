# services/aws_bedrock.py
# -----------------------------------------------------------------------------
# 役割：
# - S3 入出力と Bedrock の「バッチ推論ジョブ」(Create/Get/Stop) を一元化
# - Step A から呼び出される簡潔な関数インターフェイスを提供
# - .env による設定読み込み（app.py で load_dotenv 済みが理想だが、ここでもフォールバック）
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

# --- .env 読み込み（親側で済んでいれば no-op） -----------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # 安全：存在しなければ何もしない
except Exception:
    pass

# --- ロギング（Streamlit があればUIへも出せる） -----------------------------------
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    try:
        # 任意：UIログへの出力
        from utils.streamlit_logging import StreamlitLogHandler  # type: ignore

        h = StreamlitLogHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        _logger.addHandler(h)
    except Exception:
        # Fallback: 標準出力
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        _logger.addHandler(sh)
    _logger.setLevel(logging.INFO)

# --- boto3 セッション/クライアント ------------------------------------------------
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _get_region() -> str:
    return os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def get_aws_clients(region_name: Optional[str] = None) -> Tuple["boto3.client", "boto3.client"]:
    """
    S3 と Bedrock のクライアントを返す。
    - region_name 未指定時は AWS_DEFAULT_REGION または us-east-1 を使用
    - 認証情報は標準の優先度（環境変数 / プロファイル / IMDS）に従う
    """
    region = region_name or _get_region()
    _logger.info(f"Initializing AWS clients (region={region})")

    session = boto3.session.Session(region_name=region)
    s3 = session.client("s3")
    bedrock = session.client("bedrock")  # control plane: Create/Get/Stop model invocation job
    return s3, bedrock


# --- S3 ユーティリティ -------------------------------------------------------------
def upload_file_to_s3(local_path: str, bucket: str, key: str) -> bool:
    """
    ローカルファイルを S3 へアップロード。成功で True。
    """
    try:
        s3, _ = get_aws_clients()
        _logger.info(f"Uploading to s3://{bucket}/{key} (src={local_path})")
        s3.upload_file(local_path, bucket, key)
        return True
    except (ClientError, BotoCoreError) as e:
        _logger.error(f"S3 upload failed: {e}")
        return False
    except Exception as e:
        _logger.error(f"S3 upload unexpected error: {e}")
        return False


def download_file_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """
    S3 からローカルへダウンロード。成功で True。
    """
    try:
        s3, _ = get_aws_clients()
        _logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        return True
    except (ClientError, BotoCoreError) as e:
        _logger.error(f"S3 download failed: {e}")
        return False
    except Exception as e:
        _logger.error(f"S3 download unexpected error: {e}")
        return False


# --- Bedrock: バッチ推論ジョブ -----------------------------------------------------
def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"環境変数 {name} が未設定です。")
    return v


def create_batch_job(job_name: str, input_key: str, output_prefix: str) -> Optional[str]:
    """
    Bedrock の「モデル推論ジョブ（バッチ）」を作成。
    - 入力: S3 の JSONL 1ファイル（input_key）
    - 出力: S3 のプレフィックス（output_prefix）配下に .out が作成される
    返り値: jobArn（失敗時は None）

    必須の環境変数:
      - BEDROCK_S3_INPUT_BUCKET
      - BEDROCK_S3_OUTPUT_BUCKET
      - BEDROCK_ROLE_ARN
    任意:
      - BEDROCK_MODEL_ID（未設定時 "amazon.nova-lite-v1:0"）
    """
    try:
        input_bucket = _require_env("BEDROCK_S3_INPUT_BUCKET")
        output_bucket = _require_env("BEDROCK_S3_OUTPUT_BUCKET")
        role_arn = _require_env("BEDROCK_ROLE_ARN")
    except RuntimeError as e:
        _logger.error(str(e))
        return None

    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")

    s3_input_uri = f"s3://{input_bucket}/{input_key}"
    s3_output_uri = f"s3://{output_bucket}/{output_prefix}".rstrip("/") + "/"

    try:
        _, bedrock = get_aws_clients()
        _logger.info(
            f"Creating Bedrock batch job: jobName={job_name}, modelId={model_id}, "
            f"input={s3_input_uri}, output={s3_output_uri}"
        )

        resp = bedrock.create_model_invocation_job(
            jobName=job_name,
            roleArn=role_arn,
            modelId=model_id,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": s3_input_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": s3_output_uri}},
        )
        job_arn = resp.get("jobArn") or resp.get("jobArn".encode(), None)
        if not job_arn:
            _logger.error(f"create_model_invocation_job: jobArn がレスポンスに存在しません: {resp}")
            return None

        _logger.info(f"Bedrock batch job created: {job_arn}")
        return job_arn
    except (ClientError, BotoCoreError) as e:
        _logger.error(f"Failed to create Bedrock job: {e}")
        return None
    except Exception as e:
        _logger.error(f"Unexpected error in create_batch_job: {e}")
        return None


def get_batch_job_status(job_arn: str) -> str:
    """
    ジョブのステータス文字列を返す。
    例: 'Submitted' | 'InProgress' | 'Completed' | 'Failed' | 'Stopping' | 'Stopped'
    """
    try:
        _, bedrock = get_aws_clients()
        resp = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = (resp.get("status") or "").strip()
        if not status:
            # 念のため別の位置も探索
            status = (resp.get("jobStatus") or "").strip()
        return status or "Unknown"
    except (ClientError, BotoCoreError) as e:
        _logger.error(f"Failed to get job status: {e}")
        return "Error"
    except Exception as e:
        _logger.error(f"Unexpected error in get_batch_job_status: {e}")
        return "Error"


def stop_batch_job(job_arn: str) -> bool:
    """
    実行中のジョブを停止。成功で True。
    """
    try:
        _, bedrock = get_aws_clients()
        _logger.info(f"Stopping Bedrock job: {job_arn}")
        bedrock.stop_model_invocation_job(jobIdentifier=job_arn)
        return True
    except (ClientError, BotoCoreError) as e:
        _logger.error(f"Failed to stop job: {e}")
        return False
    except Exception as e:
        _logger.error(f"Unexpected error in stop_batch_job: {e}")
        return False