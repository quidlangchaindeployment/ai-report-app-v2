# analysis/export.py
# --- Step B: 実行結果を JSONL に変換するヘルパ -------------------------------
# 役割：
# - Step B の実行結果（dict: タスク名 -> 結果dict）を JSONL 文字列へ変換
# - 「全体のメトリクス」を先頭行(OverallSummary)に配置し、他タスクのサマリを集約
# - DataFrame / Series / dict / list / str / None を安全にシリアライズ
# - 画像(Base64)は 1MB 超なら除外し、注記を追加
# ---------------------------------------------------------------------------

from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Dict, List, Optional
from datetime import datetime
import subprocess

# pandas はオプショナルに扱う（未導入でもクラッシュしない）
try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    _HAS_PANDAS = False


# ============= 内部ユーティリティ ==================================================
def _utc_now_iso() -> str:
    """
    現在時刻を ISO 8601 形式（例: 2026-03-24T01:23:45Z）で返す
    """
    return datetime.utcnow().isoformat() + "Z"


def _get_git_sha() -> str:
    """
    現在の Git の commit SHA を返す（取得できない場合は 'unknown'）
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha
    except Exception:
        return "unknown"


def _is_dataframe(x: Any) -> bool:
    return _HAS_PANDAS and isinstance(x, pd.DataFrame)  # type: ignore


def _is_series(x: Any) -> bool:
    return _HAS_PANDAS and isinstance(x, pd.Series)  # type: ignore


def _truncate_base64_if_needed(image_b64: Optional[str], limit_bytes: int = 1_000_000) -> (Optional[str], str):
    """
    Base64 画像のサイズが大きすぎる場合は None を返し、注記を付ける。
    """
    if not image_b64:
        return None, "No image generated for this task."
    try:
        size = len(image_b64.encode("utf-8"))
        if size <= limit_bytes:
            return image_b64, "Base64 encoded PNG image attached."
        else:
            return None, "Image was generated but exceeded 1MB and was not included."
    except Exception:
        return None, "Image decode error: omitted."


def _serialize_data_payload(data: Any, *, df_row_limit: int = 500) -> Dict[str, Any]:
    """
    「結果dict['data']」の値をシリアライズして返す。
    返却：
        {"data": <json serializable or str or None>, "note": <optional str>}
    """
    out: Dict[str, Any] = {"data": None}
    note_parts: List[str] = []

    # pandas.DataFrame
    if _is_dataframe(data):
        try:
            df = data  # type: ignore
            if len(df) > df_row_limit:
                out["data"] = df.head(df_row_limit).to_json(orient="records", force_ascii=False)
                note_parts.append(f"Data truncated. Showing {df_row_limit} of {len(df)} records.")
            else:
                out["data"] = df.to_json(orient="records", force_ascii=False)
        except Exception as e:
            out["data"] = f"(DataFrame serialization error: {e})"

    # pandas.Series
    elif _is_series(data):
        try:
            out["data"] = data.to_dict()  # type: ignore
        except Exception as e:
            out["data"] = f"(Series serialization error: {e})"

    # dict / list（そのまま載せる）
    elif isinstance(data, (dict, list)):
        try:
            # 一旦ダンプできるか試験（失敗時は str 化）
            json.dumps(data, ensure_ascii=False, default=str)
            out["data"] = data
        except Exception:
            out["data"] = str(data)

    # str（そのまま）
    elif isinstance(data, str):
        out["data"] = data

    # None or 空（明示）
    elif data is None or (hasattr(data, "empty") and getattr(data, "empty")):
        out["data"] = None
        note_parts.append("No data returned from analysis.")

    # その他（str 化）
    else:
        try:
            out["data"] = json.loads(str(data))
        except Exception:
            out["data"] = str(data)

    if note_parts:
        out["note"] = " ".join(note_parts)
    return out


# ============= 公開：JSONL 変換関数 ===============================================

def convert_results_to_json_string(results_dict: Dict[str, Any]) -> str:
    """
    Step B で作成した「タスク名 -> 結果dict」を JSONL 文字列に変換する。

    結果dict の期待フォーマット（Step B 側と合わせる）：
        {
            "data": <pd.DataFrame | pd.Series | dict | list | str | None>,
            "image_base64": <str | None>,
            "html_content": <str | None>,  # 任意（pyvisなど）
            "summary": <str>
        }

    出力（JSONL 各行の例）：
        {"analysis_task": "OverallSummary", "data": {...}, "summary": "...", "image_base64": null, "image_note": "No image", "analysis_summaries": {...}}
        {"analysis_task": "単純集計: 市区町村キーワード", "data": "[{...}, ...]", "summary": "...", "image_base64": "<base64 or null>", "image_note": "..."}
        ...
    """
    json_lines: List[str] = []
    task_summaries: Dict[str, str] = {}

    # --- 1) 「全体のメトリクス」タスクを先に探す（任意） -------------------------
    overall_key = "全体のメトリクス"
    overall_payload: Optional[Dict[str, Any]] = None
    if overall_key in results_dict:
        res = results_dict.get(overall_key, {}) or {}
        data = res.get("data", {})
        summary = res.get("summary", "")
        # 画像は通常なし
        overall_payload = {
            "analysis_task": "OverallSummary",
            "data": data if isinstance(data, (dict, list, str)) else _serialize_data_payload(data)["data"],
            "summary": summary,
            "image_base64": None,
            "image_note": "No image",
            "analysis_summaries": {},  # 後で埋める
        }
    else:
        # 無い場合は空の overall を用意
        overall_payload = {
            "analysis_task": "OverallSummary",
            "data": {"message": "Overall metrics not provided."},
            "summary": "",
            "image_base64": None,
            "image_note": "No image",
            "analysis_summaries": {},
        }

    # --- 2) その他タスクを処理 -----------------------------------------------------
    for task_name, result in results_dict.items():
        if task_name == overall_key:
            # 後で overall に summaries を詰めて最初に入れる
            continue

        try:
            line: Dict[str, Any] = {"analysis_task": task_name}
            line["summary"] = result.get("summary", "N/A")

            # data の型に応じてシリアライズ
            payload = _serialize_data_payload(result.get("data"))
            for k, v in payload.items():
                line[k] = v  # "data" と "note"(任意)
                
            # -------------------------------------------------
            # Addendum v2: audit / explanation schema (empty)
            # -------------------------------------------------
            line.update({
                "analysis_type": "",
                "question": "",
                "key_findings": [],
                "confidence": None,
                "recommended_slide": False,

                "evidence_ids": [],
                "evidence_snippets": [],
                "counter_examples": [],
                "limitations": [],

                "provenance": {
                    "prompt_id": "",
                    "model_id": "",
                    "parameters": {},
                    "dataset_hash": "",
                    "code_version": _get_git_sha(),
                    "timestamp": _utc_now_iso(),
                },
            })

            # 画像（Base64）と HTML の扱い
            html_content = result.get("html_content")
            image_b64 = result.get("image_base64")

            if html_content:
                # HTMLは JSONL に直接は載せず注記で表現
                line["image_base64"] = None
                line["image_note"] = "No image (HTML content available in app preview)."
            else:
                img, note = _truncate_base64_if_needed(image_b64)
                line["image_base64"] = img
                line["image_note"] = note

            # 1行としてダンプ
            json_lines.append(json.dumps(line, ensure_ascii=False, default=str))

            # 後で overall に集約する summaries
            task_summaries[task_name] = line.get("summary", "")

        except Exception as e:
            json_lines.append(json.dumps({"analysis_task": task_name, "error": str(e)}, ensure_ascii=False))

    # --- 3) overall に summaries を入れて「先頭」に追加 ----------------------------
    if overall_payload is not None:
        overall_payload["analysis_summaries"] = task_summaries
        json_lines.insert(0, json.dumps(overall_payload, ensure_ascii=False, default=str))

    # --- 4) 連結して返す -----------------------------------------------------------
    return "\n".join(json_lines)