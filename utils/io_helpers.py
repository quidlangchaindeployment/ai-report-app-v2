# utils/io_helpers.py
# -----------------------------------------------------------------------------
# 役割：
# - CSV/Excel 読み込み（安全・堅牢：文字コード/エンジン/行数上限/列名クリーン）
# - 書き出しユーティリティ（CSV/JSON の安全保存、UTF-8-SIG）
# - 一時ファイル作成ヘルパ、S3 キー生成ヘルパ
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import io
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd


# ============================ 読み込み（CSV/Excel） ================================

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名の空白・改行・連続スペース・タブを正規化。重複名は .1, .2 を付与。"""
    cols = []
    seen = {}
    for c in df.columns:
        name = re.sub(r"\s+", " ", str(c)).strip()
        base = name
        k = base
        i = 1
        while k in seen:
            i += 1
            k = f"{base}.{i}"
        seen[k] = True
        cols.append(k)
    df.columns = cols
    return df


def _try_read_csv(file_like: Any, encoding: Optional[str], max_rows: Optional[int]) -> pd.DataFrame:
    """エンコーディング指定でCSVを読む。`max_rows` 指定があれば先頭N行に限定。"""
    kwargs = {
        "encoding": encoding or "utf-8",
        "on_bad_lines": "skip",
        "dtype": "object",          # 型は解析系で適宜変換
        "keep_default_na": False,   # "NA" 文字列をNaNにしない
        "na_values": ["", "NaN"],   # 明示的なNAのみ
    }
    if max_rows:
        kwargs["nrows"] = int(max_rows)
    df = pd.read_csv(file_like, **kwargs)
    return _clean_columns(df)


def _sniff_csv_encoding(sample: bytes) -> Optional[str]:
    """簡易推定：UTF-8 BOM / UTF-8 / cp932 の順で試す。"""
    # BOM あり
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # ASCII または UTF-8 として解釈できるか試験
    try:
        sample.decode("utf-8")
        return "utf-8"
    except Exception:
        pass
    # 日本語CSVの定番
    return "cp932"


def read_file(file: Any, *, max_rows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    CSV / Excel (xls/xlsx) を読み込んで DataFrame を返す。
    成功時: (DataFrame, None), 失敗時: (None, "エラーメッセージ")
    """
    if file is None:
        return None, "ファイルが指定されていません。"

    # 拡張子
    filename = getattr(file, "name", "") or ""
    lower = filename.lower()

    # ---- CSV -------------------------------------------------------------------
    if lower.endswith(".csv"):
        try:
            # バイト全体を一度確保（StreamlitのUploadedFileはファイルライク）
            raw: bytes = file.read() if hasattr(file, "read") else bytes(file)
            if not raw:
                return None, "CSVが空です。"

            # 先にサンプルで方針を決める
            enc_guess = _sniff_csv_encoding(raw[:4096])

            # 推測順に再試行
            for enc in [enc_guess, "utf-8-sig", "cp932", "utf-8"]:
                if not enc:
                    continue
                try:
                    df = _try_read_csv(io.BytesIO(raw), enc, max_rows)
                    return df, None
                except Exception:
                    continue

            # 最後の手段：delimiter を sniff
            try:
                sample_text = raw[:65536].decode(enc_guess or "utf-8", errors="ignore")
                dialect = csv.Sniffer().sniff(sample_text)
                df = pd.read_csv(io.StringIO(sample_text), dialect=dialect, dtype="object", keep_default_na=False)
                return _clean_columns(df), None
            except Exception as e:
                return None, f"CSV の解析に失敗しました: {e}"
        finally:
            # Streamlitの UploadedFile は read() 後にカーソルが末尾になるため、次回の read に備えて巻き戻し
            try:
                file.seek(0)
            except Exception:
                pass

    # ---- Excel (xlsx/xls) ------------------------------------------------------
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        try:
            if lower.endswith(".xlsx"):
                # openpyxl で読む
                df = pd.read_excel(file, engine="openpyxl", dtype="object", nrows=max_rows)
            else:
                # xls は xlrd（pandas 側の推奨に従う）
                df = pd.read_excel(file, engine="xlrd", dtype="object", nrows=max_rows)
            return _clean_columns(df), None
        except Exception as e:
            return None, f"Excel の読み込みに失敗しました: {e}"

    # ---- 未対応 -----------------------------------------------------------------
    else:
        return None, f"未対応のファイル形式です: {filename}"


# ============================ 書き出しユーティリティ ===============================

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: str, *, index: bool = False, encoding: str = "utf-8-sig") -> Tuple[bool, Optional[str]]:
    """
    DataFrame を安全に CSV 保存（UTF-8-SIG）。成功/失敗とメッセージを返す。
    """
    try:
        _ensure_parent_dir(path)
        df.to_csv(path, index=index, encoding=encoding)
        return True, None
    except Exception as e:
        return False, f"CSV 保存に失敗しました: {e}"


def safe_to_json(data: Any, path: str, *, ensure_ascii: bool = False, indent: int = 2) -> Tuple[bool, Optional[str]]:
    """
    任意オブジェクトを JSON 保存。pandas.DataFrame は records で保存。
    """
    import json

    try:
        _ensure_parent_dir(path)
        if isinstance(data, pd.DataFrame):
            payload = json.loads(data.to_json(orient="records", force_ascii=ensure_ascii))
        else:
            payload = data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent)
        return True, None
    except Exception as e:
        return False, f"JSON 保存に失敗しました: {e}"


# ============================ そのほか（便利ヘルパ） ===============================

def mk_tmpfile(suffix: str = "", prefix: str = "tmp_", dir: Optional[str] = None) -> str:
    """
    一時ファイルパスを返す（ファイルは作成しない）。例：/tmp/tmp_abcd1234.jsonl
    """
    dir = dir or tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    try:
        os.close(fd)  # 空のまま閉じる
    except Exception:
        pass
    return path


def to_s3_key(prefix: str, filename: str, *, timestamped: bool = True) -> str:
    """
    S3 のキー（オブジェクトパス）を生成。prefix に末尾スラッシュを付与し、必要ならタイムスタンプを付ける。
    """
    p = prefix.rstrip("/") + "/"
    base = os.path.basename(filename)
    if timestamped:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        name, ext = os.path.splitext(base)
        base = f"{name}-{ts}{ext}"
    return f"{p}{base}"