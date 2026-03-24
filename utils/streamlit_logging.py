# utils/streamlit_logging.py
# -----------------------------------------------------------------------------
# 役割：
# - Python の logging を Streamlit の UI に安全に流すためのハンドラとヘルパを提供
# - Streamlit セッションが無い場合でも落ちずに標準ログへフォールバック
#
# 提供：
#   - class StreamlitLogHandler(logging.Handler)
#   - class StreamlitLogView
#   - get_logger(name: str) -> logging.Logger
#   - attach_streamlit_log_view(level=logging.INFO, max_lines=2000) -> StreamlitLogView
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List

# Streamlit は任意依存：未実行時でも落ちないように扱う
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


# ===== 内部：スレッド安全なメッセージバッファ =====================================

@dataclass
class _LogRecordText:
    created: float
    level: str
    message: str


class _BufferedSink:
    """ログを一時的に溜めるスレッド安全なバッファ。"""
    def __init__(self, maxsize: int = 5000) -> None:
        self._q: "queue.Queue[_LogRecordText]" = queue.Queue(maxsize=maxsize)

    def put(self, item: _LogRecordText) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # 古いものを捨ててでも新しいものを入れる
            try:
                self._q.get_nowait()
                self._q.put_nowait(item)
            except Exception:
                pass

    def drain_all(self) -> List[_LogRecordText]:
        items: List[_LogRecordText] = []
        try:
            while True:
                items.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return items


# ===== Streamlit 画面（コンテナ）へ出力するビュー =================================

class StreamlitLogView:
    """
    Streamlit 側の表示コンテナを保持し、drain() の呼び出し時に UI を更新する。
    """
    def __init__(self, level: int = logging.INFO, max_lines: int = 2000) -> None:
        self.level = level
        self.max_lines = max_lines
        self._sink = _BufferedSink()
        self._lock = threading.Lock()
        self._lines: List[str] = []

        # Streamlit がある場合にのみ UI コンテナを作る
        if _HAS_ST:
            # 1つのエリアにログを描画
            self._container = st.container()
            # CSS で軽く見やすく（任意）
            st.markdown(
                """
                <style>
                .st-logbox {font-family: ui-monospace,Consolas,Monaco,Menlo,monospace;
                            font-size: 12px; white-space: pre-wrap;}
                .st-logbox .level-INFO {color:#1f6feb;}
                .st-logbox .level-WARNING {color:#9e6a03;}
                .st-logbox .level-ERROR {color:#b60205; font-weight:600;}
                .st-logbox .level-DEBUG {color:#57606a;}
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            self._container = None  # type: ignore

    def get_sink(self) -> _BufferedSink:
        return self._sink

    def drain(self) -> None:
        """
        バッファからログを取り出して UI（または内部配列）に反映。
        Streamlit が無い場合は no-op。
        """
        items = self._sink.drain_all()
        if not items:
            return

        with self._lock:
            for it in items:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(it.created))
                line = f"{timestamp} [{it.level}] {it.message}"
                self._lines.append(line)

            # 行数制限
            if len(self._lines) > self.max_lines:
                self._lines = self._lines[-self.max_lines :]

            if _HAS_ST and self._container is not None:
                # レベルごとに色付け（最低限）
                html_lines = []
                for ln in self._lines[-1000:]:  # 描画は直近のみで十分
                    if " [ERROR] " in ln:
                        cls = "level-ERROR"
                    elif " [WARNING] " in ln:
                        cls = "level-WARNING"
                    elif " [DEBUG] " in ln:
                        cls = "level-DEBUG"
                    else:
                        cls = "level-INFO"
                    html_lines.append(f'<div class="{cls}">{ln}</div>')
                self._container.markdown(f'<div class="st-logbox">{"".join(html_lines)}</div>', unsafe_allow_html=True)


# ===== logging.Handler 実装 ========================================================

class StreamlitLogHandler(logging.Handler):
    """
    Python logging の Handler。emit() 時に StreamlitLogView のバッファへ流す。
    View は attach_streamlit_log_view() で作成し、本ハンドラに紐付ける。
    """
    def __init__(self, view: Optional[StreamlitLogView] = None, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._view = view or _get_or_create_global_view(level=level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        packet = _LogRecordText(created=record.created, level=record.levelname, message=msg)
        try:
            self._view.get_sink().put(packet)
        except Exception:
            # View が無い（Streamlit未実行）などの場合は無視
            pass


# ===== グローバル：1セッションに1つのビューを持つための簡易管理 ===================

_GLOBAL_VIEW: Optional[StreamlitLogView] = None
_GLOBAL_LOCK = threading.Lock()

def _get_or_create_global_view(level: int = logging.INFO, max_lines: int = 2000) -> StreamlitLogView:
    global _GLOBAL_VIEW
    if _GLOBAL_VIEW is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_VIEW is None:
                _GLOBAL_VIEW = StreamlitLogView(level=level, max_lines=max_lines)
    return _GLOBAL_VIEW


# ===== パブリック・ヘルパ ==========================================================

def attach_streamlit_log_view(level: int = logging.INFO, max_lines: int = 2000) -> StreamlitLogView:
    """
    画面にログビュー（1つ）をアタッチし、そのインスタンスを返す。
    返ってきたオブジェクトの .drain() をイベントの合間で呼ぶと UI が更新されます。
    """
    return _get_or_create_global_view(level=level, max_lines=max_lines)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    重複ハンドラを避けつつロガーを初期化して返す。
    - 1) 既に StreamlitLogHandler が付いていなければ付与
    - 2) 既に Console(StreamHandler) が無ければ付与
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    has_streamlit_handler = any(isinstance(h, StreamlitLogHandler) for h in logger.handlers)
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, StreamlitLogHandler) for h in logger.handlers)

    if not has_streamlit_handler:
        logger.addHandler(StreamlitLogHandler(level=level))
    if not has_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console)

    return logger