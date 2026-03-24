# steps/step_c.py
# --- Step C: AIレポート生成（v2） -------------------------------------------------
# 役割：
# - Step B の JSON/JSONL（分析結果の要約や表、画像情報など）を読み込み
# - LLM（Gemini; services.llm.get_llm）があれば、指示プロンプトに基づき「レポートJSON（章/スライド構成）」を生成
# - LLM未移植やAPI未設定でも、フォールバック（ヒューリスティック）でレポートJSONの雛形を生成
# - 生成したレポートJSONをプレビューし、ダウンロードできるようにする
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st

# --- LLM の遅延インポート（未移植でも UI が落ちないように） -----------------------
_HAS_LLM = False
try:
    from services.llm import get_llm  # Gemini クライアント
    _HAS_LLM = True
except Exception:
    get_llm = None

# --- 小さなユーティリティ ---------------------------------------------------------
def _parse_uploaded_json_or_jsonl(text: str) -> Dict[str, Any]:
    """
    Step B で出力した JSONL or JSON を受け取り、統一的な dict に整形して返す。
    - JSONL であれば行ごとに json.loads → list へ
    - JSON であればそのまま dict/list を包む
    """
    text = text.strip()
    # JSON Lines (複数行）と仮定
    if "\n" in text and not text.startswith("{") and not text.startswith("["):
        items: List[Any] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                # 行に ```json ... ``` が混ざるなどのゆらぎに備える
                m = re.search(r"\{.*\}", s, re.DOTALL)
                if m:
                    try:
                        items.append(json.loads(m.group(0)))
                    except Exception:
                        pass
        return {"type": "jsonl", "items": items, "raw": text}

    # JSON（dict or list）
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return {"type": "list", "items": obj, "raw": text}
        elif isinstance(obj, dict):
            return {"type": "dict", "items": [obj], "raw": text}
        else:
            return {"type": "unknown", "items": [], "raw": text}
    except Exception:
        # どうしてもパースできなければ raw として保持
        return {"type": "raw", "items": [], "raw": text}


def _fallback_report_from_items(items: List[Dict[str, Any]], user_goal: str) -> Dict[str, Any]:
    """
    LLM を使わずに、Step B の items から大枠のスライド構成を組むフォールバック。
    - 先頭数件の summary や data のキーを拾って “章立てのタネ” にする。
    """
    summaries: List[str] = []
    sample_tables: List[Dict[str, Any]] = []
    for it in items[:8]:  # 軽く上位のみ
        s = it.get("summary") if isinstance(it, dict) else None
        if isinstance(s, str) and s.strip():
            summaries.append(s.strip())
        d = it.get("data") if isinstance(it, dict) else None
        if isinstance(d, dict):
            # テーブル候補として dict はそのまま1件保存
            sample_tables.append(d)

    title = "分析レポート（ドラフト）"
    if user_goal.strip():
        title = f"分析レポート（ドラフト）—{user_goal[:50]}"

    # 最低限の構造：タイトル / 目次 / サマリ / 参考表
    report = {
        "title": title,
        "slides": [
            {
                "layout": "title",
                "heading": title,
                "bullets": ["本レポートは自動生成のドラフトです", "Step B の結果をもとに仮説ベースの要点を抜粋"],
            },
            {
                "layout": "toc",
                "heading": "目次",
                "bullets": ["サマリ", "主要指標の概況", "ハイライト/仮説", "参考表"],
            },
            {
                "layout": "bullets",
                "heading": "サマリ（ドラフト）",
                "bullets": summaries[:5] or ["Step B の結果から主要な論点を抽出予定（LLM未使用のため簡易要約）。"],
            },
        ],
        "notes": {
            "source_count": len(items),
            "has_tables": bool(sample_tables),
        },
    }
    if sample_tables:
        report["slides"].append(
            {
                "layout": "table",
                "heading": "参考表（上位サンプル）",
                "table": sample_tables[0],
            }
        )
    return report


def _llm_build_report(items: List[Dict[str, Any]], user_goal: str, style_hint: str) -> Optional[Dict[str, Any]]:
    """
    Gemini（services.llm.get_llm）を用いて、章立て・スライド構成の JSON を生成。
    - 未移植/API未設定時は None を返す → フォールバックへ。
    """
    if not _HAS_LLM or get_llm is None:
        return None

    llm = get_llm(model_name="gemini-2.5-pro", temperature=0.2, timeout_seconds=180)
    if llm is None:
        return None

    # items が長大になりすぎないように先頭数件 & サマリのみ渡す
    items_for_prompt: List[Dict[str, Any]] = []
    for it in items[:12]:
        row = {}
        if isinstance(it, dict):
            if "summary" in it and isinstance(it["summary"], str):
                row["summary"] = it["summary"][:800]
            if "analysis_task" in it:
                row["task"] = str(it["analysis_task"])
            # data は重いのでキーだけ
            if "data" in it and isinstance(it["data"], dict):
                row["data_keys"] = list(it["data"].keys())[:10]
        items_for_prompt.append(row)

    prompt = (
        "あなたはデータ分析レポートの作成アシスタントです。以下の「分析結果の要約（サンプル）」を読み、"
        "実務でそのまま使える**スライド構成のJSON**を生成してください。\n\n"
        "## 目的（自由記述）\n"
        f"{user_goal}\n\n"
        "## 表現スタイル（例: 経営会議向け / 施策提案向け など）\n"
        f"{style_hint}\n\n"
        "## 入力データ（Step B の結果サンプル / 最大12件）\n"
        f"{json.dumps(items_for_prompt, ensure_ascii=False)}\n\n"
        "## 出力フォーマット（厳格なJSON）\n"
        "{\n"
        '  "title": "レポートのタイトル",\n'
        '  "slides": [\n'
        '    {"layout": "title", "heading": "表紙タイトル", "bullets": ["一行要約1","一行要約2"]},\n'
        '    {"layout": "toc", "heading": "目次", "bullets": ["章1","章2","章3"]},\n'
        '    {"layout": "bullets", "heading": "サマリ", "bullets": ["箇条書き..."]},\n'
        '    {"layout": "bullets", "heading": "ハイライト/仮説", "bullets": ["..."]}\n'
        "  ],\n"
        '  "notes": {"source_count": 12, "disclaimer": "モデル生成のため要確認"}\n'
        "}\n\n"
        "説明文やマークダウンは不要、**JSONのみ**を出力してください。"
    )

    try:
        # get_llm 実装に応じて invoke / predict のいずれにも対応
        try:
            raw = llm.invoke(prompt)  # type: ignore
        except Exception:
            raw = llm.predict(prompt)  # type: ignore
        text = str(raw)

        # ```json ... ``` にも耐える抽出
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = (m.group(1) if (m and m.lastindex) else (m.group(0) if m else "")).strip()
        if not json_str:
            return None
        return json.loads(json_str)
    except Exception:
        return None


# --- UI 本体 ----------------------------------------------------------------------
def render():
    st.title("🧠 Step C: AIレポート生成（v2）")

    # セッション状態の初期化
    if "step_c_report_json" not in st.session_state:
        st.session_state["step_c_report_json"] = None

    st.header("1) Step B の出力（JSON / JSONL）を読み込む")
    st.caption("※ Step B の『JSONL を生成』で得たファイル、あるいは同等の JSON/JSONL をアップロードしてください。")

    uploaded = st.file_uploader("analysis_output_stepB.jsonl / .json", type=["jsonl", "json"])
    raw_text = ""
    if uploaded is not None:
        try:
            raw_text = uploaded.read().decode("utf-8")
            parsed = _parse_uploaded_json_or_jsonl(raw_text)
            st.success(f"読み込み成功: type={parsed.get('type')} / items={len(parsed.get('items', []))}")
            with st.expander("プレビュー（先頭 2 件）", expanded=False):
                for i, it in enumerate(parsed.get("items", [])[:2], start=1):
                    st.write(f"● item #{i}")
                    st.json(it)
        except Exception as e:
            st.error(f"読み込みエラー: {e}")
            return
    else:
        st.info("ファイルを選択してください。")
        return

    st.markdown("---")
    st.header("2) レポートの作成条件を入力")
    col1, col2 = st.columns(2)
    with col1:
        user_goal = st.text_area(
            "目的 / 伝えたいこと（自由記述）",
            placeholder="例：広島観光のSNS分析。主要トレンドと施策仮説を経営会議に提案したい。",
            height=100,
        )
    with col2:
        style_hint = st.text_area(
            "表現スタイル（ターゲット / トーン）",
            placeholder="例：経営会議向けに簡潔・定量・結論先出し。",
            height=100,
        )

    st.markdown("---")
    st.header("3) レポートJSONを生成")

    parsed = _parse_uploaded_json_or_jsonl(raw_text)
    items = parsed.get("items", [])
    use_llm = st.checkbox("LLM（Gemini）で高精度生成を使う", value=_HAS_LLM)

    if st.button("🧪 生成する", type="primary"):
        report_json: Optional[Dict[str, Any]] = None

        if use_llm:
            with st.spinner("LLM（Gemini）でレポートJSONを生成中..."):
                report_json = _llm_build_report(items, user_goal or "", style_hint or "")

        if report_json is None:
            with st.spinner("フォールバック（ヒューリスティック）で雛形を生成中..."):
                # LLM 未使用/失敗時の簡易レポート
                report_json = _fallback_report_from_items(items, user_goal or "")

        st.session_state["step_c_report_json"] = report_json
        st.success("レポートJSONを生成しました。下でプレビュー／ダウンロードできます。")

    # プレビュー & ダウンロード
    if st.session_state.get("step_c_report_json"):
        st.markdown("---")
        st.header("4) 結果プレビュー & ダウンロード")

        with st.expander("レポートJSON（全体）", expanded=True):
            st.json(st.session_state["step_c_report_json"])

        # ダウンロード
        st.download_button(
            "レポートJSONをダウンロード",
            data=json.dumps(st.session_state["step_c_report_json"], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="report_for_powerpoint.json",
            mime="application/json",
            type="primary",
        )

        st.info("この JSON を Step D に渡すと、PowerPoint を自動生成できます。")