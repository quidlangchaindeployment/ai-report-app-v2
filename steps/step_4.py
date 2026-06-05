# steps/step_4.py
# --- Step 4: レポート生成 (Draft) ---
import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st

# --- 共通ユーティリティへの依存 ---
from utils.dependencies import HAS_LLM, get_llm
# utils.session_manager 側で状態管理されているため、st.session_state は安全に参照可能

def _parse_uploaded_json_or_jsonl(text: str) -> Dict[str, Any]:
    """Step 3 で出力した JSONL or JSON を受け取り、統一的な dict に整形して返す。"""
    text = text.strip()
    
    if "\n" in text and not text.startswith("{") and not text.startswith("["):
        items: List[Any] = []
        for line in text.splitlines():
            s = line.strip()
            if not s: continue
            try:
                items.append(json.loads(s))
            except Exception:
                m = re.search(r"\{.*\}", s, re.DOTALL)
                if m:
                    try: items.append(json.loads(m.group(0)))
                    except Exception: pass
        return {"type": "jsonl", "items": items, "raw": text}

    try:
        obj = json.loads(text)
        if isinstance(obj, list): return {"type": "list", "items": obj, "raw": text}
        if isinstance(obj, dict): return {"type": "dict", "items": [obj], "raw": text}
        return {"type": "unknown", "items": [], "raw": text}
    except Exception:
        return {"type": "raw", "items": [], "raw": text}

def _fallback_report_from_items(items: List[Dict[str, Any]], user_goal: str) -> Dict[str, Any]:
    """LLM を使わずに大枠のスライド構成を組むフォールバック。"""
    summaries: List[str] = []
    sample_tables: List[Dict[str, Any]] = []
    
    for it in items[:8]:
        s = it.get("summary") if isinstance(it, dict) else None
        if isinstance(s, str) and s.strip(): summaries.append(s.strip())
        d = it.get("data") if isinstance(it, dict) else None
        if isinstance(d, dict): sample_tables.append(d)

    title = f"分析レポート（ドラフト）—{user_goal[:50]}" if user_goal.strip() else "分析レポート（ドラフト）"

    report = {
        "title": title,
        "slides": [
            {
                "layout": "title",
                "heading": title,
                "bullets": ["本レポートは自動生成のドラフトです", "Step 3 の結果をもとに仮説ベースの要点を抜粋"],
            },
            {
                "layout": "toc",
                "heading": "目次",
                "bullets": ["サマリ", "主要指標の概況", "ハイライト/仮説", "参考表"],
            },
            {
                "layout": "bullets",
                "heading": "サマリ（ドラフト）",
                "bullets": summaries[:5] or ["Step 3 の結果から主要な論点を抽出予定（LLM未使用のため簡易要約）。"],
            },
        ],
        "notes": {"source_count": len(items), "has_tables": bool(sample_tables)},
    }
    
    if sample_tables:
        report["slides"].append({
            "layout": "table",
            "heading": "参考表（上位サンプル）",
            "table": sample_tables[0],
        })
    return report

def _llm_build_report(items: List[Dict[str, Any]], user_goal: str, style_hint: str) -> Optional[Dict[str, Any]]:
    """Geminiを用いて、章立て・画像を含むスライド構成の JSON を生成。"""
    if not HAS_LLM or get_llm is None:
        return None

    llm = get_llm(model_name="gemini-2.5-pro", temperature=0.2, timeout_seconds=180)
    if llm is None: return None

    # トークンオーバーを防ぐためのチャンク化とメタデータ抽出
    items_for_prompt: List[Dict[str, Any]] = []
    image_pool: Dict[str, str] = {} # task_name と Base64 画像の紐付け辞書

    for it in items[:15]: # 最大15タスクまで処理
        row = {}
        if isinstance(it, dict):
            task_name = str(it.get("analysis_task", "unknown_task"))
            row["task_name"] = task_name
            
            if "summary" in it and isinstance(it["summary"], str): 
                row["summary"] = it["summary"][:800] # サマリを800文字に制限
            if "data" in it and isinstance(it["data"], dict): 
                row["data_keys"] = list(it["data"].keys())[:10]
            
            # 画像データが存在する場合はメタデータとしてフラグを立てる
            if it.get("image_base64"):
                row["has_image"] = True
                image_pool[task_name] = it["image_base64"]
                
        items_for_prompt.append(row)

    prompt = (
        "あなたはデータ分析レポートの作成アシスタントです。以下の「分析結果の要約」を読み、"
        "実務でそのまま使える**スライド構成のJSON**を生成してください。\n\n"
        "## 目的（自由記述）\n"
        f"{user_goal}\n\n"
        "## 表現スタイル\n"
        f"{style_hint}\n\n"
        "## 入力データ（has_image=True のタスクはグラフ画像が存在します）\n"
        f"{json.dumps(items_for_prompt, ensure_ascii=False)}\n\n"
        "## 出力フォーマット（厳格なJSON）\n"
        "{\n"
        '  "title": "レポートのタイトル",\n'
        '  "slides": [\n'
        '    {"layout": "title", "heading": "表紙タイトル", "bullets": ["一行要約1","一行要約2"]},\n'
        '    {"layout": "toc", "heading": "目次", "bullets": ["章1","章2"]},\n'
        '    {"layout": "bullets", "heading": "サマリ", "bullets": ["箇条書き..."]},\n'
        '    {"layout": "image", "heading": "グラフタイトル", "task_name": "（入力データのtask_nameを指定）", "bullets": ["グラフから読み取れる考察"]}\n'
        "  ],\n"
        '  "notes": {"source_count": 12, "disclaimer": "モデル生成のため要確認"}\n'
        "}\n\n"
        "説明文やマークダウンは不要、**JSONのみ**を出力してください。"
    )

    try:
        raw = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        text = str(raw)
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
        json_str = (m.group(1) if (m and m.lastindex) else (m.group(0) if m else "")).strip()
        
        if not json_str: return None
        report_json = json.loads(json_str)
        
        # LLMが指定したタスク名に基づき、画像をJSONに埋め込む
        if "slides" in report_json:
            for slide in report_json["slides"]:
                if slide.get("layout") == "image":
                    tname = slide.get("task_name")
                    if tname in image_pool:
                        slide["image_base64"] = image_pool[tname]
                        
        return report_json
    except Exception:
        return None

# =============================================================================
# 画面描画（UI 本体）
# =============================================================================
def render():
    st.title("🧠 Step 4: レポート生成 (Draft)")

    st.header("1) Step 3 の出力（JSON / JSONL）を読み込む")
    st.caption("※ Step 3 でエクスポートしたファイル、あるいは同等の JSON/JSONL をアップロードしてください。")

    uploaded = st.file_uploader("analysis_output_step3.jsonl / .json", type=["jsonl", "json"])
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
    items = parsed.get("items", [])
    use_llm = st.checkbox("LLM（Gemini）で高精度生成・画像紐付けを使う", value=HAS_LLM)

    if st.button("🧪 生成する", type="primary"):
        report_json: Optional[Dict[str, Any]] = None

        if use_llm:
            with st.spinner("LLM（Gemini）でレポートJSONを生成し、グラフ画像を紐付けています..."):
                report_json = _llm_build_report(items, user_goal or "", style_hint or "")

        if report_json is None:
            with st.spinner("フォールバック（ヒューリスティック）で雛形を生成中..."):
                report_json = _fallback_report_from_items(items, user_goal or "")

        st.session_state["step_c_report_json"] = report_json
        st.success("レポートJSONを生成しました。下でプレビュー／ダウンロードできます。")

    if st.session_state.get("step_c_report_json"):
        st.markdown("---")
        st.header("4) 結果プレビュー & ダウンロード")

        with st.expander("レポートJSON（全体）", expanded=True):
            st.json(st.session_state["step_c_report_json"])

        st.download_button(
            "レポートJSONをダウンロード",
            data=json.dumps(st.session_state["step_c_report_json"], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="report_for_powerpoint.json",
            mime="application/json",
            type="primary",
        )

        if st.button("✅ 次のステップ (PowerPoint出力) へ進む"):
            st.session_state["current_step"] = 5
            st.rerun()