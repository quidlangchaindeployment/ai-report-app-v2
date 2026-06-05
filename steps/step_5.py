# steps/step_5.py
# --- Step 5: PowerPoint 出力 (Output) ---
import io
import json
import base64
from typing import Any, Dict, List, Optional, Callable

import streamlit as st

# --- 共通ユーティリティへの依存 ---
from utils.dependencies import (
    HAS_LLM, get_llm, 
    HAS_PPTX, Presentation, Inches, Pt
)

def _ai_polish(text: str, tone_hint: str = "") -> str:
    """見出しや箇条書きの言い回しを軽く整える。LLM未設定の場合は原文を返す。"""
    if not text.strip() or not HAS_LLM or get_llm is None:
        return text

    llm = get_llm(model_name="gemini-2.5-flash-lite", temperature=0.1, timeout_seconds=30)
    if llm is None: return text

    prompt = (
        "次の文を箇条書きのトーンで、与えられた文体に軽く整えてください。意味は変えないでください。\n"
        f"【トーンの指示】{tone_hint or 'ビジネス/経営会議向け・簡潔'}\n"
        f"---\n{text}\n---\n"
        "出力はテキストのみ。"
    )
    try:
        out = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm.predict(prompt)
        return str(out).strip()
    except Exception:
        return text

def _validate_report_json(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict): return "JSONの最上位がオブジェクトではありません。"
    if "title" not in obj or "slides" not in obj: return "JSON に 'title' または 'slides' がありません。"
    if not isinstance(obj["slides"], list) or not obj["slides"]: return "'slides' が空、または配列ではありません。"
    return None

# =============================================================================
# スライド生成ロジック群
# =============================================================================
def _add_title_slide(prs: 'Presentation', heading: str, bullets: Optional[List[str]] = None):
    layout = prs.slide_layouts[0 if len(prs.slide_layouts) > 0 else 0]
    slide = prs.slides.add_slide(layout)
    if slide.shapes.title: slide.shapes.title.text = heading[:120]
    
    body = next((shp.text_frame for shp in slide.shapes if getattr(shp, "has_text_frame", False) and shp != slide.shapes.title), None)
    if body and bullets:
        body.clear()
        for i, b in enumerate(bullets[:5]):
            if i == 0: body.text = b[:200]
            else: body.add_paragraph().text = b[:200]

def _add_toc_slide(prs: 'Presentation', heading: str, entries: List[str]):
    layout = prs.slide_layouts[1 if len(prs.slide_layouts) > 1 else 0]
    slide = prs.slides.add_slide(layout)
    if slide.shapes.title: slide.shapes.title.text = heading[:120]
        
    body = next((shp.text_frame for shp in slide.shapes if getattr(shp, "has_text_frame", False) and shp != slide.shapes.title), None)
    if body:
        body.clear()
        for i, e in enumerate(entries[:10]):
            if i == 0: body.text = f"• {e[:200]}"
            else: body.add_paragraph().text = f"• {e[:200]}"

def _add_bullets_slide(prs: 'Presentation', heading: str, bullets: List[str]):
    layout = prs.slide_layouts[1 if len(prs.slide_layouts) > 1 else 0]
    slide = prs.slides.add_slide(layout)
    if slide.shapes.title: slide.shapes.title.text = heading[:120]
        
    body = next((shp.text_frame for shp in slide.shapes if getattr(shp, "has_text_frame", False) and shp != slide.shapes.title), None)
    if body:
        body.clear()
        for i, b in enumerate(bullets[:12]):
            if i == 0: body.text = f"• {b[:250]}"
            else: body.add_paragraph().text = f"• {b[:250]}"

def _add_image_slide(prs: 'Presentation', heading: str, image_b64: str, bullets: List[str]):
    """画像をデコードしてスライドに挿入する"""
    layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[1] # Blank or Title and Content
    slide = prs.slides.add_slide(layout)
    if slide.shapes.title: slide.shapes.title.text = heading[:120]
    
    try:
        # 画像のデコードと挿入 (左側)
        image_stream = io.BytesIO(base64.b64decode(image_b64))
        left = Inches(0.5)
        top = Inches(1.5)
        width = Inches(5.0)
        slide.shapes.add_picture(image_stream, left, top, width=width)
        
        # 考察テキストの挿入 (右側)
        txBox = slide.shapes.add_textbox(Inches(5.8), Inches(1.5), Inches(3.8), Inches(5.0))
        tf = txBox.text_frame
        tf.word_wrap = True
        
        for i, b in enumerate(bullets[:10]):
            if i == 0: tf.text = f"• {b[:200]}"
            else: tf.add_paragraph().text = f"• {b[:200]}"
            
    except Exception as e:
        _add_bullets_slide(prs, heading, ["(画像の埋め込みに失敗しました)"] + bullets)

def _add_table_slide(prs: 'Presentation', heading: str, table_dict: Dict[str, Any]):
    layout = prs.slide_layouts[5 if len(prs.slide_layouts) > 5 else (1 if len(prs.slide_layouts) > 1 else 0)]
    slide = prs.slides.add_slide(layout)
    if slide.shapes.title: slide.shapes.title.text = heading[:120]

    columns = list(table_dict.keys())
    rows = max([len(v) for v in table_dict.values() if isinstance(v, list)] or [0])
    
    if not columns or rows == 0:
        _add_bullets_slide(prs, heading, ["(表データの取得に失敗しました)"])
        return

    try:
        tbl_shape = slide.shapes.add_table(rows + 1, len(columns), Inches(0.5), Inches(1.8), Inches(9.0), Inches(5.0))
        tbl = tbl_shape.table
        for j, col_name in enumerate(columns): tbl.cell(0, j).text = str(col_name)
        for i in range(rows):
            for j, col_name in enumerate(columns):
                vals = table_dict.get(col_name, [])
                val = vals[i] if isinstance(vals, list) and i < len(vals) else ""
                tbl.cell(i + 1, j).text = str(val)[:200]
    except Exception:
        _add_bullets_slide(prs, heading, ["（表の描画に失敗したため、箇条書きにフォールバックしました）"])

def _build_presentation_from_report(report_json: Dict[str, Any], template_bytes: Optional[bytes], polish_fn: Optional[Callable[[str], str]] = None) -> bytes:
    if not HAS_PPTX: raise RuntimeError("python-pptx が見つかりません。")

    prs = Presentation(io.BytesIO(template_bytes)) if template_bytes else Presentation()
    title = report_json.get("title", "AI レポート")
    slides = report_json.get("slides", [])
    if polish_fn is None: polish_fn = lambda x: x

    for s in slides:
        layout = str(s.get("layout", "bullets")).lower()
        heading = polish_fn(str(s.get("heading", title))[:120])

        if layout == "title":
            bullets = [polish_fn(b) for b in s.get("bullets", [])] if isinstance(s.get("bullets"), list) else []
            _add_title_slide(prs, heading, bullets)
        elif layout == "toc":
            toc_items = [str(x) for x in s.get("bullets", [])] if isinstance(s.get("bullets"), list) else []
            _add_toc_slide(prs, heading, toc_items)
        elif layout == "image":
            image_b64 = s.get("image_base64")
            bullets = [polish_fn(b) for b in s.get("bullets", [])] if isinstance(s.get("bullets"), list) else []
            if image_b64:
                _add_image_slide(prs, heading, image_b64, bullets)
            else:
                _add_bullets_slide(prs, heading, ["(画像データが見つかりませんでした)"] + bullets)
        elif layout == "table":
            table_dict = s.get("table", {})
            if isinstance(table_dict, dict):
                _add_table_slide(prs, heading, table_dict)
            else:
                _add_bullets_slide(prs, heading, ["（table が不正のため箇条書きにフォールバック）"])
        else:
            bullets = [polish_fn(b) for b in s.get("bullets", [])] if isinstance(s.get("bullets"), list) else []
            _add_bullets_slide(prs, heading, bullets)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()

# =============================================================================
# 画面描画（UI 本体）
# =============================================================================
def render():
    st.title("📑 Step 5: PowerPoint 出力 (Output)")

    if not HAS_PPTX:
        st.error("python-pptx が見つかりません。`pip install python-pptx` を実行してください。")
        return

    st.header("1) レポートJSONの読み込み")
    st.caption("Step 4 で作成した構成データを使用します。")
    uploaded_report = st.file_uploader("report_for_powerpoint.json をアップロード", type=["json"])
    report_json: Optional[Dict[str, Any]] = None

    default_text = ""
    if st.session_state.get("step_c_report_json"):
        default_text = json.dumps(st.session_state["step_c_report_json"], ensure_ascii=False, indent=2)

    text_area = st.text_area("（任意）JSONを直接貼り付け", value=default_text, height=220)

    if uploaded_report:
        try:
            report_json = json.loads(uploaded_report.read().decode("utf-8"))
            st.success("アップロードしたレポートJSONを使用します。")
        except Exception as e:
            st.error(f"JSON の読み込みに失敗: {e}")
            return
    elif text_area.strip():
        try:
            report_json = json.loads(text_area)
            st.info("テキストエリアのレポートJSONを使用します。")
        except Exception as e:
            st.error(f"JSON の解析に失敗: {e}")
            return
    else:
        st.warning("レポートJSONをアップロードするか、テキスト欄に貼り付けてください。")
        return

    err = _validate_report_json(report_json)
    if err:
        st.error(f"レポートJSONが不正: {err}")
        return

    st.markdown("---")
    st.header("2) テンプレPPTX（任意）を読み込む")
    uploaded_tmpl = st.file_uploader("テンプレート（.pptx）", type=["pptx"], key="tmpl_uploader")
    tmpl_bytes = uploaded_tmpl.read() if uploaded_tmpl else None

    st.markdown("---")
    st.header("3) 生成オプション")
    tone_hint = st.text_input("（任意）AI整形のトーン", value="ビジネス・簡潔・結論先出し")
    use_ai_polish = st.checkbox("見出し/箇条書きをAIで軽く整形する", value=HAS_LLM)

    def _polish_fn_local(s: str) -> str:
        return _ai_polish(s, tone_hint) if use_ai_polish else s

    st.markdown("---")
    st.header("4) PowerPoint を生成")
    if st.button("📤 生成してダウンロード", type="primary"):
        with st.spinner("PowerPoint を生成しています..."):
            try:
                ppt_bytes = _build_presentation_from_report(
                    report_json=report_json,
                    template_bytes=tmpl_bytes,
                    polish_fn=_polish_fn_local,
                )
                st.success("PowerPoint の生成に成功しました。下のボタンからダウンロードできます。")
                st.download_button(
                    "🔽 PPTX をダウンロード",
                    data=ppt_bytes,
                    file_name="ai_report_output.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    type="primary",
                )
            except Exception as e:
                st.error(f"PowerPoint 生成エラー: {e}")

    st.markdown("---")
    st.header("5) レポートJSONのプレビュー")
    with st.expander("JSON（再掲）", expanded=False):
        st.json(report_json)