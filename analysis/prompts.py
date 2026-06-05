# analysis/prompts.py
import json
from typing import List, Dict, Any

def get_ai_summary_batch_prompt(colnames: List[str], head_rows: List[Dict[str, Any]]) -> str:
    return (
        "以下は分析対象データのサンプルです。\n"
        "このデータ全体の傾向を、要点3〜7点の bullet 形式で簡潔にまとめてください。\n"
        "・バズしている投稿傾向\n"
        "・地名/カテゴリの偏り\n"
        "・特性語のパターン\n"
        "など、気づきにつながる仮説も歓迎します。\n\n"
        "【列名】\n"
        f"{json.dumps(colnames, ensure_ascii=False)}\n\n"
        "【先頭サンプル（最大8件）】\n"
        f"{json.dumps(head_rows, ensure_ascii=False)}\n\n"
        "【出力形式】\n"
        "- 箇条書きで簡潔に\n"
        "- 事実と仮説を分けてもよい\n"
        "- JSON ではなくテキスト"
    )

def get_ai_category_insight_prompt(cat_cols: List[str], head_rows: List[Dict[str, Any]]) -> str:
    return (
        "与えられたデータのカテゴリ列ごとに、特徴的な傾向や仮説を bullet で述べてください。\n"
        "特に『高いエンゲージメントをもたらす要因』『地域差』『テーマ別の強さ』などに注目してください。\n\n"
        f"【対象カテゴリ列】{cat_cols}\n"
        "【先頭サンプル（最大8件）】\n"
        f"{json.dumps(head_rows, ensure_ascii=False)}\n\n"
        "【出力形式】テキスト（箇条書き）"
    )

def get_ai_proposal_prompt(col_info_str: str, existing_block: str, user_prompt: str) -> str:
    return (
        "あなたはデータ分析の専門家です。ユーザーの『分析指示』と『データ構造』を読み、"
        "実行可能な『AI考察タスク』を JSON リストで提案してください。\n\n"
        "【データ構造（列と例）】\n"
        f"{col_info_str}\n\n"
        "【既に提案済み/禁止タスク（これらは提案しない）】\n"
        f"{existing_block}\n\n"
        "【ユーザーの分析指示】\n"
        f"{user_prompt}\n\n"
        "【出力（厳格なJSON配列）】\n"
        "[\n"
        "  {\n"
        '    "priority": 5,\n'
        '    "name": "（タスク名）",\n'
        '    "description": "（このタスクでAIに実行させる具体指示＝プロンプト）",\n'
        '    "reason": "ユーザー指示に基づく",\n'
        '    "suitable_cols": [],\n'
        '    "type": "ai"\n'
        "  }\n"
        "]\n"
        "説明文や前置きは不要。JSONのみを返してください。"
    )