# analysis/config.py
# --- 分析対象列のエイリアス定義（ドメイン知識の一元管理） ---

COLUMN_ALIASES = {
    "text": ["ANALYSIS_TEXT_COLUMN", "text", "content", "本文", "テキスト", "post"],
    "location": ["市区町村キーワード", "location", "city", "地域", "都道府県", "エリア"],
    "category": ["話題カテゴリ", "topic", "category", "カテゴリ", "分類", "ハッシュタグ", "keyword", "キーワード"],
    "engagement": ["eng", "like", "いいね", "エンゲージメント", "retweet", "リツイート", "share", "シェア", "views", "再生数"],
    "date": ["date", "time", "日付", "日時", "created_at", "投稿日"],
    "sentiment": ["センチメント", "sentiment", "感情", "ポジネガ"]
}