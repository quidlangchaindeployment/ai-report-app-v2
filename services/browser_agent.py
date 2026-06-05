# services/browser_agent.py

import os
import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ BrowserConfig のインポートを削除し、Agent と Browser のみインポート
from browser_use import Agent, Browser

async def run_quid_metrics_agent(topic_name: str):
    """
    QUID Monitorからメトリクスを取得するAIエージェント
    """
    browser = None  # finallyで確実にcloseするため

    try:
        # ✅ 最もシンプルなデフォルト設定でブラウザを初期化
        # （Playwrightのデフォルトで Headless モードとして起動します）
        browser = Browser()

        # ✅ LLM（Gemini）
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ✅ タスク
        task_description = f"""
        1. QUID Monitor (https://monitor.quid.com/) にアクセスしてください。
        2. ログインが必要な場合は、ユーザーがログインを完了するまで待機してください。
        3. トピック「{topic_name}」を検索し、「分析」タブ内の「メトリクス要約（新）」を開きます。
        4. 以下の数値を取得してください：
           - メンション
           - 投稿
           - 感情
           - 情熱度
           - エンゲージメント
           - インプレッション
           - 投稿者数
        5. 結果をJSON形式で返してください。
        """

        # ✅ Agent生成
        agent = Agent(
            task=task_description,
            llm=llm,
            browser=browser
        )

        # ✅ 実行
        history = await agent.run(max_steps=20)

        # ✅ 結果取得
        if history and hasattr(history, "final_element"):
            return history.final_element()
        else:
            return "結果取得に失敗しました（historyが空）"

    except Exception as e:
        st.error(f"エージェント実行エラー: {str(e)}")
        return f"エージェント実行エラー: {str(e)}"

    finally:
        # ✅ Browserクリーンアップ
        if browser is not None:
            try:
                await browser.close()
            except:
                pass