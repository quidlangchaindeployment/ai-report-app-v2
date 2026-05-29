# services/browser_agent.py

import os
import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from browser_use import Agent
    from browser_use.browser.browser import Browser, BrowserConfig
except ImportError as e:
    st.error(
        "browser-use のインポートに失敗しました。Docker環境で pip install browser-use が実行されているか確認してください。"
    )
    raise e


async def run_quid_metrics_agent(topic_name: str):
    """
    QUID Monitorからメトリクスを取得するAIエージェント
    """

    browser = None  # finallyで確実にcloseするため

    try:
        # ✅ Docker環境判定
        is_docker = os.path.exists("/.dockerenv")

        # ✅ 1. Browser設定
        browser = Browser(
            config=BrowserConfig(
                headless=True if is_docker else False,  # Dockerでは画面なし必須
                disable_security=True,
                user_data_dir="./tmp/quid_profile"
            )
        )

        # ✅ 2. LLM（Gemini）
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ✅ 3. タスク
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

        # ✅ 4. Agent生成
        agent = Agent(
            task=task_description,
            llm=llm,
            browser=browser
        )

        # ✅ 5. 実行
        history = await agent.run(max_steps=20)

        # ✅ 安全に結果取得（null対策）
        if history and hasattr(history, "final_element"):
            return history.final_element()
        else:
            return "結果取得に失敗しました（historyが空）"

    except Exception as e:
        st.error(f"エージェント実行エラー: {str(e)}")
        return f"エージェント実行エラー: {str(e)}"

    finally:
        # ✅ Browserクリーンアップ（重要）
        if browser is not None:
            try:
                await browser.close()
            except:
                pass