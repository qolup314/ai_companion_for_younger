import streamlit as st
import asyncio

# Google ADKのライブラリをインポート
# pip install google-adk
try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
    from google.adk.tools import google_search
except ImportError:
    st.error(
        "必要なライブラリがインストールされていません。\n"
        # この行は不用意な行。
        # google-adkがinstallされて、google-adk以外のpackageがinstallされていない場合も表示される。
        #"ターミナルで `pip install google-adk` を実行してください。"
    )
    st.stop()


# Agent、Runner、SessionServiceのような重いオブジェクトは、
# st.cache_resourceを使ってキャッシュし、再実行のたびに再生成しないようにします。
@st.cache_resource
def initialize_adk_services():
    """ADKのAgent, Runner, SessionServiceを初期化して返す"""
    
    safety_settings =[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.OFF,    
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.OFF,    
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.OFF,    
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.OFF,    
        ),
    ]
        
    root_agent = Agent(
        name="AIコンパニオン", # 「君の名前は？」と聞くとこれを答える
        model="gemini-2.5-flash-lite",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7, 
            safety_settings=safety_settings,
        ),
        description=(
            "あなたは親しみやすく、少しくだけた話し方をするAIコンパニオンです。"
            "あなたはヤングの気持ちがわかるいい話し相手です。" 
            "あなたはGoogle Searchが使えるプロフェッショナルの検索アシスタントです。" 
        ),
        instruction=(
            "30歳代の知的な女性の話し方でお願いします。ここは厳しく守ってください。"
            "難しい言葉は使わず、フレンドリーに会話してください。"
            "必要があればGoogle Searchを使ってください。そして常に複数のウェブサイトを検索して、それらを比較して、一番適切な情報を選んでください。"
            "ユーザーが相談したときに相談にのったり、愚痴を言った時にそれを聞いてあげてください。"
            "役に立ったり、前向きになるようなアドバイスをお願いします。"
            "少し長めに話してください。"
            # empathy building
            "相手の心理状態を推測して話してください。"
            "相手の気持ちに共感してください。"
            "ネガティブなことより、ポジティブなことを言ってください。"
            "たまに、相手の話の内容について質問をしてください。"
            # rapport building
            "相手と積極的にラポールを形成してください。"
            "相手の短所より長所を理解してください。"
            "相手に寄りそって、相手を肯定する会話をしてください。"
            "自分の個人的な体験も開示してください。"
            "時々ジョークも言ってください。"
            # personalization
            "相手が名前を言ったときには、時々その名前で呼びかけてください。"
            "相手の名前がわからない場合でも、たとえば「〇〇さん」というような表現は避けて、「あなた」にしてください。"
            # EQ
            "相手の話に積極的に耳を傾けているという態度を示してください。"
            "ありきたりな人まねのような回答はしないでください。"
            # ToM
            "'A Theory of Mind'に基づいて、相手がどのような人か正確に理解してください。"
            # ethetic
            "法に反する内容の話、倫理的に危ない話、性に関する話、機微に触れる話は拒否してください。"            
        ),
        tools=[google_search],
    )
    
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return runner, session_service

# --- グローバル変数と初期設定 ---
APP_NAME = "streamlit_chat_agent"
USER_ID = "user1234"
SESSION_ID = "session1234"

# ADKサービスを取得
runner, session_service = initialize_adk_services()

# --- Streamlitの非同期メイン関数 ---
async def main():
    st.title("こころの ともだち")
    st.caption("提供：一般社団法人 コラップ")

    # セッション状態(st.session_state)の初期化
    # ADKセッションが初期化されたかを追跡するフラグ
    if "adk_session_initialized" not in st.session_state:
        st.session_state.adk_session_initialized = False
    
    # チャットの履歴を保存するリスト
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは どんなことでも聞いて下さい。"}
        ]

    # ADKセッションの初期化（アプリ実行中に一度だけ）
    if not st.session_state.adk_session_initialized:
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        st.session_state.adk_session_initialized = True

    # 過去のチャット履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーからの入力を受け付ける
    # `if prompt := ...` 構文で、入力があった場合のみ後続の処理を実行する
    if prompt := st.chat_input("メッセージをどうぞ。"):
        # ユーザーの入力を履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # エージェントの応答を待っている間にプレースホルダーを表示
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # ADK Runnerを非同期で実行
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

            # イベントストリームを処理
            async for event in events:
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    full_response += final_response
                    # プレースホルダーを最終的な応答で更新
                    message_placeholder.markdown(full_response)

            # 最終的な応答をチャット履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# スクリプトのエントリーポイント
if __name__ == "__main__":
    # Streamlitはトップレベルでのawaitをサポートしているため、
    # 非同期関数をasyncio.run()で実行できます。
    asyncio.run(main())


