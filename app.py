import streamlit as st
from typing import Any, Dict, List
from openai import OpenAI
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from dotenv import load_dotenv
import os
import httpx

load_dotenv()
proxy_url = os.getenv("OPENAI_PROXY_URL")
api_key = os.getenv("OPENAI_API_KEY")


class CustomOpenAIModel(OpenAIModel):
    def __init__(self, key, proxy, parameters: Dict[str, Any]):
        self.parameters = parameters
        # self.client = OpenAI(api_key=key) if proxy is None or proxy == "" else OpenAI(api_key=key,
        #                                                                              http_client=httpx.Client(
        #                                                                                  proxy=proxy))

        self.client = OpenAI(api_key=key, http_client=httpx.Client(proxy=proxy))


st.set_page_config(
    page_title="AI Mentor",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Mentor")
st.markdown("### MCQ Generator")
st.markdown("Загрузите учебные материалы и получите вопросы по теме.")

open_ai_text_completion_model = CustomOpenAIModel(
    key=api_key,
    proxy=proxy_url,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

topic = st.chat_input("Enter Topic")
ielts_agent = Agent(
    role="Ielts expert",
    prompt_persona=f"Ваша задача — разработать 10 ВОПРОСОВ (MCQ) по {topic}, а также, дать на них ответы. Отвечать нужно только на русском языке"
)

ielts_task = Task(
    name="get Ielts study",
    model=open_ai_text_completion_model,
    agent=ielts_agent,
    instructions="Give 10 MCQ Questions with answers",
)

if topic:
    output = LinearSyncPipeline(
        name="Ielts details",
        completion_message="pipeline completed",
        tasks=[
            # tasks are instance of Task class
            ielts_task  # Task C
        ],
    ).run()

    print(output[0]['task_output'])
    st.markdown(output[0]['task_output'])
