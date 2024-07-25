import streamlit as st
from typing import Any, Dict, List
from lyzr_automata.ai_models.perplexity import PerplexityModel
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from dotenv import load_dotenv
import os
import httpx

load_dotenv()
proxy_url = os.getenv("OPENAI_PROXY_URL")
api_key = os.getenv("OPENAI_API_KEY")

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

model = PerplexityModel(
    api_key=api_key,
    parameters={
        "model": "pplx-70b-online",
    },
)

topic = st.chat_input("Enter Topic")
agent = Agent(
    role="Ielts expert",
    prompt_persona=f"Ваша задача — разработать 10 ВОПРОСОВ (MCQ) по {topic}, а также, дать на них ответы. Отвечать нужно только на русском языке"
)

task = Task(
    name="get Ielts study",
    model=model,
    agent=agent,
    instructions="Give 10 MCQ Questions with answers",
)

if topic:
    output = LinearSyncPipeline(
        name="Ielts details",
        completion_message="pipeline completed",
        tasks=[
            # tasks are instance of Task class
            task  # Task C
        ],
    ).run()

    print(output[0]['task_output'])
    st.markdown(output[0]['task_output'])
