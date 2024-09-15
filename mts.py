import streamlit as st
from typing import Any, Dict
from lyzr_automata.ai_models.model_base import AIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from dotenv import load_dotenv
import os
import httpx
import json  # Для логирования JSON-запросов

# Load environment variables
load_dotenv()
api_key = os.getenv("COTYPE_API_KEY")  # Bearer token for COTYPE API
model_endpoint = "https://demo5-fundres.dev.mts.ai/v1/chat/completions"

# Define the Custom Model class inheriting from AIModel
class CustomCotypeModel(AIModel):
    def __init__(self, key, parameters: Dict[str, Any]):
        self.api_key = key  # Store API key (Bearer token)
        self.parameters = parameters  # Store parameters like temperature and max_tokens

    # Implement the abstract method generate_text (required by AIModel)
    def generate_text(self, prompt: str, task_id: str = None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # Set the Bearer token for authorization
            "Content-Type": "application/json"
        }

        # Update system message and user prompt
        json_payload = {
            "model": "cotype_pro_16k_1.1",
            "messages": [
                {
                    "role": "system",
                    "content": "Ты помощник, создающий вопросы MCQ с вариантами ответов по предоставленной теме. Весь ответ, включая вопросы, варианты ответов и любые пояснения, должен быть на русском языке."
                },
                {
                    "role": "user",
                    "content": f"Сгенерируй 10 вопросов с вариантами ответов по теме: {prompt}. Весь ответ, включая вопросы и варианты ответов, должен быть только на русском языке."
                }
            ],
            "temperature": self.parameters.get("temperature", 0.2),
            "max_tokens": self.parameters.get("max_tokens", 1500),
        }

        # Log the request payload for debugging
        print(f"Request payload:\n{json.dumps(json_payload, indent=2, ensure_ascii=False)}")

        # Set a timeout of 60 seconds and send the request
        try:
            response = httpx.post(model_endpoint, headers=headers, json=json_payload, timeout=60.0)
        except httpx.RequestError as exc:
            raise Exception(f"An error occurred while requesting {exc.request.url!r}.") from exc
        except httpx.TimeoutException as exc:
            raise Exception(f"Request timed out: {exc}.") from exc

        # Log the response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response content:\n{response.text}")

        # Check if the request was successful and return the content, otherwise raise an exception
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Implement the abstract method generate_image (required by AIModel)
    def generate_image(self, prompt: str) -> None:
        raise NotImplementedError("This model does not support image generation.")

# Streamlit UI setup
st.set_page_config(
    page_title="МТС. AI Mentor",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS to control formatting
st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    .stApp {
        background-color: #8B0000;  /* Dark red background */
    }
    .st-emotion-cache-vj1c9o {
        background-color: #8B0000 !important;  /* Dark red footer */
        color: white !important;
    }
    .css-12ttj6m {
        background-color: #8B0000 !important;  /* Dark red footer for specific Streamlit classes */
    }
    .mcq-answers {
        margin-left: 30px;  /* Shift answers to the right */
        line-height: 1.2;  /* Reduce line spacing for answers */
    }
    .mcq-correct-answer {
        margin-left: 30px;  /* Shift correct answer to the right */
        font-style: italic;  /* Make the correct answer italic */
        color: gray;  /* Make the text color gray */
        font-size: inherit;  /* Ensure the font size is the same as the rest of the text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("МТС. AI Mentor")
st.markdown("### MCQ Generator")
st.markdown("Загрузите учебные материалы и получите вопросы по теме.")

# Initialize the custom model
cotype_model = CustomCotypeModel(
    key=api_key,
    parameters={
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

# Capture user input
topic = st.chat_input("Введите тему")  # Поле ввода
if topic:
    ielts_agent = Agent(
        role="Ielts expert",
        prompt_persona=f"Сгенерируй 10 вопросов с вариантами ответов по теме: {topic}. Ответ должен быть полностью на русском языке, включая вопросы, ответы и любые пояснения."
    )

    # Create the task with the prompt and model
    ielts_task = Task(
        name="get Ielts study",
        model=cotype_model,
        agent=ielts_agent,
        instructions=f"Сгенерируй 10 вопросов с вариантами ответов по теме: {topic}. Ответ должен быть полностью на русском языке, включая вопросы, ответы и любые пояснения.",
    )

    # Run the task when the user provides a topic
    output = LinearSyncPipeline(
        name="Ielts details",
        completion_message="pipeline completed",
        tasks=[
            ielts_task  # Add the task to the pipeline
        ],
    ).run()

    # Format output to apply CSS only to answers and correct answer
    formatted_output = output[0]['task_output'].split("\n")  # Split the output by lines
    final_output = ""
    for line in formatted_output:
        if "Вот 10 вопросов" in line:  # Убираем строку с "Вот 10 вопросов с вариантами ответов по теме"
            continue
        if "Ответ:" in line:
            final_output += f"<div class='mcq-correct-answer'>{line}</div>"  # Correct answer style (italic, gray)
        elif line.startswith("А)") or line.startswith("Б)") or line.startswith("В)") or line.startswith("Г)"):
            final_output += f"<div class='mcq-answers'>{line}</div>"  # Answer options style
        else:
            final_output += f"<div>{line}</div>"  # Default for questions

    st.markdown(final_output, unsafe_allow_html=True)
