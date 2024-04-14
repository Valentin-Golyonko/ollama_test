import logging
from datetime import timedelta
from functools import wraps
from time import perf_counter
from typing import Mapping, Any
import json
from ollama import Client

logger = logging.getLogger(__name__)

ollama_client = Client(host="http://localhost:11434")


def time_it(func):
    @wraps(func)
    def run_time(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        # logger.debug(f"'{func.__name__}' = {timedelta(seconds=perf_counter() - t0)}")
        print(f"--- '{func.__name__}' = {timedelta(seconds=perf_counter() - t0)} ---")
        return result

    return run_time


@time_it
def try_ollama_chat() -> None:
    model_out: Mapping[str, Any] = ollama_client.chat(
        model="llama2",  # mistral
        messages=[
            {
                "role": "user",
                "content": (
                    f"Отредактируй текст и исправь ошибки правописания."
                    f" Верни только исправленный текст."
                    f" Язык текста - русский."
                ),
            },
            {
                "role": "user",
                "content": f"""Текст:
        
                Серёзные рибята получили радосное известие. 
                Всюду звенят децкие галаса. 
                Вшколу при ехал извесный песатель. 
                Он будет четать сваи интерестные рассказы. 
                Щясливые школьники собрались на месный празник.
                """,
            },
        ],
        stream=False,
        # format="json",
    )

    content: str = model_out.get("message", {}).get("content")
    print(content)

    return None


@time_it
def try_ollama_generate() -> None:
    """
    docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

    ollama model: llama2 | mistral | gemma | codegemma |
    """

    person_level = "junior"

    model_out: Mapping[str, Any] = ollama_client.generate(
        model="llama2",
        prompt=(
            f"""You are professional Python software engineer."""
            f"""Now you are hiring a new person for your new project."""
            f"""Your goal is to evaluate a candidate's abilities."""
            f"""You should generate ten different interview questions for {person_level} python developer."""
            f"""Python version should be 3.6 and higher."""
            f"""Your response should be a JSON, """
            f"""where all ten questions should be structured like a """
            f"""list of dictionaries with 'type', 'question', 'complexity' (from 1 to 5) keys."""
        ),
        format="json",
        stream=False,
        keep_alive="5m",
    )

    created_at = model_out.get("created_at", "")

    """TARGET"""
    response = model_out.get("response", 0)

    done = model_out.get("done", False)

    """time spent generating the response | nanoseconds"""
    total_duration = model_out.get("total_duration", 0)

    """time spent in nanoseconds loading the model"""
    load_duration = model_out.get("load_duration", 0)

    """number of tokens in the prompt"""
    prompt_eval_count = model_out.get("prompt_eval_count", 0)

    """time spent in nanoseconds evaluating the prompt"""
    prompt_eval_duration = model_out.get("prompt_eval_duration", 0)

    """number of tokens the response"""
    eval_count = model_out.get("eval_count", 0)

    """time in nanoseconds spent generating the response"""
    eval_duration = model_out.get("eval_duration", 0)

    """
    an encoding of the conversation used in this response,
    this can be sent in the next request to keep a conversational memory
    """
    context = model_out.get("context", [0])  #

    _response: dict = json.loads(response)
    questions: list[dict[str, str]] = _response.get("questions", [])
    for count, question in enumerate(questions, 1):
        print(
            f"""#{count}\n"""
            f"""Type: {question.get('type')}\n"""
            f"""Complexity: {question.get('complexity')}\n"""
            f"""Question: {question.get('question')}\n"""
        )

    return None


if __name__ == "__main__":
    # try_ollama_chat()
    """
    0:00:09.859082
    0:00:10.065670
    0:00:10.553045
    """

    try_ollama_generate()
    """
    
    """
