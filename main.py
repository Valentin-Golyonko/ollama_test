import logging
from datetime import timedelta
from functools import wraps
from time import perf_counter

from ollama import Client

logger = logging.getLogger(__name__)

ollama_client = Client(host="http://localhost:11434")


def time_it(func):
    @wraps(func)
    def run_time(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        # logger.debug(f"'{func.__name__}' = {timedelta(seconds=perf_counter() - t0)}")
        print(f"\t--- '{func.__name__}' = {timedelta(seconds=perf_counter() - t0)} ---")
        return result

    return run_time


@time_it
def try_ollama() -> None:
    response = ollama_client.chat(
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

    content: str = response.get("message", {}).get("content")
    print(content)

    return None


if __name__ == "__main__":
    try_ollama()

    """
    0:00:09.859082
    0:00:10.065670
    0:00:10.553045
    """