import copy
import os
from functools import cache

import dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI, OpenAI
from tiktoken import encoding_for_model
from tokenizers.tokenizers import Encoding


def get_openai_client(use_langchain=False, model_name=None, temperature=None):
    """Load the openai client from the environment variables."""

    dotenv.load_dotenv()

    api_type = os.environ.get("OPENAI_API_TYPE", "openai")

    if api_type == "openai":
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=os.getenv("OPENAI_MAX_RETRIES", 5),
        )

    if api_type == "azure":
        if use_langchain:
            # replace the openai environment variables with the azure ones
            # otherwise the azure client will not work
            if "OPENAI_API_BASE" in os.environ.keys():
                os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["OPENAI_API_BASE"]
                del os.environ["OPENAI_API_BASE"]
            if "OPENAI_API_KEY" in os.environ.keys():
                os.environ["AZURE_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
                del os.environ["OPENAI_API_KEY"]

            return AzureChatOpenAI(
                api_version=" 2024-10-01-preview",
                azure_deployment=model_name,
                model_name=model_name,
                temperature=temperature,
            )

        else:
            client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
                azure_endpoint=os.getenv("OPENAI_API_BASE"),
                max_retries=os.getenv("OPENAI_MAX_RETRIES", 5),
            )

            print(client)
            print(os.getenv("OPENAI_API_BASE"))
            return client


    raise ValueError(f"Unknown api type {api_type}")


def get_number_of_tokens(text: str, model: str = "gpt-35-turbo") -> int:
    tokenizer = _get_encoding_for_model(model)
    return len(tokenizer.encode(text))


@cache
def _get_encoding_for_model(model: str) -> Encoding:
    return encoding_for_model(model)


def pop_message_untill_less_tokens_then(messages: list[dict], tokens: int) -> dict:
    """Ensure that the text fits into the token limit of the model"""
    messages = copy.deepcopy(messages)

    while get_n_tokens_in_message(messages) > tokens:
        messages.pop(0)

    return messages


def get_n_tokens_in_message(messages: list[dict]):
    """Get the number of tokens in a message"""
    all_text = " ".join(message["content"] for message in messages)
    return get_number_of_tokens(all_text)
