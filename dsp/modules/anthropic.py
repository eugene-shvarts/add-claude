import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('anthropic_usage.log')
    ]
)

import functools
import json
from typing import Any, Literal, Optional, cast

import dsp
import backoff
import anthropic

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class Claude(LM):
    """Wrapper around Anthropic's Claude API

    Args:
        model (str, optional): Anthropic model to use. Defaults to "claude-2.1".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "claude-2.1",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        super().__init__(model)

        self.anthropic = anthropic.Anthropic(
            model=model,
            api_key=api_key
        )

        default_model_type = (
            "text"
        )
        self.model_type = model_type if model_type else default_model_type

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.history: list[dict[str, Any]] = []

    def _anthropic_client(self):
        return self.anthropic

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            kwargs["messages"] = [{"role": "user", "content": prompt}]
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = self.chat_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = self.completions_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retreival of Claude completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice.content[0].text
        return choice.completion

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Claude.

        Args:
            prompt (str): prompt to send to Claude
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)
        choices = [response]

        if dsp.settings.log_anthropic_usage:
            self.log_usage(response)

        completions = [self._get_choice_text(c) for c in choices]
        return completions
    
    def chat_request(self, **kwargs):
        return self.anthropic.completions.create(**kwargs)
    
    def completions_request(self, **kwargs):
        return self.anthropic.messages.create(**kwargs)
