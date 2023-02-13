"""Wrapper around Anthropic APIs."""
from typing import Any, Dict, Generator, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class Anthropic(LLM, BaseModel):
    r"""Wrapper around Anthropic large language models.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            import anthropic
            from langchain import Anthropic
            model = Anthropic(model="<model_name>", anthropic_api_key="my-api-key")

            prompt = "What are the biggest risks facing humanity?"
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            response = model(prompt)
    """

    client: Any  #: :meta private:
    model: Optional[str] = None
    """Model name to use."""

    max_tokens_to_sample: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: float = 1.0
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int = 0
    """Number of most likely tokens to consider at each step."""

    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""

    anthropic_api_key: Optional[str] = None

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        anthropic_api_key = get_from_dict_or_env(
            values, "anthropic_api_key", "ANTHROPIC_API_KEY"
        )
        try:
            import anthropic

            values["client"] = anthropic.Client(anthropic_api_key)
            values["HUMAN_PROMPT"] = anthropic.HUMAN_PROMPT
            values["AI_PROMPT"] = anthropic.AI_PROMPT
        except ImportError:
            raise ValueError(
                "Could not import anthropic python package. "
                "Please it install it with `pip install anthropic`."
            )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        return {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anthropic"

    def _get_anthropic_stop(self, stop: Optional[List[str]] = None) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if stop is None:
            stop = []

        # Never want model to invent new turns of Human / Assistant dialog.
        stop.extend([self.HUMAN_PROMPT, self.AI_PROMPT])

        return stop

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        r"""Call out to Anthropic's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "What are the biggest risks facing humanity?"
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                response = model(prompt)

        """
        stop = self._get_anthropic_stop(stop)
        response = self.client.completion(
            model=self.model, prompt=prompt, stop_sequences=stop, **self._default_params
        )
        text = response["completion"]
        return text

    def stream(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        r"""Call Anthropic completion_stream and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens from Anthropic.

        Example:
            .. code-block:: python


                prompt = "Write a poem about a stream."
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                generator = anthropic.stream(prompt)
                for token in generator:
                    yield token
        """
        stop = self._get_anthropic_stop(stop)
        return self.client.completion_stream(
            model=self.model, prompt=prompt, stop_sequences=stop, **self._default_params
        )


class AnthropicInstruct(Anthropic, BaseModel):
    r"""Wrapper around Anthropic class to make it compatible with instruct-type models.

    Emulates instruction following models by wrapping the input prompt
    with "\n\nHuman:" and "\n\nAssistant".

    So
    .. code-block:: python
        anthropic = AnthropicInstruct(...)
        response = anthropic("Write a poem.")

    is equivalent to
    .. code-block:: python
        anthropic = Anthropic(...)
        prompt = "\n\nHuman: Write a poem.\n\nAssistant: Sure, here you go:\n"
        response = anthropic(prompt)
    """

    def _wrap_prompt(self, prompt: str) -> str:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")
        if prompt.startswith(self.HUMAN_PROMPT):
            return prompt  # Already wrapped.
        else:
            return f"{self.HUMAN_PROMPT} {prompt}{self.AI_PROMPT} Sure, here you go:\n"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        r"""Call out to Anthropic's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = anthropic("What are the biggest risks facing humanity?.")
        """
        # Wrap the prompt so it emulates an instruction following model.
        return super()._call(self._wrap_prompt(prompt), stop)

    def stream(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        r"""Call Anthropic completion_stream and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            instruct_mode: Whether to emulate an instruction-following model.

        Returns:
            A generator representing the stream of tokens from Anthropic.

        Example:
            .. code-block:: python

                generator = anthropic.stream("Write a poem about a stream.")
                for token in generator:
                    yield token
        """
        return super().stream(self._wrap_prompt(prompt), stop)
