import asyncio
import time
from typing import Dict, List, Tuple

import tiktoken
from anthropic.types import Usage
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from .gen_config import get_exp_setting
from .log_utils import get_logger

logger = get_logger(__name__)

settings = get_exp_setting()
setting_args = {
    "temperature": settings.temperature,
    "top_p": settings.top_p,
}


class TokenCount(BaseModel):
    """Token count of an LLM call"""

    in_token_cnt: int
    out_token_cnt: int

    class Config:
        frozen = True

    def __add__(self, other: "TokenCount"):
        return TokenCount(
            in_token_cnt=self.in_token_cnt + other.in_token_cnt,
            out_token_cnt=self.out_token_cnt + other.out_token_cnt,
        )

    def __str__(self) -> str:
        return f"in {self.in_token_cnt:>8} tokens, out {self.out_token_cnt:>8} tokens"


class TokenCountCached(TokenCount):
    cache_write_cnt: int = 0
    cache_read_cnt: int = 0

    class Config:
        frozen = True

    def __add__(self, other: "TokenCountCached"):
        return TokenCountCached(
            in_token_cnt=self.in_token_cnt + other.in_token_cnt,
            out_token_cnt=self.out_token_cnt + other.out_token_cnt,
            cache_read_cnt=self.cache_read_cnt + other.cache_read_cnt,
            cache_write_cnt=self.cache_write_cnt + other.cache_write_cnt,
        )

    def __str__(self) -> str:
        if not (self.cache_read_cnt or self.cache_write_cnt):
            return super().__str__()
        return (
            f"in {self.in_token_cnt:>8} tokens, "
            f"out {self.out_token_cnt:>8} tokens, "
            f"cache write {self.cache_write_cnt:>8} tokens, "
            f"cache read {self.cache_read_cnt:>8} tokens"
        )


class TokenCost(BaseModel):
    """Token cost of an LLM call"""

    in_token_cost_per_token: float
    out_token_cost_per_token: float


token_costs = {
    "claude-3-5-sonnet-20241022": TokenCost(
        in_token_cost_per_token=3.0 / 1000000, out_token_cost_per_token=15.0 / 1000000
    ),
    "gpt-4o-2024-08-06": TokenCost(
        in_token_cost_per_token=2.5 / 1000000, out_token_cost_per_token=10.0 / 1000000
    ),
    "o1-preview-2024-09-12": TokenCost(
        in_token_cost_per_token=15.0 / 1000000, out_token_cost_per_token=60.0 / 1000000
    ),
    "o1-mini-2024-09-12": TokenCost(
        in_token_cost_per_token=3.0 / 1000000, out_token_cost_per_token=12.0 / 1000000
    ),
    "gpt-4o-2024-05-13": TokenCost(
        in_token_cost_per_token=5.0 / 1000000, out_token_cost_per_token=15.0 / 1000000
    ),
    "gemini-1.5-pro-002": TokenCost(
        in_token_cost_per_token=1.25 / 1000000, out_token_cost_per_token=5.0 / 1000000
    ),
}


class TokenCounter:
    """Token counter based on tiktoken / Anthropic"""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.token_cnts: Dict[str, List[TokenCount]] = {"": []}
        self.token_cnts_lock = asyncio.Lock()
        self.cur_tag = ""
        self.max_parallel_requests: int = 10
        model = llm.metadata.model_name
        if isinstance(llm, OpenAI):
            self.encoding = tiktoken.encoding_for_model(model)
        elif isinstance(llm, Anthropic):
            self.encoding = llm.tokenizer
        else:
            raise Exception(f"gen_config: No tokenizer for model {model}")
        logger.info(f"Found tokenizer for model '{model}'")
        self.token_cost = token_costs[model] if model in token_costs else None
        if self.token_cost is None:
            logger.warning(
                f"Cannot find token cost for model '{model}' in record. Won't display cost in USD"
            )

    def set_cur_tag(self, tag: str) -> None:
        self.cur_tag = tag
        if tag not in self.token_cnts:
            self.token_cnts[tag] = []

    def count(self, string: str) -> int:
        if self.encoding is None:
            return 0
        return len(self.encoding.encode(string))

    def reset(self) -> None:
        self.token_cnts = {"": []}

    def count_chat(
        self, messages: List[ChatMessage], llm: LLM | None = None
    ) -> Tuple[ChatResponse, TokenCount]:
        llm = llm or self.llm
        in_token_cnt = self.count(llm.messages_to_prompt(messages))
        logger.info(
            "TokenCounter count_chat Triggered at temp: %s, top_p: %s"
            % (settings.temperature, settings.top_p)
        )
        response = llm.chat(
            messages, top_p=settings.top_p, temperature=settings.temperature
        )
        out_token_cnt = self.count(response.message.content)
        token_cnt = TokenCount(in_token_cnt=in_token_cnt, out_token_cnt=out_token_cnt)
        self.token_cnts[self.cur_tag].append(token_cnt)
        return (response, token_cnt)

    async def count_achat(
        self, messages: List[ChatMessage], llm: LLM | None = None
    ) -> Tuple[ChatResponse, TokenCount]:
        llm = llm or self.llm
        in_token_cnt = self.count(llm.messages_to_prompt(messages))
        logger.info(
            "TokenCounter count_achat Triggered at temp: %s, top_p: %s"
            % (settings.temperature, settings.top_p)
        )
        response = await llm.achat(
            messages, top_p=settings.top_p, temperature=settings.temperature
        )
        out_token_cnt = self.count(response.message.content)
        token_cnt = TokenCount(in_token_cnt=in_token_cnt, out_token_cnt=out_token_cnt)
        async with self.token_cnts_lock:
            self.token_cnts[self.cur_tag].append(token_cnt)
        return (response, token_cnt)

    async def count_achat_batch(
        self, chat_inputs: List[List[ChatMessage]], llm: LLM | None = None
    ) -> List[Tuple[ChatResponse, TokenCount]]:
        llm = llm or self.llm
        results = []
        for i in range(0, len(chat_inputs), self.max_parallel_requests):
            batch = chat_inputs[i : i + self.max_parallel_requests]
            tasks = [
                self.count_achat(llm=llm, messages=chat_input) for chat_input in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    def count_chat_batch(
        self, chat_inputs: List[List[ChatMessage]], llm: LLM | None = None
    ) -> List[Tuple[ChatResponse, TokenCount]]:
        llm = llm or self.llm
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no current event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        start_time = time.time()
        results = loop.run_until_complete(
            self.count_achat_batch(llm=llm, chat_inputs=chat_inputs)
        )
        logger.info(f"Total batch chat time: {time.time() - start_time:.2f}s")
        return results

    def log_token_stats(self) -> None:
        total_sum_cnt = TokenCount(in_token_cnt=0, out_token_cnt=0)
        for tag in self.token_cnts:
            token_cnt = self.token_cnts[tag]
            if not token_cnt:
                continue
            sum_cnt = sum(token_cnt, start=TokenCount(in_token_cnt=0, out_token_cnt=0))
            assert isinstance(sum_cnt, TokenCount)
            total_sum_cnt += sum_cnt
            logger.info(f"{tag + ' cnt':<25}: {sum_cnt}")
        logger.info((f"{'Total cnt':<25}: {total_sum_cnt}"))
        if self.token_cost:
            total_cost = (
                total_sum_cnt.in_token_cnt * self.token_cost.in_token_cost_per_token
                + total_sum_cnt.out_token_cnt * self.token_cost.out_token_cost_per_token
            )
            logger.info(f"{'Total cost':<25}: ${total_cost:.2f} USD")

    def get_sum_count(self, tag: str | None = None) -> TokenCount:
        # If have tag: return sum of token counts with that tag
        # If no tag: return sum of all token counts
        if tag:
            token_cnt = self.token_cnts[tag]
            sum_cnt = sum(token_cnt, start=TokenCount(in_token_cnt=0, out_token_cnt=0))
        else:
            sum_cnt = TokenCount(in_token_cnt=0, out_token_cnt=0)
            for token_cnt in self.token_cnts.values():
                sum_cnt += sum(
                    token_cnt, start=TokenCount(in_token_cnt=0, out_token_cnt=0)
                )
        assert isinstance(sum_cnt, TokenCount)
        return sum_cnt

    def get_total_token(self) -> int:
        """Return token number regarding to token limit"""
        sum_cnt = TokenCount(in_token_cnt=0, out_token_cnt=0)
        for token_cnt in self.token_cnts.values():
            tag_cnt = sum(token_cnt, start=TokenCount(in_token_cnt=0, out_token_cnt=0))
            assert isinstance(tag_cnt, TokenCount)
            sum_cnt += tag_cnt
        assert isinstance(sum_cnt, TokenCount)
        return sum_cnt.in_token_cnt + sum_cnt.out_token_cnt


class TokenCounterCached(TokenCounter):
    """Token counter with cache based on Anthropic"""

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        assert isinstance(llm, Anthropic)
        self.write_cost_ratio: float = 1.25
        self.read_cost_ratio: float = 0.1
        self.enable_cache = True

    def set_enable_cache(self, enable_cache: bool) -> None:
        self.enable_cache = enable_cache

    def equivalent_cost(self, token_count_cached: TokenCountCached) -> TokenCount:
        equi_cost = round(
            token_count_cached.in_token_cnt
            + token_count_cached.cache_write_cnt * self.write_cost_ratio
            + token_count_cached.cache_read_cnt * self.read_cost_ratio
        )
        return TokenCount(
            in_token_cnt=equi_cost,
            out_token_cnt=token_count_cached.out_token_cnt,
        )

    @classmethod
    def is_cache_enabled(cls, llm: LLM) -> bool:
        return isinstance(llm, Anthropic)

    def add_cache_tag(self, target: ChatMessage) -> None:
        target.additional_kwargs["cache_control"] = {"type": "ephemeral"}

    def count_chat(
        self, messages: List[ChatMessage], llm: LLM | None = None
    ) -> Tuple[ChatResponse, TokenCountCached]:
        llm = llm or self.llm
        logger.info(
            "TokenCounterCached count_chat Triggered at temp: %s, top_p: %s"
            % (settings.temperature, settings.top_p)
        )
        response = llm.chat(
            messages,
            extra_headers=(
                {"anthropic-beta": "prompt-caching-2024-07-31"}
                if self.enable_cache
                else {}
            ),
            top_p=settings.top_p,
            temperature=settings.temperature,
        )
        usage = response.raw["usage"]
        assert isinstance(usage, Usage), f"Unknown usage type: {type(usage)}"
        token_cnt = TokenCountCached(
            in_token_cnt=usage.input_tokens,
            out_token_cnt=usage.output_tokens,
            cache_write_cnt=(
                usage.cache_creation_input_tokens
                if hasattr(usage, "cache_creation_input_tokens")
                else 0
            ),
            cache_read_cnt=(
                usage.cache_read_input_tokens
                if hasattr(usage, "cache_read_input_tokens")
                else 0
            ),
        )
        self.token_cnts[self.cur_tag].append(token_cnt)
        return (response, token_cnt)

    async def count_achat(
        self, messages: List[ChatMessage], llm: LLM | None = None
    ) -> Tuple[ChatResponse, TokenCountCached]:
        llm = llm or self.llm
        logger.info(
            "TokenCounterCached count_achat Triggered at temp: %s, top_p: %s"
            % (settings.temperature, settings.top_p)
        )
        response = await llm.achat(
            messages,
            extra_headers=(
                {"anthropic-beta": "prompt-caching-2024-07-31"}
                if self.enable_cache
                else {}
            ),
            top_p=settings.top_p,
            temperature=settings.temperature,
        )
        usage = response.raw["usage"]
        assert isinstance(usage, Usage), f"Unknown usage type: {type(usage)}"
        token_cnt = TokenCountCached(
            in_token_cnt=usage.input_tokens,
            out_token_cnt=usage.output_tokens,
            cache_write_cnt=(
                usage.cache_creation_input_tokens
                if hasattr(usage, "cache_creation_input_tokens")
                else 0
            ),
            cache_read_cnt=(
                usage.cache_read_input_tokens
                if hasattr(usage, "cache_read_input_tokens")
                else 0
            ),
        )
        async with self.token_cnts_lock:
            self.token_cnts[self.cur_tag].append(token_cnt)
        return (response, token_cnt)

    def log_token_stats(self) -> None:
        total_sum_cnt = TokenCountCached(in_token_cnt=0, out_token_cnt=0)
        for tag in self.token_cnts:
            token_cnt = self.token_cnts[tag]
            if not token_cnt:
                continue
            sum_cnt = sum(
                token_cnt, start=TokenCountCached(in_token_cnt=0, out_token_cnt=0)
            )
            assert isinstance(sum_cnt, TokenCountCached)

            total_sum_cnt += sum_cnt
            sum_equal_cnt = self.equivalent_cost(sum_cnt)

            if sum_cnt.cache_write_cnt or sum_cnt.cache_read_cnt:
                logger.info(f"{tag + ' cnt':<25}: {sum_cnt}")
                logger.info(f"{tag + ' equal cnt':<25}: {sum_equal_cnt}")
            else:
                logger.info(f"{tag + ' cnt':<25}: {sum_equal_cnt}")

        total_sum_equal_cnt = self.equivalent_cost(total_sum_cnt)
        if total_sum_cnt.cache_write_cnt or total_sum_cnt.cache_read_cnt:
            saved_tokens = round(
                total_sum_cnt.cache_write_cnt * (1 - self.write_cost_ratio)
                + total_sum_cnt.cache_read_cnt * (1 - self.read_cost_ratio)
            )
            logger.info(
                f"{'Total cached cnt':<25}: {total_sum_cnt}, saved {saved_tokens:>8} tokens"
            )
            logger.info(f"{'Total equal cnt':<25}: {total_sum_equal_cnt}")
        else:
            logger.info(f"{'Total cnt':<25}: {total_sum_equal_cnt}")
        if self.token_cost:
            total_cost = (
                total_sum_equal_cnt.in_token_cnt
                * self.token_cost.in_token_cost_per_token
                + total_sum_equal_cnt.out_token_cnt
                * self.token_cost.out_token_cost_per_token
            )
            logger.info(f"{'Total cost':<25}: ${total_cost:.2f} USD")

    def get_sum_count_cached(self, tag: str | None = None) -> TokenCount:
        # If have tag: return sum of token counts with that tag
        # If no tag: return sum of all token counts
        if tag:
            token_cnt = self.token_cnts[tag]
            sum_cnt = sum(
                token_cnt, start=TokenCountCached(in_token_cnt=0, out_token_cnt=0)
            )
        else:
            sum_cnt = TokenCountCached(in_token_cnt=0, out_token_cnt=0)
            for token_cnt in self.token_cnts.values():
                sum_cnt += sum(
                    token_cnt, start=TokenCountCached(in_token_cnt=0, out_token_cnt=0)
                )
        assert isinstance(sum_cnt, TokenCount)
        return sum_cnt

    def get_sum_count(self, tag: str | None = None) -> TokenCount:
        sum_cnt_cached = self.get_sum_count_cached(tag)
        sum_cnt = (
            self.equivalent_cost(sum_cnt_cached)
            if isinstance(sum_cnt_cached, TokenCountCached)
            else sum_cnt_cached
        )
        return sum_cnt

    def get_total_token(self) -> int:
        """Return token number regarding to token limit"""
        sum_cnt_cached = self.get_sum_count_cached()
        sum_token_cnt = sum_cnt_cached.in_token_cnt + sum_cnt_cached.out_token_cnt
        if isinstance(sum_cnt_cached, TokenCountCached):
            sum_token_cnt += (
                sum_cnt_cached.cache_write_cnt + sum_cnt_cached.cache_read_cnt
            )
        return sum_token_cnt
