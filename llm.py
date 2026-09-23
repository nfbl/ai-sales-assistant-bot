"""Обёртка над LLM-провайдерами (LLM_PROVIDER в .env):
  openai    — OpenAI и совместимые API (Groq, OpenRouter, Gemini...)
  anthropic — Claude
  gigachat  — GigaChat
  none      — без ИИ, бот работает по сценарию
"""
import logging

from config import Settings

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gigachat": "GigaChat",
}


class LLM:
    def __init__(self, s: Settings):
        self.provider = s.llm_provider
        self.model = s.llm_model or DEFAULT_MODELS.get(self.provider, "")
        self.extra_body = {"reasoning_effort": s.llm_reasoning_effort} if s.llm_reasoning_effort else None
        self.client = self._make_client(s)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def _make_client(self, s: Settings):
        if self.provider == "none":
            return None
        if not s.llm_api_key:
            raise SystemExit(f"Для LLM_PROVIDER={self.provider} нужен LLM_API_KEY в .env")
        if self.provider == "openai":
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            return AsyncAnthropic(api_key=s.llm_api_key)
        if self.provider == "gigachat":
            from gigachat import GigaChat
            return GigaChat(credentials=s.llm_api_key, model=self.model, verify_ssl_certs=False)
        raise SystemExit(f"Неизвестный LLM_PROVIDER: {self.provider}")

    async def complete(self, system: str, history: list[dict], schema: dict | None = None) -> str | None:
        """Ответ модели или None, если ИИ выключен или API недоступен.

        schema — JSON-схема ответа: для OpenAI-совместимых API включает строгий режим,
        в котором модель не может ответить не по формату."""
        messages = list(history)
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        if not self.enabled or not messages:
            return None
        try:
            if self.provider == "openai":
                return await self._openai(system, messages, schema)
            if self.provider == "anthropic":
                resp = await self.client.messages.create(
                    model=self.model, system=system, messages=messages, temperature=0.3, max_tokens=1000,
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            if self.provider == "gigachat":
                from gigachat.models import Chat, Messages, MessagesRole
                roles = {"user": MessagesRole.USER, "assistant": MessagesRole.ASSISTANT}
                chat = Chat(
                    temperature=0.3, max_tokens=1000,
                    messages=[Messages(role=MessagesRole.SYSTEM, content=system)]
                    + [Messages(role=roles[m["role"]], content=m["content"]) for m in messages],
                )
                resp = await self.client.achat(chat)
                return resp.choices[0].message.content
        except Exception:
            log.exception("Ошибка LLM (%s)", self.provider)
        return None

    async def _openai(self, system: str, messages: list[dict], schema: dict | None) -> str | None:
        from openai import BadRequestError

        formats = [None]
        if schema:
            formats = [{"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}},
                       {"type": "json_object"}]  # запасной вариант для API без строгих схем
        for attempt, fmt in enumerate(formats + formats[-1:]):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, temperature=0.3, max_tokens=1500,
                    messages=[{"role": "system", "content": system}, *messages],
                    extra_body=self.extra_body, **({"response_format": fmt} if fmt else {}),
                )
                return resp.choices[0].message.content
            except BadRequestError as e:
                log.warning("LLM отклонил запрос (попытка %d): %s", attempt + 1, e)
        return None
