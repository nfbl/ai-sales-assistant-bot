"""Битрикс24: контакт + сделка через входящий вебхук (права: CRM).

Сделки, а не лиды: в новых порталах по умолчанию включён режим CRM без лидов."""
import html
import re

import aiohttp

from qualifier import LABELS, TEMPERATURES, Score, rub
from school import Course

TITLES = {"hot": "Горячий", "warm": "Тёплый", "cold": "Холодный"}


class BitrixError(Exception):
    pass


class Bitrix:
    def __init__(self, webhook: str | None):
        self.webhook = webhook

    @property
    def enabled(self) -> bool:
        return bool(self.webhook)

    def deal_url(self, deal_id: int) -> str:
        portal = re.match(r"https?://[^/]+", self.webhook).group(0)
        return f"{portal}/crm/deal/details/{deal_id}/"

    async def call(self, method: str, payload: dict) -> dict:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.webhook}{method}.json", json=payload) as resp:
                data = await resp.json(content_type=None)
        if "error" in data:
            raise BitrixError(f"{data['error']}: {data.get('error_description', '')}")
        return data

    async def add_deal(self, contact: dict, deal: dict) -> int:
        """Создаёт контакт и привязанную к нему сделку, возвращает id сделки."""
        contact_id = int((await self.call("crm.contact.add", {"fields": contact}))["result"])
        data = await self.call("crm.deal.add", {"fields": {**deal, "CONTACT_ID": contact_id},
                                                "params": {"REGISTER_SONET_EVENT": "Y"}})
        return int(data["result"])


def deal_fields(*, answers: dict, course: Course, score: Score, phone: str | None,
                username: str | None, history: list[dict]) -> tuple[dict, dict]:
    """Поля контакта и сделки для Битрикс24."""
    esc = lambda s: html.escape(str(s))
    budget = answers.get("budget")
    anketa = [
        f"{LABELS[k]}: {esc(rub(budget) if k == 'budget' and isinstance(budget, int) else answers.get(k, '—'))}"
        for k in ("goal", "level", "format", "start", "budget")
    ]
    dialog = [f"{'Клиент' if m['role'] == 'user' else 'Бот'}: {esc(m['content'])}" for m in history[-12:]]
    comments = "<br>".join(
        [f"<b>Оценка: {TITLES[score.temperature]} ({score.points}/9)</b>"]
        + [f"• {esc(r)}" for r in score.reasons]
        + ["", "<b>Анкета</b>"] + anketa
        + [f"Рекомендованный курс: {esc(course.title)} — {rub(course.price)}/мес"]
        + ["", "<b>Переписка</b>"] + dialog
    )
    contact = {"NAME": answers.get("name", ""), "SOURCE_ID": "OTHER", "SOURCE_DESCRIPTION": "Telegram-бот"}
    if phone:
        contact["PHONE"] = [{"VALUE": phone, "VALUE_TYPE": "MOBILE"}]
    if username:
        contact["IM"] = [{"VALUE": username, "VALUE_TYPE": "TELEGRAM"}]
    deal = {
        "TITLE": f"{TITLES[score.temperature]} лид — {course.title} ({answers.get('name', '')})",
        "STAGE_ID": "NEW",
        "SOURCE_ID": "OTHER",
        "SOURCE_DESCRIPTION": "Telegram-бот",
        "OPPORTUNITY": course.price,
        "CURRENCY_ID": "RUB",
        "COMMENTS": comments,
    }
    return contact, deal


def manager_card(*, answers: dict, course: Course, score: Score, phone: str | None,
                 username: str | None, first_name: str) -> str:
    esc = lambda s: html.escape(str(s))
    budget = answers.get("budget")
    who = esc(answers.get("name") or first_name) + (f" (@{esc(username)})" if username else "")
    lines = [
        f"<b>{TEMPERATURES[score.temperature]} лид · {score.points}/9</b>",
        f"Клиент: {who}",
        f"Телефон: {esc(phone) if phone else 'не оставил — написать в Telegram'}",
        f"Курс: {esc(course.title)} — {rub(course.price)}/мес",
        "",
        f"Цель: {esc(answers.get('goal', '—'))}",
        f"Уровень: {esc(answers.get('level', '—'))} · Формат: {esc(answers.get('format', '—'))}",
        f"Старт: {esc(answers.get('start', '—'))} · Бюджет: "
        f"{rub(budget) if isinstance(budget, int) else esc(budget or '—')}",
        "",
        "<b>Почему такая оценка:</b>",
        *[f"• {esc(r)}" for r in score.reasons],
    ]
    return "\n".join(lines)
