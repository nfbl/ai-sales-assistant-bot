"""Данные школы: курсы, цены и база знаний для ИИ."""
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Course:
    id: str
    title: str
    price: int  # ₽ в месяц
    description: str


class School:
    def __init__(self, raw: dict, faq: str):
        self.name: str = raw["name"]
        self.about: str = raw["about"]
        self.manager_hours: str = raw["manager_hours"]
        self.courses = {c["id"]: Course(**c) for c in raw["courses"]}
        self.faq = faq

    @classmethod
    def load(cls) -> "School":
        raw = json.loads((DATA_DIR / "school.json").read_text("utf-8"))
        return cls(raw, (DATA_DIR / "faq.md").read_text("utf-8"))

    def price_list(self) -> str:
        return "\n\n".join(
            f"<b>{c.title}</b> — {c.price:,} ₽/мес".replace(",", " ") + f"\n{c.description}"
            for c in self.courses.values()
        )

    def knowledge_base(self) -> str:
        courses = "\n".join(
            f"- id={c.id}: {c.title}, {c.price} руб. в месяц. {c.description}" for c in self.courses.values()
        )
        return (f"Школа: {self.name}. {self.about}\n{self.manager_hours}\n\n"
                f"Курсы:\n{courses}\n\n{self.faq}")
