"""Квалификация лида: диалог через LLM, извлечение анкеты и оценка «температуры»."""
import json
import re
from dataclasses import dataclass

from llm import LLM
from school import Course, School

REQUIRED = ("goal", "level", "format", "start", "budget", "name")
LABELS = {"goal": "Цель", "level": "Уровень", "format": "Формат", "start": "Старт", "budget": "Бюджет", "name": "Имя"}

# Вопросы для режима без ИИ (и на случай, если API недоступен)
QUESTIONS = {
    "goal": "Для чего вам английский? Работа, путешествия, экзамен или переезд, для себя или для ребёнка?",
    "level": "Как оцениваете свой уровень: с нуля, базовый, средний или продвинутый? "
             "Если не знаете — не страшно, определим на пробном уроке.",
    "format": "Как удобнее заниматься: в группе или индивидуально с преподавателем?",
    "start": "Когда хотите начать: на этой неделе, в течение месяца или позже?",
    "budget": "На какой бюджет в месяц ориентируетесь?",
    "name": "Как к вам обращаться?",
}

QUALIFY_PROMPT = """Ты — менеджер по продажам онлайн-школы английского «{name}» и общаешься с клиентом в Telegram.

Задача: дружелюбно выяснить потребность клиента, подобрать курс и подвести к бесплатному пробному уроку. На вопросы о школе отвечай строго по базе знаний.

Что нужно выяснить (по одному вопросу за раз, живо, не анкетой):
- goal — зачем нужен английский (работа, путешествия, экзамен или переезд, для себя, для ребёнка и т.п.), коротко;
- level — уровень: A0 (с нуля), A1, A2, B1, B2, C1 или «не знаю» (тогда скажи, что определим на пробном уроке);
- format — «группа» или «индивидуально»;
- start — когда начать: «неделя» (в ближайшие дни), «месяц», «позже» или «не знаю»;
- budget — бюджет в рублях в месяц одним числом (если назван диапазон — верхняя граница; если клиент не хочет говорить — «не знаю»);
- name — как обращаться к клиенту.

Правила:
- Цены, курсы, скидки и условия — только из базы знаний. Никогда не выдумывай.
- Если в последнем сообщении клиента есть вопрос — reply обязательно начинается с ответа на него по базе знаний, и только потом следующий вопрос.
- Если клиент ответил сразу на несколько вопросов — запиши все поля.
- Следующий вопрос — про первое поле из «Ещё не известно», которое клиент не назвал в последнем сообщении. Никогда не переспрашивай то, что клиент уже сказал.
- Спрашивай простыми словами: не перечисляй клиенту коды уровней и названия полей.
- Если ответ клиента непонятен (например, «а», шутка или не по теме) — вежливо задай тот же вопрос ещё раз.
- Когда после ответа клиента известны все поля — коротко порекомендуй один курс из базы знаний и предложи бесплатный пробный урок. Телефон не спрашивай — бот запросит его сам.
- Не обещай звонок менеджера и не говори, что заявка оформлена: это сделает бот, когда клиент оставит телефон.
- Курсы называй их названиями из базы знаний (не id), цены пиши с пробелом: 12 900 ₽.
- Пиши коротко: 1–3 предложения, на «вы», без markdown. Эмодзи — не больше одного и только к месту.

Уже известно: {known}
Ещё не известно: {missing}

Ответь JSON-объектом: {{"reply": "сообщение клиенту", "fields": {{...}}, "course": "id курса или null"}}.
В fields для каждого поля укажи значение, только если клиент сам назвал его в последнем сообщении, иначе null. Никогда не придумывай имя, бюджет и другие данные клиента. course — id рекомендованного курса, когда рекомендуешь курс, иначе null.

База знаний:
{kb}"""

DONE_PROMPT = """Ты — менеджер онлайн-школы английского «{name}» в Telegram. Клиент уже оставил заявку ({known}), менеджер скоро свяжется, чтобы подобрать время пробного урока.

Отвечай на вопросы клиента коротко (1–3 предложения), на «вы», без markdown, строго по базе знаний; эмодзи — не больше одного и только к месту. Чего нет в базе — скажи, что уточнит менеджер. Никогда не выдумывай.

Ответь JSON-объектом: {{"reply": "сообщение клиенту", "fields": {{все поля null}}, "course": null}}

База знаний:
{kb}"""

# Строгая схема ответа модели: API не даст ответить не по формату
REPLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "fields", "course"],
    "properties": {
        "reply": {"type": "string"},
        "fields": {
            "type": "object",
            "additionalProperties": False,
            "required": list(REQUIRED),
            "properties": {k: {"type": ["string", "null"]} for k in REQUIRED},
        },
        "course": {"type": ["string", "null"]},
    },
}

FALLBACK_DONE = "Спасибо за вопрос! Менеджер уточнит это, когда свяжется с вами."
NOT_UNDERSTOOD = "Не совсем понял вас 🙂 "
NAME_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё \-]{1,39}$")
DONT_KNOW_RE = re.compile(r"не\s*зна|не\s*уверен|затрудня|не\s*скаж|пока\s*нет|не\s*важно")


@dataclass
class StepResult:
    reply: str
    fields: dict
    course: str | None
    pending: str | None


@dataclass
class Score:
    points: int
    temperature: str  # hot | warm | cold
    reasons: list[str]


TEMPERATURES = {"hot": "🔥 Горячий", "warm": "🟡 Тёплый", "cold": "❄️ Холодный"}


def rub(n: int) -> str:
    return f"{n:,}".replace(",", " ") + " ₽"


def is_ready(fields: dict) -> bool:
    return all(f in fields for f in REQUIRED)


def parse_budget(value: str) -> int | str:
    nums = [int(re.sub(r"\D", "", n)) for n in re.findall(r"\d+(?:[  ]\d{3})*", value)]
    if not nums:
        return "не знаю"
    n = max(nums)
    if n < 1000 and re.search(r"тыс|\d\s*[кk]\b", value.lower()):
        n *= 1000
    # «ребёнку 10 лет» — не бюджет
    return n if 500 <= n <= 500_000 else "не знаю"


def normalize(raw: dict, strict: bool = False) -> dict:
    """Приводит ответы к единому виду, чтобы по ним можно было считать оценку.

    strict — ответ клиента разбирается без ИИ: нераспознанное не записываем,
    а «не знаю» принимаем, только если клиент так и написал."""
    out = {}
    for key, value in raw.items():
        if key not in REQUIRED or value is None or str(value).strip().lower() in ("", "null", "?", "-", "—"):
            continue
        v = str(value).strip()
        low = v.lower()
        if key == "name":
            if NAME_RE.match(v) and not DONT_KNOW_RE.search(low) and len(v.split()) <= 3:
                out[key] = v.title() if v.islower() else v
            continue
        if key == "goal":
            if len(re.findall(r"[a-zа-яё]", low)) >= 3:
                out[key] = v[:80]
            continue
        if key == "level":
            latin = v.upper().translate(str.maketrans("АВС", "ABC"))
            m = re.search(r"[ABC][0-2]", latin)
            out[key] = (m.group(0) if m else "A0" if "нул" in low else "A2" if "базов" in low
                        else "B1" if "средн" in low else "B2" if "продвин" in low else "не знаю")
        elif key == "format":
            out[key] = "индивидуально" if "инд" in low else "группа" if "груп" in low else "не знаю"
        elif key == "start":
            out[key] = ("неделя" if re.search(r"недел|сейчас|сразу|сегодн|завтра|скорее|срочно", low)
                        else "месяц" if "месяц" in low
                        else "позже" if re.search(r"позж|не скоро|осен|зимой|летом|весной|через полгода", low)
                        else "не знаю")
        elif key == "budget":
            out[key] = parse_budget(v)
        if strict and out.get(key) == "не знаю" and not DONT_KNOW_RE.search(low):
            del out[key]
    return out


def _numbers(text: str) -> list[int]:
    return [int(re.sub(r"\D", "", n)) for n in re.findall(r"\d+(?:[  ]\d{3})*", text)]


def grounded(fields: dict, user_text: str) -> dict:
    """Оставляет имя и бюджет, только если клиент действительно их написал — защита от выдумок модели."""
    out = dict(fields)
    low = user_text.lower()
    name = out.get("name")
    if name and not all(part.lower()[:4] in low for part in str(name).split()):
        del out["name"]
    budget = out.get("budget")
    if isinstance(budget, int):
        nums = _numbers(user_text)
        if budget not in nums and budget not in [n * 1000 for n in nums]:
            del out["budget"]
    return out


def recommend(fields: dict, school: School, choice: str | None) -> Course:
    if choice in school.courses:
        return school.courses[choice]
    goal = str(fields.get("goal", "")).lower()
    if re.search(r"реб[её]н|дет|сын|доч", goal):
        return school.courses["kids"]
    if re.search(r"ielts|toefl|экзам|переезд", goal):
        return school.courses["ielts"]
    if fields.get("format") == "индивидуально":
        return school.courses["individual"]
    return school.courses["group"]


def score_lead(fields: dict, course: Course, phone: str | None) -> Score:
    """Прозрачная оценка: менеджер видит, за что начислены баллы. Максимум — 9."""
    points, reasons = 0, []
    start = fields.get("start")
    if start == "неделя":
        points += 3
        reasons.append("хочет начать на этой неделе (+3)")
    elif start == "месяц":
        points += 2
        reasons.append("хочет начать в течение месяца (+2)")
    else:
        reasons.append("сроки не определены (0)")

    budget = fields.get("budget")
    if isinstance(budget, int):
        if budget >= course.price:
            points += 3
            reasons.append(f"бюджет {rub(budget)} покрывает цену курса {rub(course.price)} (+3)")
        elif budget >= course.price * 0.7:
            points += 1
            reasons.append(f"бюджет {rub(budget)} немного ниже цены {rub(course.price)} (+1)")
        else:
            reasons.append(f"бюджет {rub(budget)} заметно ниже цены {rub(course.price)} (0)")
    else:
        reasons.append("бюджет не назван (0)")

    if phone:
        points += 2
        reasons.append("оставил телефон (+2)")
    else:
        reasons.append("телефон не оставил (0)")

    if fields.get("goal"):
        points += 1
        reasons.append("понятная цель обучения (+1)")

    temperature = "hot" if points >= 7 else "warm" if points >= 4 else "cold"
    return Score(points, temperature, reasons)


def _parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    for candidate in (raw, raw[raw.find("{"): raw.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return None


class Qualifier:
    def __init__(self, llm: LLM, school: School):
        self.llm = llm
        self.school = school
        self.kb = school.knowledge_base()

    def _prompt(self, fields: dict, done: bool) -> str:
        known = json.dumps(fields, ensure_ascii=False) if fields else "ничего"
        missing = ", ".join(f for f in REQUIRED if f not in fields) or "всё известно"
        template = DONE_PROMPT if done else QUALIFY_PROMPT
        return template.format(name=self.school.name, known=known, missing=missing, kb=self.kb)

    async def step(self, history: list[dict], fields: dict, done: bool, pending: str | None) -> StepResult:
        raw = await self.llm.complete(self._prompt(fields, done), history[-14:], schema=REPLY_SCHEMA)
        data = _parse_json(raw)
        if data and isinstance(data.get("reply"), str) and data["reply"].strip():
            new = {}
            if not done:
                # Через перевод строки: иначе «недели 2» + «15000» склеятся в число «2 150»
                user_text = "\n".join(m["content"] for m in history if m["role"] == "user")
                new = grounded(normalize(data.get("fields") or {}), user_text)
                # Страховка: модель иногда не записывает простой ответ на свой же вопрос.
                # Для полей с понятными значениями разбираем последний ответ сами.
                asked = next((f for f in REQUIRED if f not in fields), None)
                last = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
                rescued = {}
                for f in ("level", "format", "start", "budget"):
                    if f not in fields and f not in new and last:
                        rescued.update(grounded(normalize({f: last}, strict=True), user_text))
                new.update(rescued)
                if asked in rescued:
                    # Модель не услышала ответ на свой же вопрос и переспросит — задаём следующий сами
                    following = next((f for f in REQUIRED if f not in {**fields, **new}), None)
                    reply = f"Спасибо! {QUESTIONS[following]}" if following else ""
                    return StepResult(reply, {**fields, **new}, None, None)
            course = data.get("course") if isinstance(data.get("course"), str) else None
            merged = {**fields, **new}
            missing = next((f for f in REQUIRED if f not in merged), None)
            reply = data["reply"].strip()
            if not done and course and missing and "?" not in reply:
                # Модель уже советует курс, но данных не хватает и вопроса нет — иначе диалог пойдёт по кругу
                reply += f"\n\nУточню ещё одно, чтобы записать вас на пробный урок. {QUESTIONS[missing]}"
            return StepResult(reply, merged, course, None)
        return self._scripted(history, fields, done, pending)

    def _scripted(self, history: list[dict], fields: dict, done: bool, pending: str | None) -> StepResult:
        """Сценарий без ИИ: задаём вопросы по порядку, непонятные ответы переспрашиваем."""
        if done:
            return StepResult(FALLBACK_DONE, fields, None, None)
        fields = dict(fields)
        missing = [f for f in REQUIRED if f not in fields]
        # Если до этого вёл ИИ, он спрашивал про первое неизвестное поле
        pending = pending or (missing[0] if missing else None)
        last = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        if pending and last:
            fields.update(normalize({pending: last}, strict=True))
            if pending not in fields:
                return StepResult(NOT_UNDERSTOOD + QUESTIONS[pending], fields, None, pending)
        missing = [f for f in REQUIRED if f not in fields]
        if missing:
            return StepResult(QUESTIONS[missing[0]], fields, None, missing[0])
        return StepResult("", fields, None, None)
