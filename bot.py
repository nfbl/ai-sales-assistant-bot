"""ИИ-менеджер по продажам онлайн-школы: квалифицирует заявку в диалоге,
подбирает курс, передаёт лид в Битрикс24 и менеджеру, напоминает о себе."""
import asyncio
import html
import logging
import re
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand, CallbackQuery, KeyboardButton, Message, ReplyKeyboardMarkup, User,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import load_settings
from crm import Bitrix, lead_fields, manager_card
from db import Dialog, Store
from llm import LLM
from qualifier import TEMPERATURES, Qualifier, is_ready, recommend, rub, score_lead
from school import School

log = logging.getLogger(__name__)

settings = load_settings()
school = School.load()
store = Store(settings.db_path)
qualifier = Qualifier(LLM(settings), school)
crm = Bitrix(settings.bitrix_webhook)
router = Router()

BTN_COURSES = "📚 Курсы и цены"
BTN_RESTART = "🔄 Начать заново"
BTN_PHONE = "📱 Отправить номер"
BTN_SKIP = "Не сейчас"

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=BTN_COURSES), KeyboardButton(text=BTN_RESTART)]],
    resize_keyboard=True, input_field_placeholder="Напишите сообщение",
)
PHONE_KB = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=BTN_PHONE, request_contact=True)], [KeyboardButton(text=BTN_SKIP)]],
    resize_keyboard=True, one_time_keyboard=True,
)
GOALS = {
    "work": ("💼 Для работы", "Для работы"),
    "travel": ("✈️ Для путешествий", "Для путешествий"),
    "exam": ("🎓 Экзамен или переезд", "Для экзамена или переезда"),
    "kids": ("👧 Для ребёнка", "Для ребёнка"),
}
FIRST_QUESTION = "Для чего вам английский?"
CONTACT_TEXT = ("Чтобы записать вас на бесплатный пробный урок, оставьте номер телефона — "
                "менеджер перезвонит и подберёт удобное время 👇")
NUDGE_QUALIFY = ("Кажется, мы не договорили 🙂 Ответьте на последний вопрос — и я подберу вам курс. "
                 "Первый урок у нас бесплатный.")
NUDGE_CONTACT = ("Напомню про бесплатный пробный урок 🙂 Оставьте номер — менеджер подберёт удобное время. "
                 "Или нажмите «Не сейчас».")


def now() -> datetime:
    return datetime.now(settings.tz)


def dialog_for(user: User) -> Dialog:
    return store.get(user.id) or store.new(user.id, user.username, user.first_name or "", now())


def goals_kb():
    kb = InlineKeyboardBuilder()
    for key, (label, _) in GOALS.items():
        kb.button(text=label, callback_data=f"goal:{key}")
    kb.adjust(2)
    return kb.as_markup()


def is_phone(text: str) -> bool:
    return 10 <= len(re.sub(r"\D", "", text)) <= 12 and not re.search(r"[a-zа-я]", text.lower())


async def start_dialog(bot: Bot, user: User, greet: bool) -> None:
    d = store.new(user.id, user.username, user.first_name or "", now())
    if greet:
        await bot.send_message(
            user.id,
            f"Здравствуйте, {html.escape(user.first_name or '')}! 👋\n\n"
            f"Я помощник онлайн-школы английского «{school.name}». Помогу подобрать курс и записаться "
            "на бесплатный пробный урок — задам пару вопросов, это займёт минуту.",
            reply_markup=MAIN_KB,
        )
    await bot.send_message(user.id, FIRST_QUESTION, reply_markup=goals_kb())
    d.say("assistant", FIRST_QUESTION)
    d.pending = "goal"
    store.save(d)


# ---------- Команды и меню ----------

@router.message(CommandStart())
@router.message(F.text == BTN_RESTART)
async def cmd_start(message: Message, bot: Bot):
    await start_dialog(bot, message.from_user, greet=True)


@router.message(F.text == BTN_COURSES)
async def show_courses(message: Message):
    kb = InlineKeyboardBuilder()
    kb.button(text="🎯 Подобрать курс", callback_data="pick")
    await message.answer(school.price_list() + "\n\nПервый урок — бесплатно.", reply_markup=kb.as_markup())


@router.callback_query(F.data == "pick")
async def pick_course(cb: CallbackQuery, bot: Bot):
    await cb.answer()
    await start_dialog(bot, cb.from_user, greet=False)


@router.callback_query(F.data.startswith("goal:"))
async def pick_goal(cb: CallbackQuery, bot: Bot):
    label, text = GOALS.get(cb.data.split(":")[1], (None, None))
    await cb.answer()
    if not text:
        return
    await cb.message.edit_text(f"{FIRST_QUESTION}\n→ {label}")
    await process(bot, cb.from_user, text)


@router.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"ID этого чата: <code>{message.chat.id}</code>\n"
                         "Укажите его в .env как ADMIN_CHAT_ID, чтобы получать лиды.")


@router.message(Command("leads"))
async def cmd_leads(message: Message):
    if message.chat.id != settings.admin_chat_id:
        await message.answer("Команда доступна только менеджеру.")
        return
    leads = store.recent_leads()
    if not leads:
        await message.answer("Лидов пока нет.")
        return
    lines = ["<b>Последние лиды:</b>"]
    for lead in leads:
        course = school.courses.get(lead.course_id)
        crm_link = f' · <a href="{crm.lead_url(lead.crm_id)}">Б24 #{lead.crm_id}</a>' if lead.crm_id else ""
        lines.append(f"{TEMPERATURES[lead.temperature].split()[0]} {lead.created_at:%d.%m %H:%M} "
                     f"{html.escape(lead.name or '—')} · {course.title if course else lead.course_id} · "
                     f"{html.escape(lead.phone or 'без телефона')}{crm_link}")
    await message.answer("\n".join(lines), disable_web_page_preview=True)


# ---------- Контакт ----------

@router.message(F.contact)
async def got_contact(message: Message, bot: Bot):
    d = dialog_for(message.from_user)
    if d.stage == "contact":
        await finish(bot, d, message.contact.phone_number)


@router.message(F.text == BTN_SKIP)
async def skip_contact(message: Message, bot: Bot):
    d = dialog_for(message.from_user)
    if d.stage == "contact":
        await finish(bot, d, None)


# ---------- Диалог ----------

@router.message(F.text)
async def on_text(message: Message, bot: Bot):
    await process(bot, message.from_user, message.text)


async def process(bot: Bot, user: User, text: str) -> None:
    d = dialog_for(user)
    if d.stage == "contact" and is_phone(text):
        await finish(bot, d, text.strip())
        return

    await bot.send_chat_action(user.id, ChatAction.TYPING)
    d.say("user", text[:1000])
    res = await qualifier.step(d.history, d.fields, done=d.stage != "qualifying", pending=d.pending)
    d.fields, d.pending = res.fields, res.pending
    if res.course in school.courses:
        d.course_id = res.course

    if d.stage == "qualifying" and is_ready(d.fields):
        course = recommend(d.fields, school, d.course_id)
        d.course_id = course.id
        reply = res.reply or (f"Спасибо! Вам подойдёт «{course.title}» — {rub(course.price)} в месяц. "
                              "Первый урок бесплатный: 30 минут с преподавателем и тест уровня.")
        d.stage = "contact"
        await bot.send_message(user.id, reply, parse_mode=None)
        await bot.send_message(user.id, CONTACT_TEXT, reply_markup=PHONE_KB)
        d.say("assistant", reply)
        d.say("assistant", CONTACT_TEXT)
    else:
        await bot.send_message(user.id, res.reply, parse_mode=None)
        d.say("assistant", res.reply)
        if d.stage == "contact":
            await bot.send_message(user.id, "Оставите номер для пробного урока? 👇", reply_markup=PHONE_KB)

    d.updated_at, d.nudged = now(), False
    store.save(d)


async def finish(bot: Bot, d: Dialog, phone: str | None) -> None:
    course = school.courses.get(d.course_id) or recommend(d.fields, school, None)
    score = score_lead(d.fields, course, phone)

    crm_id, crm_note = None, "⚠️ Битрикс24 не подключён — лид сохранён только в боте."
    if crm.enabled:
        try:
            crm_id = await crm.add_lead(lead_fields(
                answers=d.fields, course=course, score=score, phone=phone,
                username=d.username, history=d.history,
            ))
            crm_note = f'🔗 <a href="{crm.lead_url(crm_id)}">Открыть лид в Битрикс24</a>'
        except Exception as e:
            log.exception("Не удалось создать лид в Битрикс24")
            crm_note = f"⚠️ Ошибка Битрикс24: {html.escape(str(e))[:200]}"

    store.add_lead(user_id=d.user_id, name=d.fields.get("name"), phone=phone, course_id=course.id,
                   points=score.points, temperature=score.temperature, crm_id=crm_id, now=now())
    d.stage, d.updated_at = "done", now()
    name = html.escape(str(d.fields.get("name") or d.first_name))
    where = "по телефону" if phone else "здесь, в Telegram,"
    thanks = (f"Спасибо, {name}! 🙌 Менеджер свяжется с вами {where} и подберёт время пробного урока.\n"
              f"{school.manager_hours}")
    await bot.send_message(d.user_id, thanks, reply_markup=MAIN_KB)
    d.say("assistant", thanks)
    store.save(d)

    if settings.admin_chat_id:
        card = manager_card(answers=d.fields, course=course, score=score, phone=phone,
                            username=d.username, first_name=d.first_name)
        try:
            await bot.send_message(settings.admin_chat_id, f"{card}\n\n{crm_note}",
                                   disable_web_page_preview=True)
        except Exception:
            log.exception("Не удалось отправить лид менеджеру")


# ---------- Напоминания замолчавшим клиентам ----------

async def send_nudges(bot: Bot) -> None:
    for d in store.stale(now() - timedelta(minutes=settings.nudge_minutes)):
        contact = d.stage == "contact"
        text = NUDGE_CONTACT if contact else NUDGE_QUALIFY
        await bot.send_message(d.user_id, text, reply_markup=PHONE_KB if contact else None)
        d.say("assistant", text)
        d.nudged = True
        store.save(d)


async def nudge_loop(bot: Bot) -> None:
    while True:
        try:
            await send_nudges(bot)
        except Exception:
            log.exception("Ошибка при отправке напоминаний")
        await asyncio.sleep(60)


@router.callback_query()
async def stale_button(cb: CallbackQuery):
    await cb.answer("Эта кнопка устарела — нажмите «Начать заново».", show_alert=True)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    await bot.set_my_commands([BotCommand(command="start", description="Подобрать курс")])
    log.info("Бот запущен. ИИ: %s, Битрикс24: %s", settings.llm_provider,
             "подключён" if crm.enabled else "не подключён")
    asyncio.create_task(nudge_loop(bot))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
