#!/usr/bin/env bash
# Установка или обновление бота на чистом Ubuntu/Debian (запускать от root).
# Перед первым запуском положите .env в /opt/leads-bot/.env
set -euo pipefail

REPO=https://github.com/nfbl/ai-sales-assistant-bot.git
DIR=/opt/leads-bot

# apt на Ubuntu 24.04 (needrestart) читает stdin и съедает остаток скрипта при `bash -s < setup.sh`
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq </dev/null
apt-get install -y -qq git python3 python3-venv >/dev/null </dev/null

id bot >/dev/null 2>&1 || useradd --system --home "$DIR" --shell /usr/sbin/nologin bot

if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only
else
    mkdir -p "$DIR"
    # .env мог быть скопирован заранее — клонируем рядом и переносим
    git clone -q "$REPO" /tmp/leads-bot-src
    cp -a /tmp/leads-bot-src/. "$DIR"/
    rm -rf /tmp/leads-bot-src
fi

python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"

chown -R bot:bot "$DIR"
chmod 600 "$DIR/.env" 2>/dev/null || echo "ВНИМАНИЕ: нет $DIR/.env — бот не запустится"

cp "$DIR/deploy/leads-bot.service" /etc/systemd/system/leads-bot.service
systemctl daemon-reload
systemctl enable -q leads-bot
systemctl restart leads-bot
sleep 3
systemctl --no-pager --lines=5 status leads-bot
