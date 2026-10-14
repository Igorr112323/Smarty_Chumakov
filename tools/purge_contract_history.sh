#!/usr/bin/env bash
# Очистка истории репозитория от PDF подписанного договора (проект 117812).
#
# Зачем: файл «Договор_0117812 (25).pdf» (5 019 351 байт, SHA-256 e5eb6ad8…)
# содержит персональные данные (паспорт, ИНН, адрес, счёт). Из рабочего дерева он
# удалён, но остался в истории: коммит 64c8930 добавил файл, f9fa026 удалил, а
# сам объект (git blob a52928f529b1496b68b0ce8e38477b6486bfd6af) остаётся
# достижимым по любому старому коммиту, пока история не переписана.
#
# Скрипт НЕ коммитит и НЕ пушит ничего сам до шага 5: сначала переписывает
# историю в отдельной копии и печатает доказательства. Пуш — отдельный шаг,
# который требует явного подтверждения (флаг --push), потому что это
# необратимая операция для всех, кто уже склонировал репозиторий.
#
# Запуск:
#   tools/purge_contract_history.sh                # перепись в копии + доказательства
#   tools/purge_contract_history.sh --push         # ещё и force-push всех веток и тегов
#
# Переменные: REMOTE (URL), WORK (каталог копии), BLOB_PATH (путь к файлу).

set -euo pipefail

REMOTE="${REMOTE:-https://github.com/Igorr112323/Smarty_Chumakov.git}"
BLOB_PATH="${BLOB_PATH:-Договор_0117812 (25).pdf}"
WORK="${WORK:-$(mktemp -d -t purge-contract-XXXXXX)}"
PUSH=0
for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    *) echo "неизвестный аргумент: $arg" >&2; exit 2 ;;
  esac
done

echo "== 0. Проверка инструментов =="
command -v git >/dev/null || { echo "нужен git" >&2; exit 2; }
if ! command -v git-filter-repo >/dev/null; then
  echo "git-filter-repo не найден; ставим из PyPI (одноразовая зависимость)"
  if python3 -m pip install --quiet --user git-filter-repo 2>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    # PEP 668 («externally managed environment») запрещает --user-установку на
    # части систем: ставим в отдельное окружение и берём команду оттуда.
    TOOLS="${TOOLS:-$HOME/.cache/spanverify-tools}"
    python3 -m venv "$TOOLS" >/dev/null 2>&1 || {
      echo "не удалось создать venv для git-filter-repo; поставьте вручную" >&2; exit 2; }
    "$TOOLS/bin/pip" install --quiet git-filter-repo || {
      echo "не удалось поставить git-filter-repo: pip install git-filter-repo" >&2; exit 2; }
    export PATH="$TOOLS/bin:$PATH"
  fi
fi
command -v git-filter-repo >/dev/null || { echo "git-filter-repo недоступен в PATH" >&2; exit 2; }

echo "== 1. Свежая зеркальная копия: $WORK =="
git clone --quiet --mirror "$REMOTE" "$WORK"
cd "$WORK"

echo "== 2. Доказательство «до» =="
BEFORE="$(git log --all --oneline --format=%h -- "$BLOB_PATH" | tr '\n' ' ')"
BLOB_SHA="$(git rev-list --all --objects | awk -v p="$BLOB_PATH" 'index($0, p) {print $1}' | head -1)"
echo "коммиты с файлом: ${BEFORE:-нет}"
echo "blob: ${BLOB_SHA:-нет}"
if [ -z "$BEFORE" ]; then
  echo "файла в истории уже нет — переписывать нечего"
fi

echo "== 3. Перепись истории (файл удаляется из всех коммитов) =="
git filter-repo --invert-paths --path "$BLOB_PATH" --force

echo "== 4. Доказательство «после» =="
AFTER="$(git log --all --oneline --format=%h -- "$BLOB_PATH" | tr '\n' ' ')"
echo "коммиты с файлом: ${AFTER:-нет}"
echo "файлов с договором среди объектов: $(git rev-list --all --objects | grep -ci 'Договор' || true)"
if [ -n "$BLOB_SHA" ]; then
  if git cat-file -e "$BLOB_SHA" 2>/dev/null; then
    echo "ВНИМАНИЕ: blob $BLOB_SHA всё ещё доступен в копии"
  else
    echo "blob $BLOB_SHA недоступен — объект удалён из истории"
  fi
fi
if [ -n "$AFTER" ]; then
  echo "ОСТАНОВ: файл всё ещё в истории, пуш не выполняется" >&2
  exit 1
fi

echo "== 5. Пуш =="
if [ "$PUSH" -ne 1 ]; then
  echo "Пуш не выполнялся (нет флага --push). Команды для владельца:"
  echo "  cd $WORK"
  echo "  git remote add origin $REMOTE   # filter-repo удаляет origin — вернуть обязательно"
  echo "  git push --force --all && git push --force --tags"
  echo "Копия оставлена по пути: $WORK"
  exit 0
fi
# git filter-repo удаляет remote origin, чтобы переписанная история не ушла в
# сеть случайно. Без явного возврата remote команда ниже падает с «No
# configured push destination», и владелец решает, что очистка не сработала.
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push --force --all
git push --force --tags

echo "== 6. Проверка удалённого репозитория =="
echo "После пуша проверьте:"
echo "  gh api repos/Igorr112323/Smarty_Chumakov/contents/%D0%94%D0%BE%D0%B3%D0%BE%D0%B2%D0%BE%D1%80_0117812%20(25).pdf?ref=64c8930"
echo "  (ожидается 404: Not Found)"
echo "Пока GitHub не провёл сборку мусора, объект может оставаться доступным по"
echo "точному SHA. Для гарантированного удаления напишите в GitHub Support"
echo "запрос на удаление недостижимого объекта $BLOB_SHA."
