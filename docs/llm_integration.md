# LLM Integration with OpenAI

Данный документ описывает интеграцию Lucky Train AI Assistant с OpenAI для улучшения качества ответов с использованием Retrieval Augmented Generation (RAG).

## Обзор

Lucky Train AI Assistant теперь использует языковые модели OpenAI (например, GPT-3.5-Turbo или GPT-4) для генерации высококачественных ответов, основанных на информации из базы знаний. Это значительно улучшает способность ассистента понимать и отвечать на сложные вопросы.

Реализация построена на принципе Retrieval Augmented Generation (RAG):
1. Запрос пользователя анализируется
2. Релевантная информация извлекается из базы знаний
3. Эта информация используется как контекст для языковой модели
4. Модель генерирует качественный ответ, основанный на контексте и запросе

## Настройка

### Требования

Для использования интеграции с OpenAI вам потребуется:
1. API ключ OpenAI (получите его на [платформе OpenAI](https://platform.openai.com/))
2. Python 3.8 или выше
3. Установленные зависимости из `requirements.txt`

### Конфигурация

1. Скопируйте файл `env.example` в `.env`:
   ```bash
   cp env.example .env
   ```

2. Отредактируйте файл `.env` и добавьте ваш API ключ OpenAI:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

3. Настройте параметры LLM в файле конфигурации `config/config.json`:
   ```json
   {
     "llm_model": "gpt-3.5-turbo",
     "temperature": 0.7,
     "max_tokens": 500
   }
   ```

## Параметры настройки

В файле `config/config.json` можно настроить следующие параметры:

- `llm_model`: Модель OpenAI для использования (например, "gpt-3.5-turbo", "gpt-4")
- `temperature`: Температура генерации (влияет на креативность и разнообразие ответов)
- `max_tokens`: Максимальное количество токенов в ответе

## Отказоустойчивость

Система спроектирована с учетом отказоустойчивости:

1. Если API ключ OpenAI не указан, система автоматически переключается на поиск TF-IDF и алгоритм генерации ответов на основе шаблонов.
2. Если API OpenAI недоступен или возвращает ошибку, система также автоматически переключается на резервный механизм.

## Примеры использования

### Запуск ассистента с интеграцией OpenAI

```bash
# Запуск в консольном режиме
python src/main.py console

# Запуск Telegram бота
python src/main.py telegram

# Запуск всех компонентов
python src/main.py all
```

## Мониторинг и логи

Все взаимодействия с API OpenAI логируются и доступны в файле логов `logs/lucky_train_ai.log`.

## Дальнейшие улучшения

В будущем планируется:

1. Использование API embeddings для семантического поиска вместо простого поиска по ключевым словам
2. Кэширование запросов к API для уменьшения расходов
3. Тонкая настройка промптов для различных типов запросов
4. Интеграция с более легковесными локальными LLM для снижения затрат

## Решение проблем

### Не удается подключиться к API OpenAI

1. Проверьте правильность API ключа в файле `.env`
2. Убедитесь, что у вас есть доступ к API OpenAI (проверьте баланс аккаунта)
3. Проверьте доступность интернет-соединения

### Ответы слишком короткие или неинформативные

1. Увеличьте параметр `max_tokens` в конфигурации
2. Уменьшите `temperature` для более сфокусированных ответов
3. Проверьте базу знаний на полноту информации по запрашиваемым темам

### Система не использует OpenAI API

1. Проверьте логи на наличие ошибок
2. Убедитесь, что API ключ указан и действителен
3. Перезапустите приложение после внесения изменений

## Ограничения

1. API OpenAI имеет ограничения на количество запросов в минуту и требует оплаты
2. Генерация ответов может иногда включать неточную информацию
3. Длина контекста ограничена моделью (например, 4096 токенов для gpt-3.5-turbo) 