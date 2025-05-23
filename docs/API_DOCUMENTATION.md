# API документация Lucky Train AI

## Введение

Данный документ описывает API для взаимодействия с системой Lucky Train AI. API позволяет интегрировать функционал AI-ассистента в различные приложения и сервисы.

## Аутентификация

Все API-запросы требуют аутентификации с использованием API-ключа. API-ключ должен быть передан в HTTP-заголовке `X-API-Key`.

```http
X-API-Key: ваш_api_ключ
```

## Базовый URL

```text
https://api.luckytrain.io/v1
```

## Конечные точки

### Чат API

#### POST /api/chat

Отправляет сообщение AI-ассистенту и получает ответ.

**Запрос:**

```json
{
  "message": "Расскажи о проекте Lucky Train",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id",
  "language": "ru"
}
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| message | string | Да | Сообщение пользователя |
| session_id | string | Нет | ID сессии для сохранения контекста разговора |
| user_id | string | Нет | ID пользователя для аналитики |
| language | string | Нет | Предпочтительный язык ответа (ru, en, es, и т.д.) |

**Ответ:**

```json
{
  "session_id": "session-uuid",
  "message_id": "message-uuid",
  "response": "Lucky Train - это инновационный проект на блокчейне TON, создающий собственную метавселенную и экосистему с уникальной экономической моделью.",
  "feedback": {
    "positive": "feedback-url-positive",
    "negative": "feedback-url-negative"
  },
  "response_time": 0.531
}
```

#### POST /api/stream-chat

Потоковая версия чат-API, возвращающая ответ по частям в реальном времени.

**Запрос:**
Аналогичен `/api/chat`

**Ответ:**
Стрим данных в формате Server-Sent Events (SSE):

```text
data: {"content": "Lucky Train - это "}
data: {"content": "инновационный проект "}
data: {"content": "на блокчейне TON, "}
...
data: {"content": "", "end": true, "message_id": "message-uuid", "feedback": {...}}
```

### Обратная связь

#### POST /api/feedback

Отправляет обратную связь по ответу AI-ассистента.

**Запрос:**

```json
{
  "message_id": "message-uuid",
  "rating": 1,
  "comments": "Очень полезный ответ"
}
```

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| message_id | string | Да | ID сообщения, полученный в ответе `/api/chat` |
| rating | number | Да | Оценка от 1 (положительно) до -1 (отрицательно) |
| comments | string | Нет | Дополнительные комментарии |

**Ответ:**

```json
{
  "success": true
}
```

### Блокчейн интеграция

#### GET /api/blockchain/info

Получает общую информацию о блокчейне TON.

**Ответ:**

```json
{
  "name": "TON",
  "active_validators": 345,
  "current_block": 20123456,
  "tps": 10000
}
```

#### GET /api/blockchain/token-info

Получает информацию о токене LTT.

**Параметры запроса:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| token | string | Нет | Символ токена (по умолчанию "LTT") |

**Ответ:**

```json
{
  "symbol": "LTT",
  "name": "Lucky Train Token",
  "price_usd": 0.15,
  "price_ton": 0.02,
  "total_supply": 1000000000
}
```

#### POST /api/blockchain/wallet-auth

Генерирует сообщение для аутентификации TON-кошелька.

**Запрос:**

```json
{
  "user_id": "user-id"
}
```

**Ответ:**

```json
{
  "message": "Lucky Train Authentication\nUser: user-id\nTimestamp: 1634567890",
  "timestamp": 1634567890
}
```

### Метавселенная

#### GET /api/metaverse/locations

Получает список доступных локаций в метавселенной.

**Ответ:**

```json
{
  "locations": [
    {
      "id": "central_station",
      "name": "Центральный вокзал",
      "description": "Главный хаб метавселенной, где пересекаются все маршруты поездов."
    },
    {
      "id": "trading_district",
      "name": "Торговый квартал",
      "description": "Место для покупки, продажи и обмена виртуальными товарами и NFT."
    }
  ]
}
```

#### POST /api/metaverse/location-preview

Генерирует 3D-превью локации.

**Запрос:**

```json
{
  "location_id": "central_station"
}
```

**Ответ:**

```json
{
  "success": true,
  "location_id": "central_station",
  "location_name": "Центральный вокзал",
  "image_base64": "base64-encoded-image-data",
  "preview_url": "https://metaverse.luckytrain.io/previews/central_station.png"
}
```

## Коды ошибок

| Код | Описание |
|-----|----------|
| 400 | Некорректный запрос |
| 401 | Ошибка аутентификации |
| 403 | Доступ запрещен |
| 404 | Ресурс не найден |
| 429 | Превышен лимит запросов |
| 500 | Внутренняя ошибка сервера |

## Лимиты запросов

API имеет следующие лимиты запросов:

- 60 запросов в минуту для стандартных клиентов
- 300 запросов в минуту для премиум-клиентов

При превышении лимита API вернет код ошибки 429. 
 