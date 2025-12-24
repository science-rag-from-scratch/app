# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем UV
RUN pip install uv

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости через UV
RUN uv sync --frozen

# Копируем код приложения
COPY app/ ./app/

# Создаем директорию для данных
RUN mkdir -p /app/data

# Устанавливаем переменные окружения по умолчанию
ENV PYTHONPATH=/app

# Открываем порт для chainlit (по умолчанию 8000)
EXPOSE 8000

# Команда запуска приложения через chainlit
CMD ["uv", "run", "chainlit", "run", "app/app.py", "--host", "0.0.0.0", "--port", "8000"]

