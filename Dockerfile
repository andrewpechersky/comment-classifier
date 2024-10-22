# Використовуємо офіційний образ Python як базовий
FROM python:3.9-slim

# Встановлюємо необхідні системні пакети
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо PyTorch і Transformers
RUN pip install torch transformers flask

# Копіюємо ваші файли до контейнера
COPY main.py /app/main.py
COPY templates/ /app/templates/
COPY model_balanced_aug_with_2_epoch.pth /app/model_balanced_aug_with_2_epoch.pth


# Вказуємо робочу директорію
WORKDIR /app

# Відкриваємо порт для Flask
EXPOSE 5000

# Команда для запуску додатку
CMD ["python", "main.py"]