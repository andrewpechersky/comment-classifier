
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Flask
EXPOSE 5000

CMD ["python", "app.py"]



docker-compose up --build