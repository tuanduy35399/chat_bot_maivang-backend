FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY chat/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY chat ./chat
COPY vector_db ./vector_db

CMD ["sh", "-c", "uvicorn chat.google_model.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
