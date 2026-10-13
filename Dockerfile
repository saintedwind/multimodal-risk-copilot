FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
RUN useradd --create-home desk
USER desk
EXPOSE 8000 8501
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
