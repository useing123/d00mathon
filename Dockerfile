FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cqa ./cqa
COPY checklists ./checklists
COPY examples ./examples
EXPOSE 8090
# secrets come from the environment / --env-file .env, never baked into the image
CMD ["python", "-m", "cqa", "serve", "--host", "0.0.0.0", "--port", "8090"]
