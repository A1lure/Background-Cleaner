FROM python:3.13-slim

WORKDIR /app

COPY requirements/requirements-base.txt .

RUN pip install --no-cache-dir -r requirements-base.txt

RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]