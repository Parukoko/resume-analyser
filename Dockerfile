FROM python:3.11-slim AS base
WORKDIR /app

# ---- app ----
FROM base AS app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- ui ----
FROM base AS ui

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# ---- test ----
FROM base AS test

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml .
COPY app ./app
COPY tests ./tests

CMD ["pytest", "-v"]
