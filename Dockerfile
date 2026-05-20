FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY streaminfer/ streaminfer/

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --retries=20 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python", "-m", "streaminfer.server"]
