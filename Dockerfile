FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY data ./data
COPY models ./models
COPY evals ./evals
RUN uv sync --frozen --no-dev && useradd --create-home --uid 10001 sentinel && chown -R sentinel:sentinel /app
USER sentinel
ENV PATH="/app/.venv/bin:${PATH}" PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/healthz', timeout=3)" || exit 1
CMD ["python", "-m", "sentinel.demo", "--host", "0.0.0.0"]
