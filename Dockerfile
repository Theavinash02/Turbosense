# Prediction API with the FD004 models. Build: docker build -t turbofan-rul .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small; serving does not need a GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

COPY configs ./configs
COPY models/FD004_condition ./models/FD004_condition

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV RUL_SUBSET=FD004
EXPOSE 8000
CMD ["uvicorn", "rul.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
