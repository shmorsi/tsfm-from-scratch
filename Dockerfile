# tsfm forecasting service: the committed CPU model, exported to ONNX at build time, behind FastAPI.
#   docker build -t tsfm-serve .
#   docker run --rm -p 8000:8000 tsfm-serve
#   curl -s localhost:8000/forecast -H 'content-type: application/json' \
#        -d '{"series": [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]], "horizon": 8}'
#
# Two stages. The build stage installs the locked research environment to run the export
# (tsfm serve export) and the drift reference; the runtime stage carries only what serving
# imports (numpy, CPU PyTorch, ONNX Runtime, FastAPI, uvicorn), pinned to the lockfile's versions.

FROM python:3.12-slim AS build
RUN pip install --no-cache-dir uv==0.9.7
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra cpu --extra serve
COPY src ./src
COPY results/runs/cpu_s0_seed0/model.pt ./results/runs/cpu_s0_seed0/model.pt
RUN uv sync --locked --no-dev --extra cpu --extra serve \
 && .venv/bin/tsfm serve export --out artifacts/serve

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TSFM_THREADS=2
WORKDIR /app
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy==1.26.4 onnxruntime==1.32.0 fastapi==0.165.0 uvicorn==0.54.0 pydantic==2.14.1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY --from=build /app/results/runs/cpu_s0_seed0/model.pt ./results/runs/cpu_s0_seed0/model.pt
COPY --from=build /app/artifacts/serve ./artifacts/serve
RUN useradd --create-home --uid 10001 tsfm && chown -R tsfm /app
USER tsfm
ENV TSFM_CHECKPOINT=/app/results/runs/cpu_s0_seed0/model.pt TSFM_ONNX=/app/artifacts/serve/model.onnx \
    TSFM_REFERENCE=/app/artifacts/serve/reference_features.npy
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "tsfm.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
