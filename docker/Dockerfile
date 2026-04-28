# Runtime image for local-model deployment.
# Models are mounted at runtime through compose.yaml.
FROM vllm/vllm-openai:v0.21.0

RUN apt-get update && \
    apt-get install -y \
    fonts-noto-core \
    fonts-noto-cjk \
    fontconfig \
    libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MinerU

COPY pyproject.toml README.md LICENSE.md ./
COPY mineru ./mineru

RUN python3 -m pip install -U '.[core]' --break-system-packages && \
    python3 -m pip cache purge
