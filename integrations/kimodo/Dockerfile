# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/kimodo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /opt/kimodo

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake libeigen3-dev ca-certificates \
      libsm6 libice6 libx11-6 libxext6 libxrender1 libxi6 libxxf86vm1 \
      libxfixes3 libxcb1 libxkbcommon0 libgl1 libegl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY integrations/kimodo/requirements.txt /tmp/kimodo-requirements.txt
# The base supplies PyTorch and its CUDA libraries. Do not resolve or replace them.
RUN python -m pip install --no-deps -r /tmp/kimodo-requirements.txt \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__"

COPY .local/kimodo/source/ /opt/kimodo/
RUN CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)" \
      python -m pip install --no-deps --no-build-isolation ./MotionCorrection \
    && SKIP_MOTION_CORRECTION_IN_SETUP=1 \
      python -m pip install --no-deps --no-build-isolation . \
    && python -m pip check \
    && python -c "import kimodo, motion_correction, bitsandbytes; from kimodo.exports.fbx import save_motion_fbx" \
    && find /opt/kimodo/tools -type f -name blender -exec chmod +x {} +

# Patches are kept after dependency layers so fixes do not rebuild PyTorch packages.
COPY integrations/kimodo/runtime_patch.py integrations/kimodo/runtime_service.py integrations/kimodo/prepare_retarget_motion.py /opt/kimodo/
RUN python /opt/kimodo/runtime_patch.py /opt/kimodo

LABEL org.opencontainers.image.title="Kimodo runtime for Cascadeur" \
      org.opencontainers.image.source="https://github.com/Rickcy/mcp-for-cascadeur" \
      org.opencontainers.image.description="Offline SOMA motion generation with a local LLM2Vec encoder"

CMD ["python", "/opt/kimodo/scripts/cascadeur_backend_service.py", "--host", "0.0.0.0", "--port", "9552", "--text-encoder-profile", "llama-4bit", "--preload-dataset", "RP"]
