FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV COMFYUI_PORT=8188

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git wget curl dos2unix aria2 megatools media-types \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev  \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-dejavu-core fontconfig \
    libegl1 libglx-mesa0 libglu1-mesa libgles2 libosmesa6 mesa-utils \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

    # Permite que aiohttp entregue correctamente JS/CSS; Firefox bloquea octet-stream con nosniff.
RUN python3 -c "import mimetypes; assert mimetypes.guess_type('asset.js')[0] in ('application/javascript', 'text/javascript'); assert mimetypes.guess_type('asset.css')[0] == 'text/css'"

# aiohttp uses the OS MIME database for static extension assets.  Without
# media-types it can label JavaScript/CSS as application/octet-stream, which
# Firefox correctly rejects when nosniff is enabled.
RUN python3 -c "import mimetypes; assert mimetypes.guess_type('asset.js')[0] in ('application/javascript', 'text/javascript'); assert mimetypes.guess_type('asset.css')[0] == 'text/css'"

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Bootstrap pip directamente para 3.11 (python3-pip del sistema instala para 3.10, no sirve aquí)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# --- PyTorch: versión explícita, controlada por ti (no heredada de un tercero) ---
# cu128 confirmado funcional en tus pruebas con 3090 Ti y 4090.
# Si más adelante RunPod resuelve el soporte de Blackwell, prueba cambiar a cu130.
RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128


# --- SageAttention: requiere compilar contra el torch/CUDA ya instalados arriba ---
# Necesita nvcc (por eso la imagen base es -devel, no -runtime) y puede tardar varios minutos en compilar.
RUN pip install --no-cache-dir triton==3.5.1 ninja
# RTX 30/A5000/A6000 = 8.6
# RTX 40 = 8.9
# RTX 50 = 12.0
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ENV MAX_JOBS=2
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention \
    && git checkout eb615cf \
    && EXT_PARALLEL=1 NVCC_APPEND_FLAGS="--threads 2" python setup.py install \
    && rm -rf /tmp/SageAttention

# --- Clonar ComfyUI directamente desde el repo oficial ---
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI
RUN pip install --no-cache-dir -r /ComfyUI/requirements.txt
# Evita los 116 blueprints globales que fallan al cargar el frontend.
RUN rm -rf /ComfyUI/blueprints

# ComfyUI 0.36.0 / frontend 1.52.7 currently fails to deserialize every bundled
# global blueprint at startup.  They are optional example workflows; removing
# them prevents the 116 failed subgraph loads without removing MiniMax or any
# custom node required by this image.
RUN rm -rf /ComfyUI/blueprints

# --- Constraint global ANTES de instalar cualquier custom node ---
RUN echo "kornia==0.6.12" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt


# --- Custom Nodes ---
RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git && \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    git clone --depth=1 https://github.com/evanspearman/ComfyMath.git && \
    git clone --depth=1 https://github.com/chrisgoringe/cg-use-everywhere.git && \
    git clone --depth=1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone --depth=1 https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone --depth=1 https://github.com/crystian/comfyui-crystools.git && \
    git clone --depth=1 https://github.com/chflame163/ComfyUI_LayerStyle.git && \
    git clone --depth=1 https://github.com/filliptm/ComfyUI_Fill-Nodes.git && \
    git clone --depth=1 https://github.com/farizrifqi/ComfyUI-Image-Saver.git && \
    git clone --depth=1 https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git && \
    git clone --depth=1 https://github.com/vantagewithai/Vantage-Nodes.git && \
    git clone --depth=1 https://github.com/Visionatrix/ComfyUI-Gemini.git && \
    git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git clone --depth=1 https://github.com/scraed/LanPaint.git && \
    git clone --depth=1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git && \
    git clone --depth=1 https://github.com/r-vage/ComfyUI_Eclipse.git && \
    git clone --depth=1 https://github.com/pixaroma/ComfyUI-Pixaroma.git && \
    git clone --depth=1 https://github.com/PGCRT/CRT-Nodes.git && \
    git clone --depth=1 https://github.com/CoreyCorza/ComfyUI-CRZnodes.git && \
    git clone --depth=1 https://github.com/RamonGuthrie/ComfyUI-RBG-SmartSeedVariance.git && \
    git clone --depth=1 https://github.com/xxmjskxx/ComfyUI_SaveImageWithMetaDataUniversal.git && \
    git clone --depth=1 https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git && \
    git clone --depth=1 https://github.com/fblissjr/ComfyUI-QwenImageWanBridge.git && \
    git clone --depth=1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone --depth=1 https://github.com/1038lab/ComfyUI-QwenVL.git && \
    git clone --depth=1 https://github.com/IAMCCS/IAMCCS-nodes.git && \
    git clone --depth=1 https://github.com/SparknightLLC/ComfyUI-INT8-Fast-Fork && \
    git clone --depth=1 https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git && \
    git clone --depth=1 https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git && \
    git clone --depth=1 https://github.com/kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth=1 https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git && \
    git clone --depth=1 https://github.com/lonecatone23/ComfyUI_LC123_nodes.git && \
    git clone --depth=1 https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git && \
    git clone --depth=1 https://github.com/vslinx/ComfyUI-vslinx-nodes.git && \
    git clone --depth=1 https://github.com/M1kep/ComfyLiterals.git && \
    git clone --depth=1 https://github.com/WASasquatch/was-node-suite-comfyui.git && \
    git clone --depth=1 https://github.com/lbouaraba/comfyui-krea2edit.git && \
    git clone --depth=1 https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod.git && \
    git clone --depth=1 https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git && \
    git clone --depth=1 https://github.com/ClownsharkBatwing/RES4LYF.git

 
RUN for dir in rgthree-comfy ComfyUI-Impact-Pack ComfyUI_essentials ComfyUI-GGUF ComfyUI-Impact-Subpack cg-use-everywhere ComfyMath ComfyUI-Custom-Scripts ComfyUI-mxToolkit comfyui-crystools ComfyUI_LayerStyle ComfyUI_Fill-Nodes ComfyUI-Image-Saver ComfyUI-AdvancedLivePortrait ComfyUI-WanVideoWrapper Vantage-Nodes ComfyUI-Gemini ComfyUI-LTXVideo LanPaint ComfyUI_Comfyroll_CustomNodes ComfyUI_Eclipse ComfyUI-Pixaroma CRT-Nodes ComfyUI-CRZnodes ComfyUI-RBG-SmartSeedVariance ComfyUI_SaveImageWithMetaDataUniversal ComfyUI-DaSiWa-Nodes ComfyUI-SeedVR2_VideoUpscaler ComfyUI-QwenImageWanBridge ComfyUI-KJNodes ComfyUI-Easy-Use ComfyUI-QwenVL IAMCCS-nodes RES4LYF ComfyUI-VideoHelperSuite was-node-suite-comfyui; do \
      REQ="/ComfyUI/custom_nodes/${dir}/requirements.txt"; \
      if [ -f "$REQ" ]; then pip install -q -r "$REQ"; fi; \
    done


# pedalboard (usado por CRT-Nodes/Audio_Compressor.py) trae un binario nativo
# que puede exigir instrucciones de CPU (AVX-512) que algunos hosts de Salad
# no tienen. Ahí no lanza un error normal de Python: manda 'Illegal instruction'
# (SIGILL), que mata TODO el proceso de ComfyUI sin que ningún try/except lo
# pueda atrapar — y Salad reinicia el contenedor en la misma máquina una y otra
# vez, repitiendo el mismo crash indefinidamente. Lo desinstalamos para que ese
# import falle como cualquier dependencia opcional faltante (ImportError, que
# ComfyUI sí sabe tolerar y sigue arrancando el resto de los nodos).
RUN pip uninstall -y pedalboard || true

RUN cd /ComfyUI/custom_nodes/ComfyUI-Impact-Pack && python3 install.py || true    
RUN rm -rf /ComfyUI/custom_nodes/ComfyUI-Login /ComfyUI/custom_nodes/ComfyUI-login

# --- Workflows ---
RUN mkdir -p /ComfyUI/user/default/workflows
COPY Krea-T2I-workflow.json /ComfyUI/user/default/workflows/Krea-T2I-workflow.json
COPY Klein-Inpainting-workflow.json /ComfyUI/user/default/workflows/Klein-Inpainting-workflow.json


RUN pip install --no-cache-dir gdown  comfyui-manager
RUN pip install --no-cache-dir -U "huggingface_hub[hf_xet]"

COPY setup_models.sh /setup_models.sh
RUN dos2unix /setup_models.sh && chmod +x /setup_models.sh

EXPOSE 8188

# Verifica el mismo endpoint que debe usar Salad para Startup/Readiness.
# La dirección IPv6 comprueba además que Container Gateway podrá conectarse.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90m --retries=3 \
    CMD curl --noproxy '*' --fail --silent --show-error --max-time 4 \
        "http://[::1]:${COMFYUI_PORT}/system_stats" > /dev/null || exit 1

CMD ["/setup_models.sh"]
