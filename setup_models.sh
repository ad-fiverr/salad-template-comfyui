#!/bin/bash


# PASO 0: Healthcheck de CUDA — falla rápido si el pod tiene GPU rota
echo "================================================"
echo "  Checking CUDA Before Continuing..."
echo "================================================"
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [ "$CUDA_OK" != "True" ]; then
    echo ""
    echo "🔴 CRITICAL ERROR: CUDA is not available in this pod."
    echo "🔴 nvidia-smi may look fine, but torch.cuda.is_available() = False"
    echo "🔴 This is a host infrastructure issue (GPU passthrough break)."
    echo "🔴 ACTION: STOP this pod and launch a new one — DON'T keep going; you'll waste time and money"
    echo ""
    echo "--- POD DIAGNOSIS ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || echo "nvidia-smi also failed"
    python3 -c "import torch; print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)" 2>&1
    echo "--------------------------"
    exit 1
fi

echo "✅ CUDA available — continuing with the normal setup."

echo "--- SageAttention GPU compatibility test ---"

python3 - <<'PY'
import torch
from sageattention import sageattn

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)

print(f"GPU: {name}")
print(f"Compute capability: {capability}")

q = torch.randn(
    1, 8, 256, 64,
    device="cuda",
    dtype=torch.float16
)

result = sageattn(q, q, q, tensor_layout="HND")
torch.cuda.synchronize()

print(f"SageAttention OK: {tuple(result.shape)}")
PY

if [ $? -ne 0 ]; then
    echo "⚠️ SageAttention failed on this GPU."
    echo "⚠️ ComfyUI will continue, but use SDPA/PyTorch attention."
fi


# =============================================================================
# setup_models.sh - Configuración de RunPod
# =============================================================================

# Token actualizado según tu solicitud
HF_TOKEN="${HF_TOKEN}"
HF_TOKEN_loras="${HF_TOKEN_loras}"
COMFYUI_DIR="/workspace/ComfyUI"

 
# Fuerza los directorios temporales de descarga al volumen de 250GB (/workspace)
# en vez del disco del contenedor (15GB), que se llena con archivos grandes
# (ej. qwen_3_8b.safetensors ~16GB no cabe ni queda espacio en un disco de 15GB).
export TMPDIR="/workspace/tmp"
mkdir -p "$TMPDIR"
 


echo "================================================"
echo "  ComfyUI Model Setup — ALL IN ONE Edition"
echo "  THANKS FOR YOUR ORDER, ADRIANFIVERR"
echo "================================================"

# PASO 1: Persistencia de ComfyUI
if [ ! -f "${COMFYUI_DIR}/main.py" ]; then
    echo "[ Copiando ComfyUI base a /workspace... ]"
    mkdir -p ${COMFYUI_DIR}
    cp -rn /ComfyUI/. ${COMFYUI_DIR}/
fi

# PASO 2: Preparar Directorios
mkdir -p ${COMFYUI_DIR}/models/loras \
         ${COMFYUI_DIR}/models/checkpoints \
         ${COMFYUI_DIR}/models/diffusion_models \
         ${COMFYUI_DIR}/models/text_encoders \
         ${COMFYUI_DIR}/models/upscale_models \
         ${COMFYUI_DIR}/models/SEEDVR2 \
         ${COMFYUI_DIR}/models/vae \
         ${COMFYUI_DIR}/models/ultralytics/bbox \
         ${COMFYUI_DIR}/models/ultralytics/segm \
         ${COMFYUI_DIR}/models/loras/ \
         ${COMFYUI_DIR}/models/sams \
         ${COMFYUI_DIR}/custom_nodes/ComfyUI_essentials/luts  \
         ${COMFYUI_DIR}/models/sam3  
         

# ── Funciones de descarga ─────────────────────────────────────────────────────

download_if_missing() {
    local url="$1"
    local dest="$2"
    local auth="${3:-}"
    local conns="${4:-16}"
    local hf_timeout="${HF_FILE_TIMEOUT:-1200}"

    local dest_dir
    local file_name
    dest_dir="$(dirname "$dest")"
    file_name="$(basename "$dest")"

    mkdir -p "$dest_dir"

    # Un archivo .aria2 indica una descarga incompleta.
    if [ -s "$dest" ] && [ ! -e "${dest}.aria2" ]; then
        echo "✅ Ya existe: $file_name"
        return 0
    fi

    # Convierte enlaces de la página HTML en enlaces de descarga.
    url="${url/\/blob\//\/resolve\/}"

    if [[ "$url" == *"huggingface.co"* ]]; then
        local repo_type=""
        local path_part="${url#*huggingface.co/}"
        path_part="${path_part%%\?*}"

        if [[ "$path_part" == datasets/* ]]; then
            repo_type="dataset"
            path_part="${path_part#datasets/}"
        fi

        # owner/repo/resolve/revision/ruta/archivo
        if [[ "$path_part" =~ ^([^/]+/[^/]+)/resolve/([^/]+)/(.*)$ ]]; then
            local repo="${BASH_REMATCH[1]}"
            local revision="${BASH_REMATCH[2]}"
            local file_in_repo="${BASH_REMATCH[3]}"
            local tmp_dir
            tmp_dir="$(mktemp -d)"

            local hf_opts=(
                download "$repo" "$file_in_repo"
                --revision "$revision"
                --local-dir "$tmp_dir"
            )

            [ -n "$repo_type" ] &&
                hf_opts+=(--repo-type "$repo_type")

            echo "⬇️ HF/Xet: $file_name"

            (
                # HF_TOKEN se mantiene fuera de los argumentos visibles de `ps`.
                [ -n "$auth" ] && export HF_TOKEN="$auth"

                # Más estable para una instancia de 60 GB que HIGH_PERFORMANCE.
                unset HF_XET_HIGH_PERFORMANCE HF_XET_HP
                export HF_XET_NUM_CONCURRENT_RANGE_GETS=32
                export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
                export HF_HUB_DOWNLOAD_TIMEOUT=60

                timeout --signal=TERM "${hf_timeout}s" \
                    hf "${hf_opts[@]}"
            ) &

            local hf_pid=$!
            local previous_size=0
            local current_size=0
            local interval_speed=0

            # Imprime progreso aunque `hf` no tenga una terminal interactiva.
            while kill -0 "$hf_pid" 2>/dev/null; do
                sleep 15

                current_size="$(
                    du -sb "$tmp_dir" 2>/dev/null |
                    awk '{print $1}'
                )"
                current_size="${current_size:-0}"

                interval_speed=$(
                    (current_size - previous_size) /
                    15 / 1024 / 1024
                )

                echo "📦 $file_name: $(
                    (current_size / 1024 / 1024)
                ) MiB | ${interval_speed} MiB/s"

                previous_size="$current_size"
            done

            wait "$hf_pid"
            local hf_status=$?

            if [ "$hf_status" -eq 0 ] &&
               [ -s "$tmp_dir/$file_in_repo" ]; then
                mv -f "$tmp_dir/$file_in_repo" "$dest"
                rm -rf "$tmp_dir"

                echo "✅ HF/Xet completado: $file_name"
                return 0
            fi

            rm -rf "$tmp_dir"

            if [ "$hf_status" -eq 124 ]; then
                echo "⚠️ HF/Xet excedió ${hf_timeout}s."
            else
                echo "⚠️ HF/Xet falló con código $hf_status."
            fi

            echo "↪️ Reintentando con aria2c: $file_name"
        fi
    fi

    local part_name="${file_name}.part"
    local part_path="${dest_dir}/${part_name}"

    local aria2_opts=(
        "-x" "$conns"
        "-s" "$conns"
        "-k" "50M"
        "--disk-cache=256M"
        "--file-allocation=falloc"
        "-c"
        "--max-tries=8"
        "--retry-wait=5"
        "--timeout=60"
        "--console-log-level=notice"
        "--summary-interval=5"
        "-d" "$dest_dir"
        "-o" "$part_name"
    )

    echo "⬇️ aria2c: $file_name ($conns conexiones)"

    if [ -n "$auth" ]; then
        aria2c \
            --header="Authorization: Bearer $auth" \
            "${aria2_opts[@]}" \
            "$url"
    else
        aria2c "${aria2_opts[@]}" "$url"
    fi

    local aria_status=$?

    if [ "$aria_status" -eq 0 ] && [ -s "$part_path" ]; then
        mv -f "$part_path" "$dest"
        rm -f "${part_path}.aria2"

        echo "✅ aria2c completado: $file_name"
        return 0
    fi

    echo "❌ Error al descargar: $file_name"
    return 1
}


download_gdown_if_missing() {
    local id="$1" dest="$2" type="$3"
    
    if [ "$type" = "folder" ]; then
        # LÓGICA PARA CARPETAS
        # Verifica si el destino es un directorio (-d) y si no está vacío
        if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
            echo "  Carpeta ya existe y tiene archivos: $(basename "$dest")"
            return 0;
        fi
        
        echo "  Descargando CARPETA desde Drive: $(basename "$dest")"
        gdown --folder "$id" -O "$dest"
        
    else
        # LÓGICA PARA ARCHIVOS (Tu código original)
        # Verifica si el archivo existe (-f) y pesa más de 5MB
        if [ -f "$dest" ] && [ $(find "$dest" -type f -size +5M 2>/dev/null) ]; then 
            return 0; 
        fi
        
        echo "  Descargando ARCHIVO desde Drive: $(basename "$dest")"
        gdown "$id" -O "$dest"
    fi
}

download_hf_repo() {
    local repo="$1" dest_dir="$2"
    echo "  Descargando repo HF: $repo en $dest_dir"
    HF_TOKEN=${HF_TOKEN} huggingface-cli download "$repo" --local-dir "$dest_dir" --local-dir-use-symlinks False
}

download_hf_repo_aria2c() {
    local repo="$1" dest_dir="$2" auth="$3"

    # 1. Autoinstalación de 'jq' si no existe
    if ! command -v jq &> /dev/null; then
        echo "⚙️ 'jq' no encontrado. Instalando automáticamente..."
        
        # Evita que apt pida confirmaciones o menús interactivos que congelen el script
        export DEBIAN_FRONTEND=noninteractive 
        
        # Actualiza las listas e instala jq de forma silenciosa
        apt-get update -qq && apt-get install -y jq > /dev/null 2>&1
        
        # Verifica si la instalación fue exitosa
        if ! command -v jq &> /dev/null; then
            echo "❌ No se pudo instalar 'jq' automáticamente. Abortando."
            return 1
        fi
        echo "✅ 'jq' instalado correctamente."
    fi

    echo "🔍 Listando archivos del repositorio: $repo"
    
    # 2. Manejo dinámico del token de seguridad
    local curl_cmd=(curl -s -f)
    if [ -n "$auth" ]; then
        curl_cmd+=(-H "Authorization: Bearer $auth")
    fi

    # 3. Consulta a la API
    local files
    files=$("${curl_cmd[@]}" "https://huggingface.co/api/models/$repo" | jq -r '.siblings[].rfilename 2>/dev/null')

    # 4. Validación estricta de errores
    if [ $? -ne 0 ] || [ -z "$files" ] || [ "$files" = "null" ]; then
        echo "❌ No se encontraron archivos o acceso denegado (Revisa el nombre del repo y tu token HF)."
        return 1
    fi

    echo "📦 Archivos encontrados. Iniciando descarga por lotes..."

    # 5. Bucle de descarga
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        
        local url="https://huggingface.co/$repo/resolve/main/$file"
        local dest="$dest_dir/$file"
        
        download_if_missing "$url" "$dest" "$auth"
        
    done <<< "$files"
    
    echo "✅ Repositorio procesado completamente: $repo"
}



echo "Instalando huggingface_hub..."

echo "Auth with Hugging Face..."
# Usamos el comando de Python para el login con el token proporcionado
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# ── SECCIÓN DE DESCARGAS MODELOS DE VIDEO  ─────────────────────────


echo "[ ------- Downloading Diffusion Models -------]"
cd ${COMFYUI_DIR}/models/diffusion_models && rm -rf split_files/
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$HF_TOKEN"

download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors" "$HF_TOKEN"



echo "[ Text Encoders ]"
cd ${COMFYUI_DIR}/models/text_encoders && rm -rf split_files/
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$HF_TOKEN"
    
# ------------------------------ LORAS ---
echo "[ LoRAs ]"
cd ${COMFYUI_DIR}/models/loras && rm -rf split_files/
download_if_missing "https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors" \
    "minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors" "$HF_TOKEN"
    


echo "[ VAE ]"
cd ${COMFYUI_DIR}/models/vae && rm -rf split_files/
download_if_missing "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors" \
    "Wan2_1_VAE_fp32.safetensors" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
    "minimax_h3_video_vae_fp16.safetensors" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
    "minimax_h3_audio_vae_fp32.safetensors" "$HF_TOKEN"


# --- VAE ---
echo "[ VAE ]"
cd ${COMFYUI_DIR}/models/vae && rm -rf split_files/
download_if_missing "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
    "ae.safetensors" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "flux2-vae.safetensors" "$HF_TOKEN"
download_if_missing "https://civitai.red/api/download/models/3068442?fileId=2947164&token=e3a803e3831ec4832fd75d014b2d385e" \
    "krea2RealVae_v10.safetensors" "$HF_TOKEN"




(
# --- SAM3 ---
echo "[ ----------- Downloading SAM3 -----------  ]"
cd ${COMFYUI_DIR}/models/sam3
download_if_missing "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt" \
    "sam3.pt" "$HF_TOKEN"


    # ── SAMS (ReActor/Segment Anything) ──────────────────────────────────────────
echo "[ SAM3 ]"
cd ${COMFYUI_DIR}/models/sams
download_if_missing "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth" \
    "sam_vit_b_01ec64.pth" "$HF_TOKEN"
download_if_missing "https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth" \
    "sam_vit_h_4b8939.pth" "$HF_TOKEN"



# ── SECCIÓN DE DESCARGAS MODELOS DE IMAGEN ─────────────────────────
# --- DIFFUSION MODELS ---
echo "[ ------- Downloading Diffusion Models -------]"
cd ${COMFYUI_DIR}/models/diffusion_models && rm -rf split_files/

download_if_missing "https://huggingface.co/exjadev/diffusion_models/resolve/main/krast_v20.safetensors" \
    "krast_v20.safetensors" "$HF_TOKEN"

cd ${COMFYUI_DIR}/models/diffusion_models 
download_if_missing "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
    "z_image_turbo_bf16.safetensors" "$HF_TOKEN"

cd ${COMFYUI_DIR}/models/diffusion_models 
download_if_missing "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" \
    "flux-2-klein-9b-fp8.safetensors" "$HF_TOKEN"


# --- TEXT ENCODERS ---
echo "[ Text Encoders ]"
cd ${COMFYUI_DIR}/models/text_encoders && rm -rf split_files/
download_if_missing "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors" \
    "qwen3vl_4b_fp8_scaled.safetensors" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors" \
    "qwen_3_8b.safetensors" "$HF_TOKEN"


echo "[-----------  Downloading BBOX Ultralytics SEGM -----------  ]"
cd ${COMFYUI_DIR}/models/ultralytics/segm
download_if_missing "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt" \
    "person_yolov8m-seg.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/24xx/segm/resolve/main/deepfashion2_yolov8s-seg.pt" \
    "deepfashion2_yolov8s-seg.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/24xx/segm/resolve/main/hair_yolov8n-seg_60.pt" \
    "hair_yolov8n-seg_60.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/24xx/segm/resolve/main/skin_yolov8n-seg_800.pt" \
    "skin_yolov8n-seg_800.pt" "$HF_TOKEN"

# ── BBOX Ultralytics ──────────────────────────────────────────────────────────
echo ""
echo "[ BBOX Ultralytics ]"
cd ${COMFYUI_DIR}/models/ultralytics/bbox && rm -rf split_files/
download_if_missing "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" \
    "face_yolov8m.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/ashllay/YOLO_Models/resolve/main/bbox/female_breast-v4.2.pt" \
    "female_breast-v4.2.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/ashllay/YOLO_Models/resolve/main/bbox/vagina-v3.2.pt" \
    "vagina-v3.2.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/ashllay/YOLO_Models/resolve/main/bbox/full_eyes_detect_v1.pt" \
    "full_eyes_detect_v1.pt" "$HF_TOKEN"
download_if_missing "https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/hand_yolov8s.pt" \
    "hand_yolov8s.pt" "$HF_TOKEN"

# ------------------------------ LORAS ---
echo "[ LoRAs ]"
cd ${COMFYUI_DIR}/models/loras && rm -rf recipes/
# Civitai filters & loras
download_if_missing "https://civitai.red/api/download/models/3067151?type=Model&format=SafeTensor&token=e3a803e3831ec4832fd75d014b2d385e" \
    "krea2filterbypass3.safetensors"




# ── Upscaler Models ──────────────────────────────────────────────────────────
echo ""
echo "[ -----------  Downloading UUpscaling  Models  ----------- ]"
cd ${COMFYUI_DIR}/models/upscale_models && rm -rf split_files/
download_if_missing "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" \
    "4x_foolhardy_Remacri.pth" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors" \
    "4x-UltraSharpV2.safetensors" "$HF_TOKEN"


download_gdown_if_missing "1N3ysO2IWkouzy4aFONLgYUjaUMrLz8AB" "4xFFHQDAT.pth"

echo "[ -----------  Creating BROKEN_NCNN  ----------- ]"
cd ${COMFYUI_DIR}/models/upscale_models/
megadl 'https://mega.nz/folder/Xc4wnC7T#yUS5-9-AbRxLhpdPW_8f2w'


# --- LUTS ---
# ── Luts  ──────────────────────────────────────────────────────────
echo "[ VAE ]"
cd ${COMFYUI_DIR}/custom_nodes/ComfyUI_essentials/luts 
echo ""
echo "[ ----------Downloading LUTs --------------]"
download_gdown_if_missing "1GJEhRrycKwMINkgicw_GjQbjuwdqRJ9P" "LUTs" "folder"



) &


cd ${COMFYUI_DIR}
# 2. Escribir los permisos de los modelos en la lista blanca
echo "4x-UltraSharpV2.safetensors" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt
echo "4xFFHQDAT.pth" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt
echo "4x_foolhardy_Remacri.pth" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt
echo "BROKEN_NCNN/4x-ClearRealityV1-fp16.bin" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt
echo "4x-ClearRealityV1.pth" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt

# Autorización para el modelo SwinIR
echo "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" >> /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt


# ── Lanzar ComfyUI ────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Setup full. starting ComfyUI..."
echo "================================================"

chmod -R 777 /workspace/ComfyUI

COMFYUI_PORT="${COMFYUI_PORT:-8188}"

# Salad Container Gateway entra por IPv6. Escuchar en :: permite que el
# gateway y las probes HTTP alcancen ComfyUI directamente.
exec python /workspace/ComfyUI/main.py \
    --listen "::" \
    --port "$COMFYUI_PORT" \
    --enable-manager
