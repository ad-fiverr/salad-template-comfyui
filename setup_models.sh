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
# PASO 0.5: Healthcheck de Red (SaladCloud Network Bandwidth Check)
# =============================================================================
echo "================================================"
echo "  Checking Network Bandwidth Before Startup..."
echo "================================================"

# Salad usa PCs residenciales con anchos de banda muy variables.
# Forzamos una reasignación (IMDS) si la red no da al menos 20 Mbps,
# para evitar quedarnos atascados horas descargando safetensors masivos.

MIN_MBPS="60" # Requisito mínimo (20 Mbps ~ 2.5 MB/s). Ajústalo si necesitas más.
SPEED_TEST_URL="https://speed.cloudflare.com/__down?bytes=25000000" # Archivo de 25MB
SPEED_TEST_ATTEMPTS="3"
best_mbps="0"
NETWORK_OK="false"

for attempt in $(seq 1 "$SPEED_TEST_ATTEMPTS"); do
  echo "Speed test attempt ${attempt}/${SPEED_TEST_ATTEMPTS}..."

  if speed_bps=$(curl -L -o /dev/null -sS \
      --connect-timeout 5 \
      --max-time 30 \
      -w "%{speed_download}" \
      "$SPEED_TEST_URL"); then

    # Convierte bytes/seg a Megabits/seg (Mbps)
    mbps=$(awk -v s="$speed_bps" 'BEGIN {printf "%.2f", s * 8 / 1000000}')
    echo "Measured download speed: ${mbps} Mbps"

    # Guarda el mejor resultado
    best_mbps=$(awk -v best="$best_mbps" -v current="$mbps" \
      'BEGIN {printf "%.2f", (current > best ? current : best)}')

    # Si supera el mínimo, la red sirve y rompemos el bucle
    if awk -v current="$mbps" -v min="$MIN_MBPS" \
      'BEGIN {exit !(current >= min)}'; then
      echo "✅ Speed check passed. Network is stable."
      NETWORK_OK="true"
      break
    fi
  else
    echo "⚠️ Speed test attempt failed (posible 'Network is unreachable')."
  fi

  sleep 10
done

if [ "$NETWORK_OK" != "true" ]; then
  reason="Insufficient download bandwidth or no network: best measured ${best_mbps} Mbps, required ${MIN_MBPS} Mbps"
  
  echo "🔴 CRITICAL ERROR: Speed check failed. Requesting Salad replica reallocation..."
  echo "$reason"

  # Llama a la API interna de Salad (IMDS) para reasignar este contenedor a otro nodo
  curl -sS \
    --request POST \
    --url "http://169.254.169.254/v1/reallocate" \
    --header "Content-Type: application/json" \
    --header "Metadata: true" \
    --data "{\"reason\":\"${reason}\"}" || true

  echo "Reallocation requested. Exiting."
  exit 1
fi

echo "================================================"


# =============================================================================
# setup_models.sh - Configuración de RunPod
# =============================================================================

# Token actualizado según tu solicitud
HF_TOKEN="${HF_TOKEN}"
HF_TOKEN_loras="${HF_TOKEN_loras}"
COMFYUI_DIR="/workspace/ComfyUI"

# =============================================================================
# PASO 0.6: Validación "fail-fast" del token de Hugging Face
# =============================================================================
# La API de HF devuelve 404 (no 401) para repos protegidos cuando el token
# es inválido/revocado — así que un 404 durante la descarga NO nos dice si el
# repo no existe o si el token está muerto. En vez de descubrirlo 15 archivos
# después (y peor, dejar que aria2c reintente el mismo token muerto), lo
# comprobamos UNA vez aquí con una llamada barata a whoami-v2 y abortamos
# inmediatamente si no es válido.
validate_hf_token() {
    local token="$1"
    local label="${2:-HF_TOKEN}"

    if [ -z "$token" ]; then
        echo "🔴 CRITICAL ERROR: $label está vacío — no se definió la variable de entorno."
        return 1
    fi

    echo "🔑 Validando $label contra la API de Hugging Face..."

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 10 --max-time 15 \
        -H "Authorization: Bearer $token" \
        "https://huggingface.co/api/whoami-v2")

    if [ "$http_code" = "200" ]; then
        echo "✅ $label válido."
        return 0
    elif [ "$http_code" = "401" ]; then
        echo "🔴 CRITICAL ERROR: $label fue RECHAZADO (401) — el token es inválido o fue revocado."
        echo "🔴 Esto suele pasar cuando GitHub/HF Secret Scanning detecta el token expuesto"
        echo "🔴 (por ejemplo en logs de CI/CD) y lo invalida automáticamente."
        echo "🔴 ACCIÓN: genera un token nuevo en https://huggingface.co/settings/tokens,"
        echo "🔴 actualízalo como secret SIN imprimirlo en ningún log, y vuelve a lanzar."
        return 1
    else
        echo "⚠️  No se pudo verificar $label (HTTP $http_code) — puede ser un problema de red."
        echo "⚠️  Continuando, pero si las descargas fallan revisa el token manualmente."
        return 0
    fi
}

if ! validate_hf_token "$HF_TOKEN" "HF_TOKEN"; then
    exit 1
fi
# Solo valida el segundo token si de verdad se usa en este script y viene definido.
if [ -n "$HF_TOKEN_loras" ]; then
    validate_hf_token "$HF_TOKEN_loras" "HF_TOKEN_loras" || exit 1
fi

 
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

                interval_speed=$(( (current_size - previous_size) / 15 / 1024 / 1024 ))

                echo "📦 $file_name: $(( current_size / 1024 / 1024 )) MiB | ${interval_speed} MiB/s"

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

    # errorCode=24 de aria2c = "Authorization failed" (HTTP 401/403).
    # No tiene sentido dejar que las siguientes ~20 descargas repitan el mismo
    # error con el mismo token; abortamos todo el script de inmediato.
    if [ "$aria_status" -eq 24 ]; then
        echo "🔴 CRITICAL ERROR: aria2c recibió 'Authorization failed' (código 24) para $file_name."
        echo "🔴 El token fue rechazado a mitad de las descargas (pudo haberse revocado recién)."
        echo "🔴 Abortando en vez de seguir fallando en cascada. Revisa/rota el token y reinicia."
        exit 1
    fi

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
download_if_missing "https://huggingface.co/TenStrip/10Eros-Max/resolve/main/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" \
    "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" "$HF_TOKEN"


echo "[ Text Encoders ]"
cd ${COMFYUI_DIR}/models/text_encoders && rm -rf split_files/
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$HF_TOKEN"
    
# ------------------------------ LORAS ---
echo "[ LoRAs ]"
cd ${COMFYUI_DIR}/models/loras && rm -rf split_files/
download_if_missing "https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI/resolve/main/taomate_h3_3step_comfy.safetensors" \
    "taomate_h3_3step_comfy.safetensors" "$HF_TOKEN"
    
download_if_missing "https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors" \
    "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors" "$HF_TOKEN"
    


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
download_if_missing "https://huggingface.co/wikeeyang/Krea2-Turbo-HD-V1/resolve/main/Krea2-HD-vae.safetensors" \
    "Krea2-HD-vae.safetensors" "$HF_TOKEN"




(
# --- SAM3 ---
echo "[ ----------- Downloading SAM3 -----------  ]"
cd ${COMFYUI_DIR}/models/sam3

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

download_if_missing "https://huggingface.co/enzinoai/IntoRealism-Krea-2/resolve/main/Krea2IntoRealismV1-Int8.safetensors" \
    "IntoRealismKrea2.safetensors" 


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



# ── Upscaler Models ──────────────────────────────────────────────────────────
echo ""
echo "[ -----------  Downloading UUpscaling  Models  ----------- ]"
cd ${COMFYUI_DIR}/models/upscale_models && rm -rf split_files/
download_if_missing "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" \
    "4x_foolhardy_Remacri.pth" "$HF_TOKEN"
download_if_missing "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors" \
    "4x-UltraSharpV2.safetensors" "$HF_TOKEN"
download_if_missing "https://huggingface.co/holwech/universal-upscaler-v2-esrgan/resolve/main/4x_UniversalUpscalerV2-Neutral_115000_swaG.pth" \
    "4x_UniversalUpscalerV2-Neutral_115000_swaG.pth" "$HF_TOKEN"


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

echo "[ Configurando la desactivación de Nodes 2.0... ]"
python3 -c "
import json, os
from contextlib import suppress
filepath = '/workspace/ComfyUI/user/default/comfy.settings.json'
os.makedirs(os.path.dirname(filepath), exist_ok=True)
data = {}
with suppress(FileNotFoundError, json.JSONDecodeError): data = json.load(open(filepath))
data['Comfy.VueNodes.Enabled'] = False
json.dump(data, open(filepath, 'w'), indent=4)
"

cd ${COMFYUI_DIR}
mkdir -p /workspace/ComfyUI/user/default/ComfyUI-Impact-Subpack
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

rm -rf /workspace/ComfyUI/user/__manager/cache/*

exec python /workspace/ComfyUI/main.py \
    --listen "::" \
    --port "$COMFYUI_PORT" \
    --enable-manager