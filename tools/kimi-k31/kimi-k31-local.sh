#!/usr/bin/env bash
# Kimi-K3.1-Pro locally with llama.cpp (upstream K3.1 support, ggml-org/llama.cpp#31779; no patch needed)
# target: Arch Linux, 5x NVIDIA Turing (sm_75, ~48 GB each), 512 GB RAM
#
#   ./kimi-k31-local.sh build      # deps check, clone llama.cpp at LLAMA_REF, CUDA build for sm_75
#   ./kimi-k31-local.sh download   # moonshotai/Kimi-K3.1-Pro, 1.56 TB
#   ./kimi-k31-local.sh convert    # -> MXFP4 experts + BF16 rest GGUF (~1.5 TB, lossless)
#   ./kimi-k31-local.sh imatrix    # importance matrix (IQ1/IQ2 types require one), needs CALIB=file
#   ./kimi-k31-local.sh quantize   # -> IQ1_S (~595 GB), fits 512 GB RAM + 240 GB VRAM
#   ./kimi-k31-local.sh serve      # llama-server, OpenAI API on :8080
#
#   shortcut: a ready-made K3.1 GGUF instead of download + convert + imatrix + quantize
#   GGUF_REPO=owner/name QTYPE=IQ1_S ./kimi-k31-local.sh fetch     # files matching *QTYPE*, into $GGUF_DIR/prebuilt
#   MODEL=<first shard> ./kimi-k31-local.sh serve
#
# disk: ~1.6 TB (HF) + ~1.5 TB (MXFP4 GGUF) + ~0.6 TB (IQ1_S). The HF download can be deleted after convert.
set -euo pipefail

WORK=${WORK:-$HOME/kimi-k31}
PATCH=${PATCH:-$(dirname "$(readlink -f "$0")")/0001-model-add-Kimi-K3.1-DSA-lightning-indexer.patch}
HF_DIR=${HF_DIR:-$WORK/hf/Kimi-K3.1-Pro}
GGUF_DIR=${GGUF_DIR:-$WORK/gguf}
BASE_GGUF=$GGUF_DIR/Kimi-K3.1-Pro-MXFP4/Kimi-K3.1-Pro-MXFP4.gguf
QTYPE=${QTYPE:-IQ1_S}                       # IQ1_S ~595 GB; IQ1_M ~650 GB is the tight upper bound for 752 GB
QUANT_GGUF=$GGUF_DIR/Kimi-K3.1-Pro-$QTYPE/Kimi-K3.1-Pro-$QTYPE.gguf
MODEL=${MODEL:-$QUANT_GGUF}                 # serve this; set it to a fetched file's first shard
IMATRIX=${IMATRIX:-$GGUF_DIR/imatrix-K3.1.gguf}
CTX=${CTX:-32768}
LCPP=$WORK/llama.cpp
# bcb01528 = "model: add Kimi-K3.1-Pro text model (#31779)", the first upstream commit with architecture kimi-k31.
# An older ref (e.g. 669204ee, 3 h before the merge) has no K3.1: the patch is then applied, but it writes a different
# architecture (kimi-k3 + indexer keys) whose GGUFs do not load on upstream, and upstream's do not load there.
LLAMA_REF=${LLAMA_REF:-bcb0152876e8f0b684ba70d3defcab6cdf35adb5}
BIN=$LCPP/build/bin
THREADS=${THREADS:-$(nproc)}

say() { printf '\n== %s\n' "$*"; }
first_split() {   # model.gguf -> model-00001-of-0000N.gguf when the model is sharded
    local s=("${1%.gguf}"-00001-of-*.gguf)
    if [ -e "${s[0]}" ]; then echo "${s[0]}"; else echo "$1"; fi
}
check_arch() {   # a GGUF written for the other llama.cpp variant (kimi-k3 + indexer vs kimi-k31) does not load
    local f want=kimi-k3
    f=$(first_split "$1")
    [ -e "$f" ] || return 0
    [ -e "$LCPP/src/models/kimi-k31.cpp" ] && want=kimi-k31
    head -c 16384 "$f" | grep -aq "${want}\.block_count" ||
        { echo "$f is not a $want GGUF for the llama.cpp at $LCPP: run convert again after build" >&2; exit 1; }
}

build() {
    say "dependencies"
    sudo pacman -S --needed --noconfirm base-devel cmake git cuda python uv
    # Arch's cuda lives in /opt/cuda; its profile script sets PATH and NVCC_CCBIN (the supported host gcc)
    # shellcheck disable=SC1091
    [ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
    export PATH=/opt/cuda/bin:$PATH
    nvcc --version | tail -2
    nvcc --list-gpu-arch | grep -qx compute_75 || { echo "this nvcc cannot target sm_75"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

    say "llama.cpp @ $LLAMA_REF"
    mkdir -p "$WORK"
    [ -d "$LCPP" ] || git clone https://github.com/ggml-org/llama.cpp "$LCPP"
    cd "$LCPP"
    git cat-file -e "$LLAMA_REF^{commit}" 2>/dev/null || git fetch --quiet origin "$LLAMA_REF"
    git checkout --quiet "$LLAMA_REF"
    if [ -e src/models/kimi-k31.cpp ]; then
        echo "upstream K3.1 present: no patch"
    else
        echo "no upstream K3.1 at this ref: applying the legacy patch (architecture kimi-k3)"
        git -c user.name=local -c user.email=local@localhost am "$PATCH"
    fi

    say "CUDA build for sm_75"
    local host=()
    [ -n "${NVCC_CCBIN:-}" ] && host=(-DCMAKE_CUDA_HOST_COMPILER="$NVCC_CCBIN")
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON \
        -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc "${host[@]}"
    cmake --build build -j"$THREADS" --target llama-server llama-cli llama-quantize llama-imatrix llama-gguf-split

    say "python env for the converter"
    uv venv "$WORK/venv"
    uv pip install --python "$WORK/venv/bin/python" -r requirements/requirements-convert_hf_to_gguf.txt tiktoken
}

download() {
    say "download moonshotai/Kimi-K3.1-Pro (1.56 TB)"
    uv tool run --from huggingface_hub hf download moonshotai/Kimi-K3.1-Pro --local-dir "$HF_DIR"
}

convert() {
    say "convert: routed experts repacked to MXFP4 (lossless), the rest BF16, DSA indexer kept"
    mkdir -p "$(dirname "$BASE_GGUF")"
    cd "$LCPP"
    "$WORK/venv/bin/python" convert_hf_to_gguf.py "$HF_DIR" --outtype bf16 \
        --split-max-size 50G --outfile "$BASE_GGUF"
}

imatrix() {
    # The MXFP4 model (~1.5 TB) is larger than RAM+VRAM: attention/shared weights go on the GPUs,
    # the experts stay mmap'd and page in from NVMe. Large ubatches amortize each pass over the experts.
    : "${CALIB:?set CALIB=/path/to/calibration.txt (a few hundred K tokens of mixed chat/code/multilingual text)}"
    check_arch "$BASE_GGUF"
    say "imatrix from $CALIB (slow: bound by NVMe reads, hours)"
    "$BIN/llama-imatrix" -m "$(first_split "$BASE_GGUF")" -f "$CALIB" -o "$IMATRIX" \
        --fit off -ngl 99 --cpu-moe -fa on \
        -c 4096 -b 4096 -ub 4096 --chunks "${CHUNKS:-48}" -t "$THREADS" --output-frequency 4
}

quantize() {
    say "quantize to $QTYPE (indexer q_b/k/proj and attn_k_b/v_b stay unquantized: llama-quant.cpp)"
    mkdir -p "$(dirname "$QUANT_GGUF")"
    check_arch "$BASE_GGUF"
    # --allow-requantize: the experts are MXFP4 in the base GGUF; they dequantize exactly
    # --keep-split: write shards like the input, no single ~600 GB file
    "$BIN/llama-quantize" --allow-requantize --keep-split --imatrix "$IMATRIX" \
        "$(first_split "$BASE_GGUF")" "$QUANT_GGUF" "$QTYPE" "$THREADS"
}

fetch() {
    : "${GGUF_REPO:?set GGUF_REPO=owner/name of a Kimi-K3.1-Pro GGUF repo}"
    local dir=$GGUF_DIR/prebuilt/$(basename "$GGUF_REPO")
    # one quant set only: QTYPE is matched as a substring of the path, e.g. IQ1_S matches KLX-IQ1_S/ but not KLX-IQ1_M/
    say "download $GGUF_REPO (files matching *$QTYPE*, plus README/LICENSE/SHA256SUMS) -> $dir"
    uv tool run --from huggingface_hub hf download "$GGUF_REPO" \
        --include "*$QTYPE*" --include 'README*' --include LICENSE --include SHA256SUMS --local-dir "$dir"
    if [[ -f $dir/SHA256SUMS ]]; then
        say "verify SHA256SUMS (only the files downloaded)"
        (cd "$dir" && sha256sum -c --ignore-missing SHA256SUMS) ||
            { echo "checksum mismatch or unreadable SHA256SUMS: re-run fetch (it resumes)" >&2; exit 1; }
    else
        say "no SHA256SUMS in the repo, skipping checksum verification"
    fi
    find "$dir" -name '*.gguf' ! -name 'mmproj*' ! -name '*imatrix*' | sort | head -1 |
        sed 's/^/first shard (serve with MODEL=...): /'
}

serve() {
    say "llama-server, $MODEL, ctx $CTX"
    check_arch "$MODEL"
    # --fit (default on) places layers/experts across the 5 GPUs and RAM; -fitt keeps a per-GPU margin.
    # manual alternative: --fit off -ngl 99 --n-cpu-moe N (experts of the FIRST N layers on CPU), lower N until VRAM is full
    "$BIN/llama-server" -m "$(first_split "$MODEL")" \
        --fit on -fitt "${FIT_MARGIN_MIB:-3072}" -fa on -c "$CTX" -t "$THREADS" \
        --jinja --temp 1.0 --top-p 0.95 \
        --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
    # thinking effort per request: "chat_template_kwargs": {"thinking_effort": "low"|"high"|"max"}
}

"${1:?usage: $0 build|download|convert|imatrix|quantize|serve|fetch}"
