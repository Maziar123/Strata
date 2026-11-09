# Kimi K3.1 Pro (llama.cpp pipeline)

**Strata's own engine cannot run Kimi.** It loads only the `qwen4exp`, `qwen35moe` and `deepseek41` GGUF architectures, and `check_architecture` in `include/strata/artifact/gguf_reader.hpp` refuses anything else. This folder adds a separate, self-contained path that builds a patched [llama.cpp](https://github.com/ggml-org/llama.cpp), converts Kimi-K3.1-Pro and serves it. Nothing here changes the Strata engine, `strata launch` or the apps.

The only Kimi support inside Strata is the hosted `moonshotai` provider in `apps/host/server/model-providers.mjs`, which calls `https://api.moonshot.ai/v1`.

## Files

| File | Purpose |
| --- | --- |
| `tools/kimi-k31/kimi-k31-local.sh` | Build, download, convert, imatrix, quantize and serve in one script |
| `tools/kimi-k31/0001-model-add-Kimi-K3.1-DSA-lightning-indexer.patch` | llama.cpp patch: Kimi-K3.1 DSA lightning indexer on top of `kimi-k3` |

The patch is stored byte-exact (`-text` in `.gitattributes`) so `git am` keeps working. Its author is credited in the patch header.

## Use

```bash
tools/kimi-k31/kimi-k31-local.sh build      # clone llama.cpp, pin, apply patch, CUDA build for sm_75
tools/kimi-k31/kimi-k31-local.sh download   # moonshotai/Kimi-K3.1-Pro, 1.56 TB
tools/kimi-k31/kimi-k31-local.sh convert    # MXFP4 experts + BF16 rest, ~1.5 TB
CALIB=calib.txt tools/kimi-k31/kimi-k31-local.sh imatrix
tools/kimi-k31/kimi-k31-local.sh quantize   # IQ1_S, ~595 GB
tools/kimi-k31/kimi-k31-local.sh serve      # OpenAI API on :8080
```

Paths and sizes are environment variables (`WORK`, `HF_DIR`, `GGUF_DIR`, `QTYPE`, `CTX`, `THREADS`, `LLAMA_REF`); read the script header.

## Requirements

The script targets this box: Arch Linux (`pacman`), 5 NVIDIA Turing GPUs (sm_75, about 48 GB each), 512 GB RAM and about 1.6 TB of disk (the HF download can be deleted after `convert`). Other systems need the dependency and `-DCMAKE_CUDA_ARCHITECTURES` lines in `build()` adapted.

## Pinned llama.cpp commit

The patch applies at llama.cpp `669204ee2b9004845b34f26466a91d003edc98aa` (2026-11-06) and conflicts on later master in `conversion/__init__.py`, `conversion/base.py`, `conversion/kimi_k3.py`, `src/llama-model.cpp` and `src/llama-quant.cpp`. The script checks out that commit before `git am`. Override with `LLAMA_REF=<commit>` once the patch has been rebased.

## Notes

- `KIMI_K31_DENSE=1` at convert time drops the indexer and produces a plain `kimi-k3` GGUF with dense attention.
- `llama-quant` keeps the indexer tensors and `attn_k_b` / `attn_v_b` unquantized.
- Thinking effort at serve time: `"chat_template_kwargs": {"thinking_effort": "low" | "high" | "max"}`.
- IQ1 and IQ2 types need an importance matrix, so `imatrix` must run first.

## Verified and not verified

Checked on a CPU-only machine: the patch applies cleanly at the pinned commit with `git am`, the changed Python files compile, and the script passes `bash -n`.

Not run: the CUDA build, the download, the conversion, quantization and serving. They need the hardware above, so output quality, speed and memory for Kimi-K3.1-Pro are unmeasured.
