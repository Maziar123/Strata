# Kimi K3.1 Pro (llama.cpp pipeline)

**Strata's own engine cannot run Kimi.** It loads only the `qwen4exp`, `qwen35moe` and `deepseek41` GGUF architectures, and `check_architecture` in `include/strata/artifact/gguf_reader.hpp` refuses anything else. This folder adds a separate, self-contained path that builds [llama.cpp](https://github.com/ggml-org/llama.cpp), converts Kimi-K3.1-Pro and serves it. Nothing here changes the Strata engine, `strata launch` or the apps.

The only Kimi support inside Strata is the hosted `moonshotai` provider in `apps/host/server/model-providers.mjs`, which calls `https://api.moonshot.ai/v1`.

## Upstream llama.cpp now supports K3.1

[ggml-org/llama.cpp#31779](https://github.com/ggml-org/llama.cpp/pull/31779) (merged 2026-11-06, commit `bcb0152876e8f0b684ba70d3defcab6cdf35adb5`) adds the `kimi-k31` architecture, the DSA lightning-indexer graph, the converter and the quant protections. The script now builds that commit and needs no patch.

**GGUF compatibility.** The legacy patch writes `general.architecture = kimi-k3` plus indexer keys; upstream writes `kimi-k31`. Files from one do not load in the other, so anything converted with the patch must be converted again. `imatrix`, `quantize` and `serve` check the architecture of the first shard against the built llama.cpp and stop with a message on a mismatch.

Text only. The vision projector is not upstream.

## Files

| File | Purpose |
| --- | --- |
| `tools/kimi-k31/kimi-k31-local.sh` | Build, download, convert, imatrix, quantize, fetch and serve in one script |
| `tools/kimi-k31/0001-model-add-Kimi-K3.1-DSA-lightning-indexer.patch` | Legacy: llama.cpp patch for `669204ee`, arch `kimi-k3`. Applied only if `LLAMA_REF` points at a tree without `src/models/kimi-k31.cpp` |

The patch is stored byte-exact (`-text` in `.gitattributes`) so `git am` keeps working. Its author is credited in the patch header.

## Use

```bash
tools/kimi-k31/kimi-k31-local.sh build      # clone llama.cpp at LLAMA_REF, CUDA build for sm_75
tools/kimi-k31/kimi-k31-local.sh download   # moonshotai/Kimi-K3.1-Pro, 1.56 TB
tools/kimi-k31/kimi-k31-local.sh convert    # MXFP4 experts + BF16 rest, ~1.5 TB
CALIB=calib.txt tools/kimi-k31/kimi-k31-local.sh imatrix
tools/kimi-k31/kimi-k31-local.sh quantize   # IQ1_S, ~595 GB
tools/kimi-k31/kimi-k31-local.sh serve      # OpenAI API on :8080
```

Shortcut with a ready-made GGUF (skips download, convert, imatrix and quantize). `fetch` downloads one quant set only, plus the README, LICENSE and `SHA256SUMS`, then verifies the files it got:

```bash
GGUF_REPO=KarLexAI/Kimi-K3.1-Pro-GGUF QTYPE=IQ1_S tools/kimi-k31/kimi-k31-local.sh fetch
MODEL=<first shard printed by fetch> tools/kimi-k31/kimi-k31-local.sh serve
```

`QTYPE` is matched as a substring of the file path, so `IQ1_S` selects `KLX-IQ1_S/` and not `KLX-IQ1_M/`.

Paths and sizes are environment variables (`WORK`, `HF_DIR`, `GGUF_DIR`, `QTYPE`, `CTX`, `THREADS`, `LLAMA_REF`, `MODEL`, `GGUF_REPO`); read the script header.

## Models that can be used

Not downloaded or run. HuggingFace was unreachable from the machine that wrote this, so the list comes from a third-party mirror of model cards (`686f6c61/radar-huggingface-dataset`, machine-written summaries) and from upstream PR discussion. Check each card before downloading. For any third-party GGUF, `general.architecture` must be `kimi-k31` and it must have been made with llama.cpp at or after `bcb01528`.

| Model | What | Fit for this box |
| --- | --- | --- |
| `moonshotai/Kimi-K3.1-Pro` | Official checkpoint, 1.56 TB, 96 shards. Input to `convert` | Needs ~3 TB of disk for the full pipeline |
| `midagedev/Kimi-K3.1-Pro-GGUF` | IQ1_S, ~580 GB, 31 shards; reportedly `kimi-k31` | Fits 240 GB VRAM + 512 GB RAM. Use with `fetch` + `serve` |
| `KarLexAI/Kimi-K3.1-Pro-GGUF` | Three sets: `KLX-IQ1_S/`, `KLX-IQ1_M/`, `KLX-IQ2_XXS/`, plus `README.md`, `BENCHMARKS.md`, `LICENSE`, `SHA256SUMS` (layout from the user; file sizes, shard names and architecture string not seen) | Start with `QTYPE=IQ1_S`. IQ2_XXS is likely above 700 GB (rough estimate from bits per weight), too tight for 752 GB with a KV cache; check the sizes on the card first |
| `DevQuasar/moonshotai.Kimi-K3.1-Pro-GGUF` | Marked experimental; made with a pre-upstream fork branch | Architecture uncertain, check before use |
| `akopytko/tiny-kimi-k3.1-pro-GGUF` | Tiny random model | Smoke test of the whole serve path on any machine |
| `tiny-random/kimi-k3.1-pro` | 144 MB random safetensors | Smoke test for `convert` |
| `daocloud/Kimi-K3.1-Pro-pruned75` | 224 of 896 experts, MXFP4, 96 shards | Smaller input for `convert`; quality unknown |

Not usable here:

- `unsloth/Kimi-K3-GGUF` and other K3 GGUFs (`kimi-k3` architecture, not K3.1).
- NVFP4 and FP8 builds (`nvidia/Kimi-K3-NVFP4`, `RedHatAI/...`): for vLLM on GPUs with FP4 support; Turing has none.
- MLX builds: Apple only.
- DSpark and Dflash drafts (`Inferact/Kimi-K3.1-Pro-DSpark-preview`, ...): vLLM speculative decoding. llama.cpp support was not checked.

## Requirements

The script targets this box: Arch Linux (`pacman`), 5 NVIDIA Turing GPUs (sm_75, about 48 GB each), 512 GB RAM and about 1.6 TB of disk (the HF download can be deleted after `convert`). Other systems need the dependency and `-DCMAKE_CUDA_ARCHITECTURES` lines in `build()` adapted.

## GPUs: sm_75 and several cards

The build passes `-DCMAKE_CUDA_ARCHITECTURES=75` (Turing: RTX 20, Quadro RTX, Titan RTX). `serve` uses llama.cpp's `--fit on`, which places
layers and experts across every visible GPU and the rest in RAM, keeping `FIT_MARGIN_MIB` (default 3072) free on each card. Restrict the
cards with `CUDA_VISIBLE_DEVICES`. For another generation change the architecture in `build()` (86 for RTX 30, 89 for RTX 40, 70 for V100).
To place experts by hand use `--fit off -ngl 99 --n-cpu-moe N` and lower `N` until VRAM is full (see the comment in `serve()`).

## Pinned llama.cpp commit

`LLAMA_REF` defaults to `bcb0152876e8f0b684ba70d3defcab6cdf35adb5`, the merge of #31779. The legacy patch needs `669204ee2b9004845b34f26466a91d003edc98aa` and does not apply on `bcb01528` (both touch the same files); to use it, set `LLAMA_REF=669204ee2b9004845b34f26466a91d003edc98aa`, which gives `kimi-k3` GGUFs.

## Notes

- `KIMI_K31_DENSE=1` at convert time applies to the legacy patch only (drops the indexer, plain `kimi-k3` GGUF).
- `llama-quant` keeps the indexer `attn_q_b`, `attn_k` and `proj` tensors and `attn_k_b` / `attn_v_b` unquantized.
- Thinking effort at serve time: `"chat_template_kwargs": {"thinking_effort": "low" | "high" | "max"}`.
- IQ1 and IQ2 types need an importance matrix, so `imatrix` must run first.

## Verified and not verified

Checked on a CPU-only machine:

- llama.cpp `bcb01528` builds on CPU, and its `test-llama-archs` passes for `kimi-k3` and `kimi-k31` (random tiny weights; NMSE vs. CPU, state roundtrip, mixed batch). The test runs on CPU only; the GPU rows were skipped.
- The script passes `bash -n`. The architecture guard gives the expected result for all four combinations of (upstream or legacy tree) and (`kimi-k31` or `kimi-k3` GGUF). `fetch` was run against a stub `hf` that imitates the KarLexAI layout: only the chosen set is downloaded, checksums of the other sets are skipped, and a corrupted shard makes it fail.
- The legacy patch applies at `669204ee` with `git am`.

Not run: the CUDA build, any download, the conversion, quantization and serving, and `fetch` against a real repo. They need the hardware and network above, so output quality, speed and memory for Kimi-K3.1-Pro are unmeasured.
