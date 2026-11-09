# GLM-5.3-Flash (work in progress)

**Status: not runnable yet.** This branch (`feat/glm53-flash`) holds stages 1 and 2a: the engine-independent C11 host rules, the
CPU FP4 expert kernel with its worker, and the checkpoint tools, all with tests that run on a CPU. There is no GLM decode
engine here yet. Strata's loader still refuses the GLM architecture (`check_architecture` in
`include/strata/artifact/gguf_reader.hpp`), and `strata launch`, `setup.py` and the apps do not list GLM.

## The model

GLM-5.3-Flash (zai-org, MIT): 321B parameters, about 18B active per token, 45 layers (3 KDA : 1 MLA-NoPE with DSA indexer),
mHC hyper-connections, 288 routed experts plus 1 shared (top-8), one MTP layer, vision, context up to 1M. llama.cpp calls the
architecture `glm5-next`.

## What is imported

From [lighttransport/Strata](https://github.com/lighttransport/Strata) (branch `glm53f`, MIT; see
`third_party/lighttransport-strata/NOTICE.md`).

Stage 1, the host rules:

- tensor names, the tuning-knob table, the FP4 expert pack, layout and worker protocol, the mixed-precision cache policy and the
  MTP draft rules (`src/host/glm_*.c`, `include/strata/host/glm_*.h`)
- header-only helpers: bounded accept, host sampler, expert cache, exact expert cache, prefix cache, state store, memfd-sealed
  executable pages

Stage 2a, the CPU path and the tools:

- `src/kernels/cpu/glm_fp4.cpp`: the CPU FP4 expert kernel (AVX2/AVX-512 and a scalar path), tested against a reference, plus
  `strata-glm-fp4-cpu-bench` and `strata-glm-fp4-exact-bench`
- `strata-glm-fp4-cpu-worker` and the `strata_ucomm` transport it talks over (Linux)
- `strata-model-inspect`, `strata-glm-fp4-check`, `strata-glm-assemble` and `strata-glm-prune-experts` for ModelOpt FP4/FP8 and
  assembled Q2/Q3 checkpoints
- lighttransport's `gguf_reader` and `model_family` (adds MiniMax-M3 recognition and more GGUF tensor types). They call
  `strata_host`, so that library is now always built and linked into `strata_artifact`; the rest of this page stays opt-in

Nothing here touches a GPU or loads a model. The tests were run on a CPU only (26 tests, label `glm`); the tools were built and
started, not run on a real checkpoint.

```bash
cmake -S . -B build -G Ninja -DSTRATA_GLM_HOST=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF
cmake --build build --target strata-glm-host-tests
ctest --test-dir build -L glm
```

`STRATA_GLM_HOST` is off by default and adds the GLM kernel, worker, tools and tests; a plain configure builds none of those.
Build the `strata-glm-host-tests` target rather than `all` when you only want these tests (`all` also builds every other Strata
test). The FP4 CPU tests take about a minute.

## Not imported yet

The lighttransport engine (`strata-glm-decode`, `strata-glm-gpu-worker`, `src/program/glm_*`, `src/kernels/cuda/glm*.cu`, the GLM
mode of `serve/server.py`) is about 6,600 files and 92 edits to shared files; 10 of those edits conflict with changes this
repository made since the shared merge base (`serve/server.py`, `serve/frontend.py`, `native_expert.cpp`, `prefill.cpp`,
`moe_mmq.cu`, `peer_experts.cpp` and four more). It needs CUDA or HIP and is not wired into `strata launch`. It follows in later
stages, each built and tested on a GPU machine, because the shared CUDA edits can change the DeepSeek and Qwen kernels. Its weights are `lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts`
(full Q23 about 121 GB, REAP-224 about 96 GB, REAP-50 about 63 GB); [Project Maya](../third_party/project-maya/NOTICE.md)'s
GGUF quants and Unsloth's `GLM-5.3-Flash-GGUF` are alternatives.

## GPUs: sm_75 and several cards

- Strata itself supports RTX 20 (Turing, sm_75) as the lowest release target; build with `-DCMAKE_CUDA_ARCHITECTURES=75`. See
  [Older GPUs](OLDER_GPUS.md) and [Multiple GPUs](MULTI_GPU.md). Volta sm_70 (V100) needs `-DSTRATA_EXPERIMENTAL_SM60=ON`.
- lighttransport measured GLM on 1-2 Tesla V100 (sm_70), an RTX 5060 Ti with an RX 9070 XT, and similar. It describes a
  `gpu_layout: "layer"` contiguous layer split across `gpu_devices`; its MTP pipeline needs exactly two GPUs. **No sm_75 run and
  no layout over more than two cards is documented**, so both are untested targets for the later GPU stages, not claims.

Nothing on this page is a Strata measurement; lighttransport's numbers are theirs and stay in their repository.
