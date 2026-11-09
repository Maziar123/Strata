# lighttransport/Strata: GLM-5.3-Flash host rules

The files listed below come from https://github.com/lighttransport/Strata (branch `glm53f`, commit
8fc37705899a9bf1604506fd1c3813874f80f0fa), a Strata fork that adds a GLM-5.3-Flash runner.

Copyright (c) 2026 Niko1221 and the Strata contributors, and the lighttransport/Strata contributors.
MIT License, the same text as the `LICENSE` at the root of this repository. The notice stays with every copy.

Imported unmodified (stage 1, the engine-independent host rules and their tests):

- `src/host/glm_*.c`, `src/host/glm_mixed_test.c`, `src/host/glm_mixed_transport_test.cpp`
- `include/strata/host/glm_*.h`, `include/strata/host/glm_names.hpp`
- `include/strata/core/{ram_executable,glm_mixed,exact_expert_cache}.hpp`
- `include/strata/spec/{bounded_accept,host_sampler}.hpp`, `include/strata/prefill/partition.hpp`
- `include/strata/kernels/cpu/glm_expert_cache.hpp`
- `src/program/{glm_prefix_cache.hpp,glm_state_store.hpp,glm_state_store.cpp}`
- `tests/host/glm_*`, `tests/glm/fp4_worker_protocol_test.c`
- `tools/glm_batch_route_sim.cpp`, `tools/test_glm_exact_cache.cpp`

Imported unmodified (stage 2a, the CPU FP4 expert path and the checkpoint tools):

- `src/kernels/cpu/glm_fp4.cpp`, `include/strata/kernels/cpu/{glm_fp4,expert_observer,llvq_expert,glm_fp4_exact}.hpp`,
  `include/strata/kernels/cpu/exl3/{math,pool,mcg_avx2}.hpp`, `include/strata/net/glm_fp4_worker.hpp`
- `src/program/glm_fp4_cpu_worker.cpp`, `tools/glm_fp4_cpu_bench.cpp`, `tools/glm_fp4_exact_bench.cpp`
- `tests/host/glm_fp4_cpu_test.cpp`, `tests/glm/{fp4_idle_test,fp4_worker_client_test}.cpp`
- `tools/ucomm/` (the TCP/RDMA transport the worker uses)
- `include/strata/core/model.hpp`, `include/strata/artifact/{glm_exl3,glm_modelopt,json,safetensors,glm_fp4_partition,fp8_experts,glm_native}.hpp`
- `src/artifact/{model_inspect,model_test}.cpp`, `tools/{glm_assemble,glm_fp4_check,glm_prune_experts}.cpp`

Imported as lighttransport's version of a file this repository already had (the fork had not changed it since the shared merge
base, so nothing is lost): `include/strata/artifact/{gguf_reader,model_family}.hpp`,
`src/artifact/{gguf_reader.cpp,gguf_reader_test.cpp}`, `tests/core/{gguf_fixture.hpp,model_family_test.cpp}`. They add
MiniMax-M3 family recognition, the IQ1_S/I16/I32/I64/F64 GGUF types and a call into `strata_host` for block names.

The CMake wiring (`STRATA_GLM_HOST`, `strata_host`, the `glm` test label) is written for this repository. Every other file listed
here is byte-identical to the upstream commit above.
