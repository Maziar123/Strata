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

The CMake wiring (`STRATA_GLM_HOST`, `strata_host`, the `glm` test label) is written for this repository. The first commit
of each imported file is the unmodified upstream copy.
