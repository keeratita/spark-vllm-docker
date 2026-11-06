#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/kv-cache-prealloc-cleanup/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

make_site_tree() {
    local root="$1"
    mkdir -p "$root/vllm/v1/worker" "$root/vllm/config"
    printf 'from vllm.config.cache import CacheConfig\n' > "$root/vllm/config/cache.py"
}

# ---------------------------------------------------------------------------
# 1. Current B12X fork layout: the estimate is natively gated through
#    cudagraph_memory_estimate_applied and the call spans multiple lines.
#    The mod must leave the file untouched and succeed.
# ---------------------------------------------------------------------------
FORK_ROOT="$TMP_DIR/fork"
make_site_tree "$FORK_ROOT"
FORK_WORKER="$FORK_ROOT/vllm/v1/worker/gpu_worker.py"
printf '%s\n' \
    'import gc' \
    'import vllm.envs as envs' \
    'from vllm.logger import init_logger' \
    '' \
    'logger = init_logger(__name__)' \
    '' \
    '' \
    'class Worker:' \
    '    def determine_available_memory(self):' \
    '        cudagraph_memory_estimate = 0' \
    '        if (' \
    '            current_platform.is_cuda_alike()' \
    '            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE' \
    '        ):' \
    '            try:' \
    '                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory(' \
    '                    self._prepare_b12x_profile_state' \
    '                )' \
    '            finally:' \
    '                self._release_b12x_profile_state()' \
    '        cudagraph_memory_estimate_applied = (' \
    '            cudagraph_memory_estimate' \
    '            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS' \
    '            else 0' \
    '        )' \
    '        return cudagraph_memory_estimate_applied' \
    > "$FORK_WORKER"
fork_before=$(sha256sum "$FORK_WORKER")

fork_output=$(PYTHON_ROOT="$FORK_ROOT" bash "$MOD")
grep -Fq 'already supported; skipping' <<< "$fork_output"
fork_after=$(sha256sum "$FORK_WORKER")
if [ "$fork_before" != "$fork_after" ]; then
    echo "mod modified a natively gated gpu_worker.py" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Legacy layout: single-line call, no native gate. The mod must guard the
#    call and remain idempotent.
# ---------------------------------------------------------------------------
LEGACY_ROOT="$TMP_DIR/legacy"
make_site_tree "$LEGACY_ROOT"
LEGACY_WORKER="$LEGACY_ROOT/vllm/v1/worker/gpu_worker.py"
printf '%s\n' \
    'import gc' \
    'import vllm.envs as envs' \
    'from vllm.logger import init_logger' \
    '' \
    'logger = init_logger(__name__)' \
    '' \
    '' \
    'class Worker:' \
    '    def determine_available_memory(self):' \
    '        cudagraph_memory_estimate = 0' \
    '        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:' \
    '            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()' \
    '        return cudagraph_memory_estimate' \
    > "$LEGACY_WORKER"

legacy_output=$(PYTHON_ROOT="$LEGACY_ROOT" bash "$MOD")
grep -Fq 'env-respectful' <<< "$legacy_output"
grep -Fq 'if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:' "$LEGACY_WORKER"
grep -Fq 'Skipping CUDA graph memory profiling' "$LEGACY_WORKER"
python3 -c "import ast; ast.parse(open('$LEGACY_WORKER').read())"

legacy_again=$(PYTHON_ROOT="$LEGACY_ROOT" bash "$MOD")
grep -Fq 'already supported; skipping' <<< "$legacy_again"

# ---------------------------------------------------------------------------
# 3. Ungated multi-line call: the mod must guard the whole statement, keep
#    continuation lines and the surrounding try/finally valid.
# ---------------------------------------------------------------------------
MULTI_ROOT="$TMP_DIR/multi"
make_site_tree "$MULTI_ROOT"
MULTI_WORKER="$MULTI_ROOT/vllm/v1/worker/gpu_worker.py"
printf '%s\n' \
    'import gc' \
    'import vllm.envs as envs' \
    'from vllm.logger import init_logger' \
    '' \
    'logger = init_logger(__name__)' \
    '' \
    '' \
    'class Worker:' \
    '    def determine_available_memory(self):' \
    '        cudagraph_memory_estimate = 0' \
    '        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:' \
    '            try:' \
    '                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory(' \
    '                    self._prepare_b12x_profile_state,' \
    '                )' \
    '            finally:' \
    '                self._release_b12x_profile_state()' \
    '        return cudagraph_memory_estimate' \
    > "$MULTI_WORKER"

multi_output=$(PYTHON_ROOT="$MULTI_ROOT" bash "$MOD")
grep -Fq 'env-respectful' <<< "$multi_output"
python3 - "$MULTI_WORKER" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text())
guards = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.If)
    and ast.unparse(node.test) == "envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
]
assert len(guards) == 1, guards
# The guarded body still passes the profiling-only argument through.
assert "_prepare_b12x_profile_state" in ast.unparse(guards[0].body[0])
# The else branch only logs the skip.
assert "info_once" in ast.unparse(guards[0].orelse[0])
PY

# ---------------------------------------------------------------------------
# 4. cache.py validator: the mod allows --gpu-memory-utilization-gb together
#    with --kv-cache-memory-bytes and is idempotent.
# ---------------------------------------------------------------------------
python3 - "$LEGACY_ROOT/vllm/config/cache.py" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    'class CacheConfig:\n'
    '    def _validate_memory_params(self) -> "CacheConfig":\n'
    '        if self.gpu_memory_utilization_gb is not None and self.kv_cache_memory_bytes is not None:\n'
    '            raise ValueError("Cannot specify both gpu_memory_utilization_gb and kv_cache_memory_bytes.")\n'
    '        return self\n'
)
PY

cache_output=$(PYTHON_ROOT="$LEGACY_ROOT" bash "$MOD")
grep -Fq 'Allowed --gpu-memory-utilization-gb' <<< "$cache_output"
grep -Fq 'allow fixed GiB reservation with manual KV cache' "$LEGACY_ROOT/vllm/config/cache.py"
python3 -c "import ast; ast.parse(open('$LEGACY_ROOT/vllm/config/cache.py').read())"

cache_again=$(PYTHON_ROOT="$LEGACY_ROOT" bash "$MOD")
grep -Fq 'Manual KV cache compatibility is already supported' <<< "$cache_again"

echo "kv-cache-prealloc-cleanup mod tests passed"
