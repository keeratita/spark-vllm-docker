#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/v1/worker/gpu_worker.py"
CACHE_CONFIG="$PYTHON_ROOT/vllm/config/cache.py"

if [ ! -f "$TARGET" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM gpu_worker.py not found at $TARGET" >&2
  exit 1
fi

if [ ! -f "$CACHE_CONFIG" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM cache.py not found at $CACHE_CONFIG" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[kv-cache-prealloc-cleanup] python3 is required to apply this mod." >&2
  exit 1
fi

python3 - "$TARGET" "$CACHE_CONFIG" <<'PY'
from pathlib import Path
import ast
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
lines = text.splitlines(keepends=True)

LOCAL_MARKER = "# spark-vllm-docker: skip CUDA graph memory profiling when disabled"

# A vLLM build is already able to skip the CUDA-graph memory charge when
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 if either:
#   - it gates the estimate before applying it to the KV budget
#     (cudagraph_memory_estimate_applied, current upstream and B12X forks), or
#   - a previous application of this mod inserted the skip guard, or
#   - the skip guard was adopted upstream with the same log line.
# Anything else needs the profiling call guarded locally.
native_skip = (
    "cudagraph_memory_estimate_applied" in text
    or "Skipping CUDA graph memory profiling" in text
    or LOCAL_MARKER in text
)

if native_skip:
    print(
        "[kv-cache-prealloc-cleanup] CUDA graph profiling skip is already "
        "supported; skipping."
    )
else:

    def find_line(pattern: str, message: str) -> tuple[int, re.Match[str]]:
        regex = re.compile(pattern)
        for index, line in enumerate(lines):
            match = regex.match(line)
            if match:
                return index, match
        raise SystemExit(f"[kv-cache-prealloc-cleanup] {message}")

    if not re.search(
        r"^(?:import vllm\.envs as envs|from vllm import .*\benvs\b.*)$",
        text,
        re.MULTILINE,
    ):
        raise SystemExit(
            "[kv-cache-prealloc-cleanup] gpu_worker.py does not import "
            "vllm.envs as envs; refusing to patch an unknown layout."
        )

    call_start, match = find_line(
        r"^(?P<indent>[ \t]+)cudagraph_memory_estimate = "
        r"self\.model_runner\.profile_cudagraph_memory\(",
        "Could not find the profile_cudagraph_memory() assignment in "
        "gpu_worker.py; vLLM layout is unsupported by this mod.",
    )
    indent = match.group("indent")

    # The assignment may be a single line or span continuation lines, e.g.
    #     cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory(
    #         self._prepare_b12x_profile_state
    #     )
    # Find the statement end by balancing parentheses across the call.
    call_end = call_start
    depth = lines[call_start].count("(") - lines[call_start].count(")")
    while depth > 0:
        call_end += 1
        if call_end >= len(lines):
            raise SystemExit(
                "[kv-cache-prealloc-cleanup] Unbalanced parentheses while "
                "locating the profile_cudagraph_memory() call."
            )
        depth += lines[call_end].count("(") - lines[call_end].count(")")

    guarded = [
        f"{indent}{LOCAL_MARKER}\n",
        f"{indent}if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:\n",
    ]
    for line in lines[call_start : call_end + 1]:
        guarded.append(f"{indent}    {line}" if line.strip() else line)
    guarded.extend(
        [
            f"{indent}else:\n",
            f"{indent}    logger.info_once(\n",
            f'{indent}        "Skipping CUDA graph memory profiling because "\n',
            f'{indent}        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0."\n',
            f"{indent}    )\n",
        ]
    )
    lines[call_start : call_end + 1] = guarded

    candidate = "".join(lines)
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        raise SystemExit(
            "[kv-cache-prealloc-cleanup] Guarding the "
            "profile_cudagraph_memory() call produced invalid Python: "
            f"{exc}. Refusing to patch."
        )
    path.write_text(candidate)
    print("[kv-cache-prealloc-cleanup] Made CUDA graph profiling skip env-respectful.")

cache_path = Path(sys.argv[2])
cache_text = cache_path.read_text()
cache_lines = cache_text.splitlines(keepends=True)
cache_changed = False

CACHE_MARKER = (
    "# spark-vllm-docker: allow fixed GiB reservation with manual KV cache"
)

if CACHE_MARKER not in cache_text and "Cannot specify both gpu_memory_utilization_gb" in cache_text:
    validator_pattern = re.compile(
        r"^(?P<indent>[ \t]+)def _validate_memory_params"
        r"\(self\) -> \"CacheConfig\":\n$"
    )
    for index, line in enumerate(cache_lines):
        match = validator_pattern.match(line)
        if match:
            body_indent = match.group("indent") + "    "
            cache_lines[index + 1:index + 1] = [
                f"{body_indent}# spark-vllm-docker: allow fixed GiB reservation with manual KV cache\n",
                f"{body_indent}return self\n",
                "\n",
            ]
            cache_changed = True
            break
    else:
        raise SystemExit(
            "[kv-cache-prealloc-cleanup] Found the memory-parameter "
            "conflict validator in cache.py, but could not patch it."
        )

if cache_changed:
    candidate = "".join(cache_lines)
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        raise SystemExit(
            "[kv-cache-prealloc-cleanup] Patching cache.py produced invalid "
            f"Python: {exc}. Refusing to patch."
        )
    cache_path.write_text(candidate)
    print(
        "[kv-cache-prealloc-cleanup] Allowed --gpu-memory-utilization-gb "
        "with --kv-cache-memory-bytes."
    )
else:
    print(
        "[kv-cache-prealloc-cleanup] Manual KV cache compatibility is "
        "already supported; skipping."
    )
PY

echo "=====> vLLM can skip CUDA graph profiling when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
echo "=====> vLLM can combine --gpu-memory-utilization-gb with --kv-cache-memory-bytes"
