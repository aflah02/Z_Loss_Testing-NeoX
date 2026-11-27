#!/usr/bin/env bash
set -euo pipefail

suite=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$suite/../.." && pwd)
if [[ "${1:-}" == "--list" ]]; then
    exec python3 "$suite/matrix.py" "$@"
fi
mkdir -p "$suite/runs"
results=$(mktemp -d "$suite/runs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
git -C "$repo" rev-parse HEAD > "$results/commit.txt"
git -C "$repo" diff HEAD -- megatron > "$results/megatron.diff"
echo "Results: $results"
if [[ $(docker info --format '{{json .SecurityOptions}}') == *'"name=rootless"'* ]]; then
    user=0:0
else
    user="$(id -u):$(id -g)"
fi
docker run --rm --network none --shm-size=1g --user "$user" \
    --gpus "\"device=${GPU_DEVICES:-0,1}\"" \
    --volume "$repo:/checkout:ro" --volume "$results:/results" \
    --env Z_LOSS_RESULTS=/results --env HOME=/tmp --env OMP_NUM_THREADS=1 \
    --env PYTHONDONTWRITEBYTECODE=1 --env NCCL_SOCKET_IFNAME=lo --env GLOO_SOCKET_IFNAME=lo \
    --env NCCL_SOCKET_FAMILY=AF_INET --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 \
    --env NCCL_IB_DISABLE=1 --env NCCL_CUMEM_ENABLE=0 \
    --entrypoint bash "${GPT_NEOX_IMAGE:-gpt-neox:te}" \
    /checkout/logs/z_loss_testing/run_in_container.sh "$@"
