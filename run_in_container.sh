#!/usr/bin/env bash
set -euo pipefail

suite=/checkout/logs/z_loss_testing
work=$(mktemp -d /tmp/z-loss-XXXXXX)
cd "$work"
# NeoX builds its dataset helper beside the Python module.
cp -a /checkout/megatron /checkout/train.py .
ln -s /checkout/{configs,eval_tasks,tests,tools} .
# NeoX reads the source revision when importing its argument definitions.
if [[ -d /checkout/.git ]]; then
    export GIT_DIR=/checkout/.git GIT_WORK_TREE=/checkout
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/checkout
fi
mkdir -p /tmp/.triton/autotune
export PYTHONPATH="$work:/checkout"
if (($# == 0)); then
    python -m pytest -c tests/pytest.ini --rootdir="$work" -p no:cacheprovider \
        -q --tb=short tests/model/test_z_loss.py \
        --junitxml="$Z_LOSS_RESULTS/z_loss_unit.xml" | tee "$Z_LOSS_RESULTS/z_loss_unit.log"
fi
python "$suite/matrix.py" "$@"
