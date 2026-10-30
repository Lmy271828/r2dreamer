#!/usr/bin/env bash
# FirstLight 训练启动器：唯一 run 目录 + 标准覆盖 + 追加参数
# 用法: ./run_train.sh <label> [额外 hydra overrides...]
set -euo pipefail
LABEL="${1:?usage: run_train.sh <label> [overrides...]}"; shift
STAMP=$(date +%m%d_%H%M%S)
LOGDIR="runs/${STAMP}_${LABEL}"
if [ -e "$LOGDIR" ]; then
  echo "REFUSED: $LOGDIR already exists — never reuse a logdir" >&2
  exit 1
fi
echo "logdir: $LOGDIR"
# Python 解析：优先 $PYTHON，其次 FirstLight venv，最后 PATH 上的 python
PY="${PYTHON:-../FirstLight/FirstLight_CR/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python)"
export CR_T0_DIR="$PWD/$LOGDIR/t0"   # T0 writes into this run logdir
exec "$PY" -u train.py \
  env=firstlight_swarm \
  model.rep_loss=r2dreamer \
  model.compile=False \
  device=cuda:0 \
  logdir="$LOGDIR" "$@"
