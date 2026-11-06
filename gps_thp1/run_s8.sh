#!/bin/bash
export PATH=${CONDA_BIN}:$PATH; cd ${GPSRCL_DIR}; mkdir -p exp2
for s in 6 7 8 9 0; do python s8_more.py --mode train_freeze --lr 0.001 --seed $s > exp2/freeze_$s.log 2>&1 & done; wait
for s in 6 7 8 9 0; do python s8_more.py --mode train_l2sp --lr 0.0001 --lam 0.001 --seed $s > exp2/l2sp_$s.log 2>&1 & done; wait
python s8_more.py --mode evaluate > exp2/eval.log 2>&1
echo "=== S8 DONE ==="
