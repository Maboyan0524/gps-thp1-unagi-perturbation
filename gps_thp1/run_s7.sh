#!/bin/bash
export PATH=${CONDA_BIN}:$PATH; cd ${GPSRCL_DIR}; mkdir -p exp
for m in "ft_4cl 0.001" "ft_mcf7 0.001" "scratch 0.001"; do set -- $m
  for s in 6 7 8 9 0; do python s7_es.py --method $1 --lr $2 --seed $s --n_epoch 15 > exp/ES_$1_$2_$s.log 2>&1 & done; wait
done
echo "=== S7 DONE ==="
