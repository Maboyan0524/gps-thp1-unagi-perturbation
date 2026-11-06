#!/bin/bash
export PATH=${CONDA_BIN}:$PATH; cd ${GPSRCL_DIR}; mkdir -p exp
python s6_exp.py --method gps_mcf7 > exp/gps_mcf7.log 2>&1
python s6_exp.py --method gps_median4 > exp/gps_median4.log 2>&1
for m in "ft_4cl 0.001" "ft_mcf7 0.001" "ft_mcf7 0.0001" "scratch 0.001"; do set -- $m
  for s in 6 7 8 9 0; do python s6_exp.py --method $1 --lr $2 --seed $s > exp/$1_$2_$s.log 2>&1 & done; wait
done
echo "=== S6 DONE ==="
