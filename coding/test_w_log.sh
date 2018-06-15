#!/bin/bash
# Use as: bash test_w_log.sh test_file.py

dt=$(date '+%Y%m%d_%H%M_');
gt=$(git rev-parse --short HEAD);
ending=".py"

python3.6 --version
python3.6 $1 > "${1/$ending/}_$dt$gt.log"
