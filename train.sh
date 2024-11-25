#!/bin/bash
ulimit -s unlimited
set -e
cd /afs/cern.ch/user/r/rasharma/work/h2l2nu/ML/HHWWyy
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python -m venv xzz2l2nu_env
source xzz2l2nu_env/bin/activate
pip install -r requirement.txt
# Run the training
python train-BinaryDNN_WWvsBB.py -t 1 -i /eos/user/a/avijay/HZZ_mergedrootfiles/

echo "Training Done"

# Copy the output to eos
echo "Copying the output to eos"
cp -r HHWWBBDNN_binary_TEST_BalanceYields /eos/user/r/rasharma/HZZ2l2nu/
echo "Output copied to eos"
ls /eos/user/r/rasharma/HZZ2l2nu/
echo "All Done"
