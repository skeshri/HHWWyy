## Setup

```bash
python -m venv xzz2l2nu_env
source xzz2l2nu_env/bin/activate
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
pip install -r requirement.txt
# Run the training
ulimit -s unlimited
python train-BinaryDNN_WWvsBB.py -t 1 -i /eos/user/a/avijay/HZZ_mergedrootfiles/
```
