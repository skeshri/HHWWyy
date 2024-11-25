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

## Training

```bash
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source xzz2l2nu_env/bin/activate
ulimit -s unlimited
python train-BinaryDNN_WWvsBB.py -t 1 -i /eos/user/a/avijay/HZZ_mergedrootfiles/
```


# Submit the condor jobs

```bash
# test
python prepare_condor_jobs.py --job_name "test_workday" --max_events 1000 --job_flavour "workday" --json "input_variables_test.json"

# final
python prepare_condor_jobs.py --job_name "Final_25Nov_tomorrow" --max_events -1 --job_flavour "tomorrow"
```
