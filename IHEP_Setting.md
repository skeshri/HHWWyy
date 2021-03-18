IHEP official doc about slurm: http://afsapply.ihep.ac.cn/cchelp/en/local-cluster/jobs/slurm/#3222-usage-of-the-slum-gpu-cluster

# Get ML environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
```

# Work on IHEP GUP

```bash
cd /hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy
# To submit job
sbatch slurm_sample_script_gpu_example.py 
# Check status
squeue -u sharma
```


# Fix some errors

## Error of matplotlib

Error:
```
Matplotlib created a temporary config/cache directory at /tmp/matplotlib-arj6o0s7 because the 
default path (/afs/ihep.ac.cn/users/s/sharma/.config/matplotlib) is not a writable directory; 
it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, 
in particular to speed up the import of Matplotlib and to better support multiprocessing.
```

Solution:
Add following line at the begining of main python script:
```python
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
```


## QT Error

Error:
```
Thu Mar 18 00:47:34 CST 2021
2021-03-18 00:47:41.356066: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
qt.qpa.xcb: could not connect to display localhost:19.0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, webgl, xcb.

 *** Break *** abort
 ```

Solution:
In slurm script add following line after loading the ML IHEP environment:
```
export QT_QPA_PLATFORM=offscreen
```
