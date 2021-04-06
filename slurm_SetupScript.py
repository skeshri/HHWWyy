# -*- coding: utf-8 -*-
# @Author: Ram Krishna Sharma
# @Date:   2021-04-06 12:05:34
# @Last Modified by:   Ram Krishna Sharma
# @Last Modified time: 2021-04-06 13:02:03

##
## USER MODIFIED STRING
##

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dirTag', dest='dirTag', help='name of directory tag', default="TEST_args", type=str)
parser.add_argument('-j', '--jobName', dest='jobName', help='Slurm job name', default="DNN", type=str)
# parser.add_argument('-l', '--load_dataset', dest='load_dataset', help='Option to load dataset from root file (0=False, 1=True)', default=0, type=int)
# parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=1, type=int)
# parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
# parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing directories \'Bkgs\' and \'Signal\' which contain background and signal ntuples respectively.', default='', type=str)
# parser.add_argument('-m', '--model', dest='model', help='model to train with', default='Nadam', type=str) # Adam, Nadam, Adamax, Adadelta, Adagrad
# parser.add_argument('-p', '--para', dest='hyp_param_scan', help='Option to run hyper-parameter scan', default=0, type=int)
# parser.add_argument('-g', '--GridSearch', dest='GridSearch', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=0, type=int)
# parser.add_argument('-r', '--RandomSearch', dest='RandomSearch', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=1, type=int)

args = parser.parse_args()

dirTag=args.dirTag
LogDirPath = "/hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/HHWWyyDNN_binary_"+dirTag+"_BalanceYields/"

CommandToRun = "python train-BinaryDNN.py -t 1 -s "+dirTag
# CommandToRun = "python train-BinaryDNN.py -t 1 -s "+dirTag+" -p 1 -g 0 -r 1"

#===================================================================
import os
def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)
        os.system("cp dnn_parameter.json "+dir)

check_dir(LogDirPath)

from datetime import datetime
CURRENT_DATETIME = datetime.now()

DateTimeString = (str(CURRENT_DATETIME.year)[-2:]
              +str(format(CURRENT_DATETIME.month,'02d'))
              +str(format(CURRENT_DATETIME.day,'02d'))
              +"_"
              +str(format(CURRENT_DATETIME.hour,'02d'))
              +str(format(CURRENT_DATETIME.minute,'02d'))
              +str(format(CURRENT_DATETIME.second,'02d'))
              )
SlurmScriptName = "slurm_submit_"+DateTimeString+".sh"

SlurmJobName = args.jobName+"_"+DateTimeString

print("Name of slurm script: %s"%(SlurmScriptName))

LimitOutTextFile = open(SlurmScriptName, "w")
message = """#! /bin/bash

######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=normal

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=%s

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=%s/logSlurm_%%u_%%x_%%j.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################

# Replace the following lines with your real workload
########################################
cd /hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/
source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
# pip install shap
export QT_QPA_PLATFORM=offscreen
# Reference: https://stackoverflow.com/a/55210689/2302094
export XDG_RUNTIME_DIR=/hpcfs/bes/mlgpu/sharma/ML_GPU/someRuntimeFix
date
time(%s)
time(python -m json.tool HHWWyyDNN_binary_%s_BalanceYields/model_serialised.json > HHWWyyDNN_binary_%s_BalanceYields/model_serialised_nice.json)
date
##########################################
# Work load end

# Do not remove below this line

# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

sleep 180

"""

message = message %(SlurmJobName,LogDirPath,CommandToRun,dirTag,dirTag)

LimitOutTextFile.write(message+"\n")

LimitOutTextFile.close()
os.system('cp '+SlurmScriptName+' '+LogDirPath)
print("Log directory name: %s"%LogDirPath)
print("Run slurm script using:")
print("\tsbatch %s"%SlurmScriptName)