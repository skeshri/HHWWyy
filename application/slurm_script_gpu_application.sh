#! /bin/bash

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
#SBATCH --job-name=HHWWgg

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/application/logSlurm_%u_%x_%j.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################

# Replace the following lines with your real workload
########################################
cd /hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/application/
source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
export QT_QPA_PLATFORM=offscreen
# Reference: https://stackoverflow.com/a/55210689/2302094
export XDG_RUNTIME_DIR=/hpcfs/bes/mlgpu/sharma/ML_GPU/someRuntimeFix
date
# dirTag="March18_LargStatBkgOnly_v2"
#time(python train-BinaryDNN.py -t 1 -s March17_v2)
#time(python train-BinaryDNN.py -t 1 -s ${dirTag})
#time(python -m json.tool HHWWyyDNN_binary_${dirTag}_BalanceYields/model_serialised.json > HHWWyyDNN_binary_${dirTag}_BalanceYields/model_serialised_nice.json)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_10March_SHiggs_BalanceYields -p HHWWgg DiPhoton QCD_Pt_30to40 QCD_Pt_40toInf GJet_Pt_20to40 GJet_Pt_40toInf TTGG TTGJets TTJets DYJetsToLL_M50 WW_TuneCP5_13TeV ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_March24_ManyVarsAngularVars_QCD_v1_BalanceYields -p HHWWgg DiPhoton Data TTGG TTGJets TTJets DYJetsToLL_M50 WW_TuneCP5_13TeV ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_March18_LargStatBkgOnly_v4_BalanceYields -p HHWWgg DiPhoton)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_March18_LargStatBkgOnly_v4_BalanceYields -p Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_March18_LargStatBkgOnly_v4_BalanceYields -p ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV )

# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithoutQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithoutQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithoutQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithQCD_BalanceYields -p HHWWgg DiPhoton Data TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)


# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithoutQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithoutQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithoutQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithQCD_BalanceYields -p  TTGJets TTGG ttHJetToGG_M125_13TeV VBFHToGG_M125_13TeV GluGluHToGG_M125_TuneCP5_13TeV VHToGG_M125_13TeV datadrivenQCD)

# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithoutQCD_BalanceYields -p  QCD Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithQCD_BalanceYields -p  QCD Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithoutQCD_BalanceYields -p  QCD Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithQCD_BalanceYields -p  QCD Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithoutQCD_BalanceYields -p  QCD Data)
# time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithQCD_BalanceYields -p  QCD Data)


#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithoutQCD_BalanceYields -p  HHZZgg HHWWgg)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWgg_WithQCD_BalanceYields -p  HHZZgg HHWWgg)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithoutQCD_BalanceYields -p  HHZZgg HHWWgg)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_WWZZgg_WithQCD_BalanceYields -p  HHZZgg HHWWgg)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithoutQCD_BalanceYields -p  HHZZgg HHWWgg)
#time(python run_network_evaluation.py -d HHWWyyDNN_binary_April2_ZZgg_WithQCD_BalanceYields -p  HHZZgg HHWWgg)

time(python run_network_evaluation.py -d HHWWyyDNN_HH_QCD_DiPho_E200_v1_BalanceYields -path /hpcfs/bes/mlgpu/sharma/ML_GPU/MultiClassifier/MultiClassifier -p HHWWgg HHZZgg QCD Data DiPhoton  ttHJetToGG_M125_13TeV  VBFHToGG_M125_13TeV  GluGluHToGG_M125_TuneCP5_13TeV   VHToGG_M125_13TeV )
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
