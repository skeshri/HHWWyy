import os
import argparse

def create_sh_script(virtual_env, script_name, job_name, project_path, eos_path, input_path, max_events, json):
    """
    Creates the .sh file for running the job.
    """
    sh_content = f"""#!/bin/bash
ulimit -s unlimited
set -e

# Navigate to the project directory
cd {project_path}

# Activate pre-configured environment
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source {virtual_env}/bin/activate

# Run the training
name='{job_name}'
python train-BinaryDNN_WWvsBB.py -t 1 -i {input_path} -s ${{name}} --nEvents {max_events} -l 1 -j {json}

echo "Training Done"

# Copy the output to eos
echo "Copying the output to eos"
cp -r HHWWBBDNN_binary_${{name}}_BalanceYields {eos_path}
echo "Output copied to eos"
ls {eos_path}
echo "All Done"
"""
    with open(script_name, "w") as sh_file:
        sh_file.write(sh_content)
    os.chmod(script_name, 0o755)  # Make the script executable


def create_jdl_file(jdl_name, sh_script_name, request_gpus, job_flavour):
    """
    Creates the .jdl file for Condor job submission.
    """
    jdl_content = f"""executable              = {sh_script_name}
arguments               = $(ClusterId)$(ProcId)
output                  = train.$(ClusterId).$(ProcId).out
error                   = train.$(ClusterId).$(ProcId).err
log                     = train.$(ClusterId).log
+JobFlavour             = "{job_flavour}"
when_to_transfer_output = ON_EXIT
request_GPUs            = {request_gpus}
request_CPUs = {request_gpus}
queue
"""
    with open(jdl_name, "w") as jdl_file:
        jdl_file.write(jdl_content)


def main():
    parser = argparse.ArgumentParser(description="Generate Condor .sh and .jdl files for job submission.")

    # Default paths
    project_path = os.getcwd()
    default_input_path = "/eos/user/a/avijay/HZZ_mergedrootfiles/"
    default_eos_path = "/eos/user/r/rasharma/HZZ2l2nu/"
    default_job_name = "test"
    default_virtual_env_name = "xzz2l2nu_env"

    parser.add_argument("--input_path", default=default_input_path, help=f"Path to the input data (default: {default_input_path}).")
    parser.add_argument("--eos_path", default=default_eos_path, help=f"Path to the EOS directory for output (default: {default_eos_path}).")
    parser.add_argument("--job_name", default=default_job_name, help=f"Name of the job (default: {default_job_name}).")
    parser.add_argument("--max_events", type=int, default=1000, help="Maximum number of events to process. Use -1 for all events (default: 1000).")
    parser.add_argument("--request_gpus", type=int, default=1, help="Number of GPUs to request (default: 1).")
    parser.add_argument("--job_flavour", default="workday", help='Job flavour (default: "workday").')
    parser.add_argument("--virtual_env", default=default_virtual_env_name, help=f"Name of the virtual environment (default: {default_virtual_env_name}).")

    parser.add_argument('-j', '--json', dest='json', help='input variable json file', default='input_variables.json', type=str)


    args = parser.parse_args()

    # Create .sh and .jdl files
    sh_script_name = f"train_{args.job_name}.sh"
    jdl_file_name = f"train_{args.job_name}.jdl"
    create_sh_script(args.virtual_env, sh_script_name, args.job_name, project_path, args.eos_path, args.input_path, args.max_events, args.json)
    create_jdl_file(jdl_file_name, sh_script_name, args.request_gpus, args.job_flavour)

    print(f"Prepared {sh_script_name} and {jdl_file_name} for Condor job submission.")
    print(f"Default paths:\n  Project Path: {project_path}\n  EOS Path: {default_eos_path}\n  Input Path: {default_input_path}")

    # Submit the job
    print(f"condor_submit {jdl_file_name}")


if __name__ == "__main__":
    main()
