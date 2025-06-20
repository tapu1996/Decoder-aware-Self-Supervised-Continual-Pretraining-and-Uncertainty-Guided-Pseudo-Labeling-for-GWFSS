"""

Author : Sebastien Quetin

This script only construct the bash script that will execute the training.

"""

import argparse
import os
import subprocess

import math

def to_python_args(dic):
    string_for_python_cmd = ""
    for key, value in dic.items():
        string_for_python_cmd = string_for_python_cmd + f" --{key}={value}"
    return string_for_python_cmd


# type is callable with value as argument, return corresponding argument as the function return
def str2bool(
    v,
):  # we could also use actions=store_true or store_false in arg pars but as I astarted like ximeng I don't want to confuse myself
    return v.lower() in ("true")


def GetArguments():
    parser = argparse.ArgumentParser(description="Arguments for SSL training")
    # For running code
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=512,
        help="Full batch size used for slotcon.",
    )
    parser.add_argument('--use-decoder', action='store_true', help='use decoder or not')
    parser.add_argument('--group-loss-weight-dec', default=0.5, type=float, help='balancing weight of the grouping loss for decoder')
    parser.add_argument('--encoder-loss-weight', default=0.5, type=float, help='balancing weight of the encoder loss when there is a decoder')
    parser.add_argument(
        '--decoder-downstream-dataset', 
        type=str, 
        default="cityscapes", 
        help='dataset used in downstream task, it influences the config of the decoder'
    )
    parser.add_argument(
        '--decoder-type', 
        type=str, 
        default="FCN", 
        help='config for decoder for now only supports FCN|FPN.'
    )

    parser.add_argument(
        "--existing-env",
        action='store_true', 
        help='load an existing env? otherwise will re- pip install every library.'
        )
    parser.add_argument(
        "-e",
        "--exp",
        type=str,
        required=True,
        help="Experiment name for output directory for the training",
    )

    # For code user 
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default="seb",
        help="Who is running? seb, tapo??",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="slotcon",
        help="Which task are you running? slotcon, detection",
    )

    parser.add_argument(
        "--enc-path",
        type=str,
        default="",
        help="Path to the encoder model for detection task",
    )

    # For Compute Canada resources
    parser.add_argument("-d", "--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of gpus")
    parser.add_argument("-t", "--hours", type=int, default=5, help="Time (hours)")
    parser.add_argument("-m", "--minutes", type=int, default=0, help="Time (minutes)")

    return parser.parse_args()


def submit_slurm(args):

    # Run the hostname command and capture the output. 
    # Should be something like narval3.narval.calcul.quebec, cedar2.int.cedar.computecanada.ca
    hostname = subprocess.check_output("hostname", shell=True).decode("utf-8").strip()
    # We remove the login node digit
    cluster = hostname.split(".")[0].translate(str.maketrans('', '', '0123456789'))

    if args.user == "seb":
        if cluster == "narval":
            data_folder = "/home/sebquet/projects/rrg-senger-ab/sebquet/VisionResearchLab/DenseSSL/Data/"
            env_path = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/densesslenv"
        else:
            assert cluster == "cedar", f"code and data not set on another cluster than narval or cedar but you have {cluster}"
            data_folder = "/home/sebquet/projects/def-senger/sebquet/VisionResearchLab/DenseSSL/Data/"
            env_path = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/densesslenv"
        # folder contianing DenseSSL repo, for me I have VRLab/DenseSSL/DenseSSL/Slotcon-main
        code_folder = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/"
        account = "def-senger"
        email = "sebastien.quetin@mail.mcgill.ca"
    else:
        assert args.user == "tapo", "author should be either seb or tapo"
        account = "def-farma"
        data_folder = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/Data"
        code_folder = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/"
        env_path = None # To define
        email = "tapotosh.ghosh@ucalgary.ca"
        
    if args.task == "slotcon":
        key_dir = "slotcon_"
    else:
        assert args.task == "detection", "task should be either slotcon or detection"
        key_dir = "detection_"
    bash_save_dir = os.path.join(
        os.path.dirname(code_folder), "output", key_dir+args.exp
    )

    os.makedirs(bash_save_dir, exist_ok=True)

    if cluster == "cedar":
        # Determined from https://docs.alliancecan.ca/wiki/Cedar
        gpu_type = "v100l:"
        gpu_size = 32  # Gb
        max_ram_GPU_node = 187  # Gb
        # 32 cores for 4 GPUs => 8 cores per gpu
        cpu_per_task = 8
        # 187G for the node =>45 GB per 1 GPU / 8 = 
        mem_per_cpu = 5
    elif cluster == "narval":
        gpu_type = "a100:"
        gpu_size = 40
        max_ram_GPU_node = 498
        # 48 cores for 4 GPUs => 12 cores per gpu
        cpu_per_task = 12
        # 498G for the node => 120 GB per 1 GPU / 8 = 
        mem_per_cpu = 10
    else:  # we don't specify GPU type and will use whatever GPU is avaiable, not recommended for handling memory consumption.
        gpu_type = ""
        gpu_size = 0

    # Create SBATCH Script
    if args.hours > 24:
        time_for_job = """
#SBATCH --array=1-{jobnb}%1   # jobnb is the number of jobs in the chain. This bash script will be executed {jobnb} times.
#SBATCH --time=12:00:00 # Each job will last this specified time
""".format(
        jobnb=int(math.ceil(args.hours/12)),           
)
    else:
        time_for_job = """
#SBATCH --time={h}:{m}:00 # Each job will last this specified time
""".format(
        h=str(args.hours).zfill(2),
        m=str(args.minutes).zfill(2),
)
    batch_part = """#!/bin/bash
{t}
#SBATCH --account={a}
#SBATCH --nodes={n}
#SBATCH --tasks-per-node={tpn}
#SBATCH --cpus-per-task={cpt}
#SBATCH --mem-per-cpu={mpc}G
#SBATCH --gres=gpu:{gt}{ng}
#SBATCH --job-name={exp}
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user={mail}
#SBATCH --mail-type=ALL
""".format(
        t=time_for_job,
        a=account,
        n=args.nodes,
        tpn=args.gpus,
        cpt=cpu_per_task,
        mpc=mem_per_cpu,
        gt=gpu_type,
        ng=args.gpus,
        mail=email,
        exp=args.exp,
    ) 
    if args.existing_env:

        base_setup_pythonlibs_str = """
# Every library should be installed beforehand, we just source the env
source {env_path}/bin/activate
""".format(env_path=env_path)
    else:
        base_setup_pythonlibs_str = """
# Installing on compute node
virtualenv --no-download $SLURM_TMPDIR/densesslenv
source $SLURM_TMPDIR/densesslenv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index termcolor 

pip install --no-index pycocotools
pip install --no-index yacs

pip install --no-index tabulate
pip install --no-index cloudpickle
pip install --no-index tqdm
pip install --no-index tensorboard
pip install --no-index fvcore
pip install --no-index iopath
pip install --no-index dataclasses
pip install --no-index omegaconf
pip install --no-index hydra-core
pip install --no-index black
pip install --no-index packaging
pip install -e {path_to_detectron}
""".format(
    path_to_detectron=os.path.join(code_folder, "detectron2")
    )   

    if args.gpus >1 or args.nodes >1:
        setup_pythonlibs_str = """
srun -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF
{libs}
EOF
# Reactivating env on current terminal because the srun command is a subshell
source {where_env}
""".format(
    libs=base_setup_pythonlibs_str, 
    where_env=(
        os.path.join(env_path, "bin", "activate") if args.existing_env else "$SLURM_TMPDIR/bin/activate"
        )
    )
        ddp_setup_part = """
############################## DDP Set Up ##############################

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  #NCCL_BLOCKING_WAIT slows down inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
"""        
    else:
        setup_pythonlibs_str = base_setup_pythonlibs_str
        ddp_setup_part = ""
        

    python_env_part = """
# # Script will fail on first error encountered
# set -e
# All commends executed are printed on the terminal
set -x

############################## ENVIORNMENT Set Up ##############################

# If recreating
module --force purge
module load StdEnv python/3.10 scipy-stack
echo "==================Modules loaded=================="
# If using pre created
# module restore densesslModules

{piplibs}
echo "==================Python env installed=================="
""".format(
        piplibs=setup_pythonlibs_str,
        )
    
    data_part = """
############################## Data Set Up ##############################
# $SLURM_TMPDIR points to /localscratch/USER.JOBID.RANK??/, exemple: /localscratch/sebquet.41248695.0/
mkdir $SLURM_TMPDIR/data
cp -r {data_path} $SLURM_TMPDIR/data

cd $SLURM_TMPDIR/data/COCO
unzip -q train2017.zip
{uz}
echo "==================Data copied and unzipped=================="


""".format(
    # I called the folder containing all the zips COCO.
    data_path=os.path.join(data_folder, "COCO"), 
    uz="""
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip 
cd ..
mv COCO/ coco/
export DETECTRON2_DATASETS=$SLURM_TMPDIR/data/
""" if args.task == "detection" else ""
)

    slotcon_script_part = """

############################## Script ##############################

cd {github_repo}
{prefix}python SlotCon-main/main_pretrain.py \\
    --cc \\{ddp}
    \\
    --dataset COCO \\
    --data-dir $SLURM_TMPDIR/data/COCO \\
    --output-dir {out_path} \\
    \\
    --arch resnet50 \\
    --dim-hidden 4096 \\
    --dim-out 256 \\
    --num-prototypes 256 \\
    --teacher-momentum 0.99 \\
    --teacher-temp 0.07 \\
    --group-loss-weight 0.5 \\{decoder}
    \\
    --batch-size {bs} \\
    --optimizer lars \\
    --base-lr 1.0 \\
    --weight-decay 1e-5 \\
    --warmup-epoch 5 \\
    --epochs 800 \\
    --fp16 \\
    \\
    --print-freq 10 \\
    --save-freq 20 \\
    --auto-resume \\
    --num-workers {nw}

    """.format(
        github_repo=os.path.join(code_folder, "DenseSSL"),
        prefix=("srun " if (args.gpus > 1 or args.nodes > 1) else ""),
        ddp = ("""
    --no-ddp \\""" if (args.gpus == 1 and args.nodes == 1) else ""),
        out_path=bash_save_dir,
        nw=cpu_per_task, 
        bs=args.batch_size,
        decoder = ("""
    --use-decoder \\
    --group-loss-weight-dec {gld} \\
    --encoder-loss-weight {elw} \\
    --decoder-downstream-dataset {ddd}\\
    --decoder-type {dt}\\""".format(
        gld=args.group_loss_weight_dec, elw=args.encoder_loss_weight, ddd=args.decoder_downstream_dataset,
        dt=args.decoder_type
        ) if args.use_decoder else ""
        )
    )

    detection_script_part = """
cd {github_repo}
cd SlotCon-main/transfer/detection
python train_net.py --config-file configs/COCO_R_50_FPN_1x_SlotCon.yaml --num-gpus {gpus} --resume MODEL.WEIGHTS {enc_path} OUTPUT_DIR {out_path}
""".format(
    github_repo=os.path.join(code_folder, "DenseSSL"),
    out_path=bash_save_dir,
    enc_path = args.enc_path,
    gpus=args.gpus
    )

    if args.task == "slotcon":
        script_part = slotcon_script_part
    else:
        script_part = detection_script_part

    batch_string = batch_part + python_env_part + data_part + ddp_setup_part + script_part


    batch_filename = os.path.join(bash_save_dir, f"slotcon_pretraining_job_{args.exp}.sh")
    # print("batch_string CMD ", batch_string)
    
    with open(batch_filename, "w") as myfile:
        myfile.write(batch_string)

    # Print
    print(
        "Sending {c} cores job with {g} {gt} GPUs with {m} Gb RAM on {n} Nodes for {h} hours.\\n".format(
            c=cpu_per_task*args.nodes*args.gpus, g=args.gpus, gt=gpu_type, m=gpu_size, n=args.nodes, h=args.hours
        )
    )


    totalmem = int(mem_per_cpu) * int(cpu_per_task)
    if totalmem < max_ram_GPU_node:
        conclusion = "It should run"
        if totalmem < max_ram_GPU_node * args.gpus/4:
            conclusion += " without impacting your priority."
        else:
            conclusion += " but impact your priority."
    else:
        conclusion = "It will never run."
    
    print(
        "Total memory requested on the node (CPU) is "
        + str(totalmem)
        + " Gb, over the {} Gb available. {}".format(max_ram_GPU_node, conclusion)
    )

    # Run Slurm Batch Script
    # Changing dir so that slurm output file will be saved on this directory
    os.chdir(bash_save_dir)
    command = "sbatch {}".format(batch_filename)
    print(command)
    subprocess.call(command, shell=True)
    # os.remove(batch_filename)
    print("Job submitted !")

    # Archive code repo to be able to come back to executed code
    archive_path = os.path.join(bash_save_dir, "DenseSSL_repo_save.tar")
    command = [
        "tar",
        "--exclude=__pycache__",
        "--exclude=.git",
        "--exclude=*.pth",
        "-cvzf",
        archive_path,
        os.path.join(code_folder, "DenseSSL")
    ]
    subprocess.call(command)


if __name__ == "__main__":

    args = GetArguments()
    submit_slurm(args)
