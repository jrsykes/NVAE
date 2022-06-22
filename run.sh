#!/bin/bash

#SBATCH --partition=big

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=NVAE

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jrs596@york.ac.uk

# run the application

module load python/anaconda3
module load cuda/11.2
module load pytorch/1.9.0

conda init bash
source ~/.bashrc
conda activate NVAE

export EXPR_ID='full_minimal_model_8channels'                     #UNIQUE_EXPR_ID
export DATA_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/ResNetFung50+_images_organised'  
#export DATA_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/test'  
export CHECKPOINT_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/NVAE/CheckPoint'       #PATH_TO_CHECKPOINT_DIR
export CODE_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/NVAE'        #PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset custom --batch_size 42 \
        --input_size 256 --epochs 400 --num_latent_scales 1 --num_groups_per_scale 2 --num_postprocess_cells 3 \
        --num_preprocess_cells 3 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 2 \
        --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 8 --num_channels_dec 8 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --fast_adamax




