export EXPR_ID='num_groups_per_scale=2_num_latent_per_group2'                     #UNIQUE_EXPR_ID
export DATA_DIR='/scratch/jrs596/dat/ResNetFung50+_images_organised'  
#export DATA_DIR='/scratch/jrs596/dat/test'                       #PATH_TO_DATA_DIR
export CHECKPOINT_DIR='/scratch/jrs596/dat/NVAE/CheckPoint'       #PATH_TO_CHECKPOINT_DIR
export CODE_DIR='/home/userfs/j/jrs596/scripts/NVAE'        #PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset custom --batch_size 29 \
        --input_size 256 --epochs 100 --num_latent_scales 1 --num_groups_per_scale 2 --num_postprocess_cells 3 \
        --num_preprocess_cells 3 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 2 \
        --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --fast_adamax