export EXPR_ID='exp'                     #UNIQUE_EXPR_ID
#export DATA_DIR='/local/scratch/jrs596/dat/ResNetFung50+_images_organised'                       #PATH_TO_DATA_DIR
export DATA_DIR='/local/scratch/jrs596/dat/test'
export CHECKPOINT_DIR='/local/scratch/jrs596/dat/NVAE/CheckPoint'       #PATH_TO_CHECKPOINT_DIR
export SAVEDIR='/local/scratch/jrs596/dat/NVAE/eval'
export FID_DIR='/local/scratch/jrs596/dat/NVAE/eval/FID'
export CODE_DIR='/home/userfs/j/jrs596/scripts/NVAE'        #PATH_TO_CODE_DIR
cd $CODE_DIR


python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR \
--eval_mode=evaluate_fid --num_iw_samples=1000 --input_size 112 --save $SAVEDIR --world_size 2 \
--fid_dir $FID_DIR --temp=0.6 --readjust_bn