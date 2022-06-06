export EXPR_ID='exp'                     #UNIQUE_EXPR_ID
#export DATA_DIR='/local/scratch/jrs596/dat/ResNetFung50+_images_organised'                       #PATH_TO_DATA_DIR
export DATA_DIR='/local/scratch/jrs596/dat/test'
export CHECKPOINT_DIR='/local/scratch/jrs596/dat/NVAE/CheckPoint'       #PATH_TO_CHECKPOINT_DIR
export FID_DIR='/local/scratch/jrs596/dat/NVAE/eval/FID'
export CODE_DIR='/home/userfs/j/jrs596/scripts/NVAE/scripts'        #PATH_TO_CODE_DIR
cd $CODE_DIR

python precompute_fid_statistics.py --data $DATA_DIR --dataset custom\
 --fid_dir /tmp/fid-stats/ --batch_size 21 --fid_dir $FID_DIR --input_size 112