export EXPR_ID='exp'                     #UNIQUE_EXPR_ID
export DATA_DIR='/local/scratch/jrs596/dat/ResNetFung50+_images_organised'                       #PATH_TO_DATA_DIR
export CHECKPOINT_DIR='/local/scratch/jrs596/dat/NVAE/CheckPoint'       #PATH_TO_CHECKPOINT_DIR
export CODE_DIR='/home/userfs/j/jrs596/scripts/CocoaReader/NVAE'        #PATH_TO_CODE_DIR
cd $CODE_DIR


python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR \
--eval_mode=evaluate --num_iw_samples=1000 --input_size 112