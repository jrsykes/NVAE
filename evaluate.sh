export EXPR_ID='HFDS'                     #UNIQUE_EXPR_ID
#export DATA_DIR='/local/scratch/jrs596/dat/ResNetFung50+_images_organised'                       #PATH_TO_DATA_DIR
#export DATA_DIR='/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_NVAE_PNP_filtered'
export DATA_DIR='/scratch/staff/jrs596/dat/CLS-LOC'
#export DATA_DIR='/scratch/staff/jrs596/dat/test4'
export CHECKPOINT_DIR='/scratch/staff/jrs596/dat'       #PATH_TO_CHECKPOINT_DIR
export SAVEDIR='/scratch/staff/jrs596/dat/NVAE/eval_HFDS'
export FID_DIR='/scratch/staff/jrs596/dat/NVAE/eval_HFDS/FID'
export CODE_DIR='/home/userfs/j/jrs596/scripts/NVAE'        #PATH_TO_CODE_DIR
cd $CODE_DIR

now=`date`

python evaluate.py --checkpoint $CHECKPOINT_DIR/checkpoint.pt --data $DATA_DIR --eval_mode=evaluate \
--num_iw_samples=1 --input_size 256 --save $SAVEDIR --world_size 1 --batch_size 1 

then=`date`
echo "Now: $now"
echo "then: $then"



