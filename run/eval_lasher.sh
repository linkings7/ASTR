config=$1 # ASTR_rgbt
weight=$2 # ./output/LasHeR.pth.tar
thresh=0.7
script='ASTR'

python ./RGBT_workspace/test_rgbt_mgpus.py --script_name ${script} --dataset_name LasHeR --yaml_name experiments/ASTR/${config}.yaml --threads 1 --weight_name ${weight}  --score_thresh ${thresh} --online
