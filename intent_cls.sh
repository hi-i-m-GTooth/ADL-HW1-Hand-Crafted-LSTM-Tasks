# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
in=$1
out=$2

python test_intent.py --test_file ${in} --ckpt_path intent_model.ckpt --pred_file ${out} --batch_size 128 --device cuda