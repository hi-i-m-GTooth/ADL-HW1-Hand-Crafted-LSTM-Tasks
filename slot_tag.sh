in=$1
out=$2
python test_slot.py --test_file ${in} --ckpt_path slot_model.ckpt --pred_file ${out} --max_len 35 --batch_size 128 --device cuda