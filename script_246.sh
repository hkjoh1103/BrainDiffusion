CUDA_VISIBLE_DEVICES=0 nohup python -u main_246.py\
    --mode 'train'\
    --name 'exp78'\
    --mri_type 'adc'\
    --age_type 'pos'\
    \
    --no-use_multiGPU\
    --num_machines 1\
    --num_gpu_processes 1\
    --machine_id 0\
    \
    --lr 2e-5\
    --batch_size 2\
    --num_iteration 40000\
    --dropout 0.1\
    \
    --time_step 300\
    --schedule 'linear'\
    --image_size 128 128 40\
    --base_channels 64\
    --channel_mults 1 2 4 8\
    --num_res_blocks 1\
    --time_emb_dim 256\
    \
    --use_ema\
    --ema_decay 0.9999\
    --ema_update_rate 1\
    \
    --log_rate 10\
    --save_rate 2000\
    >output.txt &