CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_239.py\
    --mode 'train'\
    --name 'exp57'\
    --mri_type 'adc'\
    --age_type 'cat'\
    \
    --no-use_multiGPU\
    --num_machines 1\
    --num_gpu_processes 1\
    --machine_id 0\
    \
    --lr 2e-5\
    --batch_size 4\
    --num_iteration 10000\
    --dropout 0.1\
    \
    --time_step 300\
    --schedule 'linear'\
    --image_size 104 104 72\
    --base_channels 64\
    --channel_mults 1 2 4 8\
    --num_res_blocks 1\
    --time_emb_dim 256\
    \
    --ema_decay 0.9999\
    --ema_update_rate 1\
    \
    --log_rate 100\
    --save_rate 1000\
    >output.txt &
