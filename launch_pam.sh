# EDM training for mice brain PAM images

mpiexec --allow-run-as-root -n 4 python edm_train.py --ema_rate 0.999 --global_batch_size 128 --image_size 256 --lr 0.0001 --schedule_sampler lognormal --use_fp16 True --weight_decay 0.01 --weight_schedule karras --dataset brain


# Consistency training on mice brain PAM images

mpiexec --allow-run-as-root -n 4 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 800000 --loss_norm huber --lr_anneal_steps 0 --teacher_model_path "" --ema_rate 0.999 --global_batch_size 256 --lr 0.0001 --schedule_sampler uniform --use_fp16 True --weight_decay 0.01 --weight_schedule uniform --dataset brain --image_size 128

# Consistency training on bone PAM images

mpiexec --allow-run-as-root -n 4 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 800000 --loss_norm huber --lr_anneal_steps 0 --teacher_model_path "" --ema_rate 0.999 --global_batch_size 64 --lr 0.00001 --schedule_sampler uniform --use_fp16 True --weight_decay 0.01 --weight_schedule uniform --dataset bone --image_size 256