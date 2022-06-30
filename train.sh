python3 anno_train_ssd.py --dataset_type open_images --datasets /home/supernode/Downloads/clean_bot --net mb3-ssd-lite  \
--model_save_folder models/test0610/new_model_cls7_ori700  \
--scheduler multi-step --lr 0.01 --t_max  50 --validation_epochs 1 --num_epochs 200 --base_net_lr 0.01 --batch_size 32
