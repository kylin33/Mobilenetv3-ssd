# python3 train_ssd.py --dataset_type open_images --datasets data/  --net mb3-ssd-lite --pretrained_ssd models/pt_model/mb3-ssd-lite-Epoch-149-Loss-5.782852862012213.pth --scheduler cosine --lr 0.001 --t_max 100 --validation_epochs 1 --num_epochs 200 --base_net_lr 0.001 --batch_size 5
python3 anno_ssd_detect.py --dataset_type open_images --datasets /home/whf/Temp/11-扫地机/data/   --net mb3-ssd-lite --resume /home/whf/Temp/11-扫地机/projects/MobileNetV3-SSD/models/new_model/last.pt  --scheduler multi-step --lr 0.01 --t_max  50 --validation_epochs 8 --num_epochs 200 --base_net_lr 0.001 --batch_size 1
# python3 train_ssd.py --dataset_type open_images --datasets data/   --net mb3-ssd-lite  --resume  /home/supernode/anno/MobileSSD/MobileNetV3-SSD/models/new_model_0212/last.pt --scheduler multi-step --lr 0.01 --t_max  50 --validation_epochs 8 --num_epochs 300 --base_net_lr 0.001 --batch_size 32