#finetune
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=swav --action=train --target=train_set --sample_rate=25 --pretrained_model=/home/xdingaf/share/surgical_code/swav/log/checkpoint.pth.tar --linear --start=1  --end=41

CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=resnet50 --action=train --target=train_set --sample_rate=25 --best_ep=199 --start=1  --end=5 --epochs=10

CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=resnet50_IN --action=train --target=train_set --sample_rate=25 --dataset=cholec80 --imagenet --start=1  --end=21
#  --pretrained_model=./saved_models/cholec80/mocov2_noIN/183.pth.tar
#extract feature
# python frame_feature_extractor.py --model=vclr_notrain --action=extract --target=train_set --sample_rate=5
# python frame_feature_extractor.py --model=vclr_notrain --action=extract --target=test_set --sample_rate=5 
CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17 --action=extract --target=train_set --sample_rate=5 --start=1  --end=10 --best_ep=4
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17 --action=extract --target=train_set --sample_rate=5 --start=10  --end=20 --best_ep=4
CUDA_VISIBLE_DEVICES=7 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17   --action=extract --target=train_set --sample_rate=5 --start=20 --end=30 --best_ep=4
CUDA_VISIBLE_DEVICES=2 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17    --action=extract --target=train_set --sample_rate=5 --start=30 --end=41 --best_ep=4
CUDA_VISIBLE_DEVICES=3 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17  --action=extract --target=test_set --sample_rate=5 --start=41 --end=50 --best_ep=4
CUDA_VISIBLE_DEVICES=4 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17 --action=extract --target=test_set --sample_rate=5 --start=50 --end=60 --best_ep=4
CUDA_VISIBLE_DEVICES=5 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17  --action=extract --target=test_set --sample_rate=5 --start=60 --end=70 --best_ep=4
CUDA_VISIBLE_DEVICES=6 python frame_feature_extractor.py --model=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e17  --action=extract --target=test_set --sample_rate=5 --start=70 --end=81 --best_ep=4

###
CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=mocov2base251 --action=extract --target=train_set --sample_rate=5 --start=1 --end=4 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=mocov2base251 --action=extract --target=train_set --sample_rate=5 --start=4 --end=8 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=mocov2base251 --action=extract --target=train_set --sample_rate=5 --start=8 --end=12 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=mocov2base251 --action=extract --target=train_set --sample_rate=5 --start=12 --end=18 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=train_set --sample_rate=5 --start=18 --end=23 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=train_set --sample_rate=5 --start=23 --end=28 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=2 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=test_set --sample_rate=5 --start=1 --end=4 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=2 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=test_set --sample_rate=5 --start=4 --end=7 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=3 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=test_set --sample_rate=5 --start=7 --end=10 --dataset=m2cai16
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=resnet50_IN --action=extract --target=test_set --sample_rate=5 --start=10 --end=15 --dataset=m2cai16

##train TCN
CUDA_VISIBLE_DEVICES=1 python train.py --action=base_train --sample_rate=5 --backbone=mocov2base_b0_s25_g0_w0_p0_sem0_w0_e0_dFalse_INpre0_onlyfcFalse_superFalse_ms2048_numin2_cat0_dis1_w5.0_epoch199_s1_e5
CUDA_VISIBLE_DEVICES=2 python train.py --action=base_predict --sample_rate=5 --backbone=resnet50_noIN_epoch199_s1_e5 --best_ep=53 --fps=5 
CUDA_VISIBLE_DEVICES=1 python train.py --action=base_predict --sample_rate=5 --backbone=mocov2_un200 --best_ep=24 --fps=5

#
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_region.py   -a resnet50   --lr 0.015   --batch-size 128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --method=base --sample_rate=25 --generative=0 --recon_weight=0 --big=0 --predictor=0
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py   -a resnet50   --lr 0.015   --batch-size 128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --method=71_heart --sample_rate=25 --generative=0 --recon_weight=0 --big=0 --predictor=0  --epochs=200
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py   -a resnet50   --lr 0.010   --batch-size 128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --method=base --sample_rate=25 --generative=0 --recon_weight=0 --recon_weight=0 --big=0 --sem_weight=0 --semantic=0 --extra=0 --moco-k=2048 --concatenate=0 --moco_pre

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py   -a resnet50   --lr 0.015   --batch-size 128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --moco-t 0.2 --aug-plus --cos --method=base --sample_rate=25 --generative=0 --recon_weight=1 --big=0 --sem_weight=0 --semantic=0 --extra=0  --decouple_weight=0.5 --decoupling 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py   -a resnet50   --lr 0.015   --batch-size 128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --moco-t 0.2 --aug-plus --cos --method=base --sample_rate=25 --generative=1 --recon_weight=1 --big=0 --sem_weight=0 --semantic=0 --extra=0  


CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py   -a resnet50   --lr 0.010   --batch-size 128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --method=base --sample_rate=25  --moco-k=2048 --dis_weight=1 --distill=1
# data_dir="/home/ubuntu/data/kinetics400_30fps_frames"
output_dir="./saved_models/cholec80/vclr"
pretrained="pretrain/moco_v2_200ep_pretrain.pth.tar"
num_replica=8

mkdir -p ${output_dir}


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_port 12857 --nproc_per_node=4 \
    train_vclr.py \
    --datasplit=train \
    --output_dir=./saved_models/cholec80/vclr_seg_pre \
    --model_mlp \
    --batch_size 32 \
    --sample_rate=1 \
    --nce_k=10000 \
    --seg_num=10 \
    --pretrained_model=./saved_models/cholec80/mocov2_un200/200.pth.tar
   

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_simsiam.py   -a resnet50   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0   --fix-pred-lr --sample_rate=25 --batch-size=512 --lr=0.05

sshfs xdingaf@eez192.ece.ust.hk:/home/xdingaf /home/xdingaf/share192/
fusermount -u /home/xdingaf/share
