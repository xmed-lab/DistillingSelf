CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=vclr --action=extract --target=train_set --sample_rate=5 --start=0 --end=10
CUDA_VISIBLE_DEVICES=0 python frame_feature_extractor.py --model=vclr --action=extract --target=train_set --sample_rate=5 --start=10 --end=20
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=vclr --action=extract --target=train_set --sample_rate=5 --start=20 --end=30
CUDA_VISIBLE_DEVICES=1 python frame_feature_extractor.py --model=vclr --action=extract --target=train_set --sample_rate=5 --start=30 --end=41
CUDA_VISIBLE_DEVICES=2 python frame_feature_extractor.py --model=vclr --action=extract --target=test_set --sample_rate=5 --start=41 --end=50
CUDA_VISIBLE_DEVICES=2 python frame_feature_extractor.py --model=vclr --action=extract --target=test_set --sample_rate=5 --start=50 --end=60
CUDA_VISIBLE_DEVICES=3 python frame_feature_extractor.py --model=vclr --action=extract --target=test_set --sample_rate=5 --start=60 --end=70
CUDA_VISIBLE_DEVICES=3 python frame_feature_extractor.py --model=vclr --action=extract --target=test_set --sample_rate=5 --start=70 --end=81