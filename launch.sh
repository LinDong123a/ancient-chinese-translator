python main.py \
    --model GRU --data_dir dataset --default_root_dir models --log_every_n_steps 1\
    --batch_size 4 --lr 0.01 --max_epochs 100\
    --embedding_dim 128 --hidden_dim 256