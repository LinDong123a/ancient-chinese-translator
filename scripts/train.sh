# python main.py \
#     --model GRU --data_dir dataset --default_root_dir models --log_every_n_steps 1\
#     --batch_size 4 --lr 0.01 --max_epochs 100\
#     --embedding_dim 128 --enc_hidden_dim 256 --dec_hidden_dim 256 --dropout 0.1

python main.py \
    --model transformer --data_dir dataset --default_root_dir models --log_every_n_steps 1\
    --batch_size 4 --lr 0.01 --max_epochs 100\
    --d_model 256 --hidden_dim 256 --n_head 4 --n_enc_layers 6 --n_dec_layers 6 --dropout 0.1