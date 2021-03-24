#python main.py -train train -test testA -text_encoder bert -ensemble_num 5 -num_hidden_layers 8 -num_attention_heads 8 -intermediate_size 2304
#python main.py -train train -test testA -text_encoder bert -ensemble_num 5 -num_hidden_layers 6 -num_attention_heads 6 -intermediate_size 1536
#python main.py -train train -test testA -text_encoder bert -ensemble_num 5 -word_embedding_dim 96 -hidden_dim 512 -num_hidden_layers 3 -num_attention_heads 2 -intermediate_size 768
python main.py -train train -test testA -text_encoder bert -ensemble_num 5