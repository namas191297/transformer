transformer_config = {
    'd_embed':512,
    'expansion_factor':5
}

encoder_config = {
    'n_layers':6,
    'n_heads':8,
    'd_key': 64,
    'd_value':64,
}

decoder_config = {
    'n_layers':6,
    'n_heads':8,
    'd_key': 64,
    'd_value':64,
}

training_config = {
    'epochs':50,
    'lr':4e-4,
    'batch_size':1,
    'max_length':77 # (HuggingFace Option) Change based on the model that you are using, if you are training from scratch, IGNORE
}

data_config = {
    'vocab_size':49411, # Change depending on the tokenizer you use, by default for CLIP it should be 49408
}                      # You can check this by using len(tokenizer) for HuggingFace