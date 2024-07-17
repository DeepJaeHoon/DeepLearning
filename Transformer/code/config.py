import torch

class Configs:
    def __init__(self):
        # Model parameters
        self.pred_len = 3
        self.enc_in = 5 
        self.dec_in = 5  
        self.c_out = 5   
        self.d_model = 512
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.15
        self.e_layers = 6
        self.d_layers = 6  
        self.d_ff = 2048
        self.n_heads = 8
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = False
        self.output_attention = False
        
        # Training parameters
        self.seq_length = 7  # Length of input sequence
        self.predict_length = 3  # Length of prediction sequence
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.nb_epochs = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"