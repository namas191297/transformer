import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from einops import repeat
from config import *
from torch.nn.functional import softmax

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_embed):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)

    def forward(self, x):
        return self.embedding(x)
    
class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, embeddings):
        
        # Create an empty vector to hold the positional encodings for just one element in the batch
        pos_enc = torch.zeros((embeddings.shape[1], embeddings.shape[2]), requires_grad=False)
        for pos in range(embeddings.shape[1]):
            for i in range(embeddings.shape[2]):
                angle = torch.tensor(pos / (10000 ** ((2 * i)/(self.d_embed))))
                pos_enc[pos, i] = torch.sin(angle) if i % 2 == 0 else torch.cos(angle)

        # Add a dimension at the start [seq_len, d_embed] -> [1, seq_len, d_embed]
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)

        # Increase the size of the embeddings
        embeddings = embeddings * np.sqrt(self.d_embed)

        # Add positional encodings to x
        embeddings = embeddings + torch.autograd.Variable(pos_enc, requires_grad=False)

        return embeddings
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()

        # Initialize the parameters for the model
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads
        assert self.d_head * n_heads == self.d_embed, "Change number of heads!"

        # Create projections for K,Q and V
        # Input to these projections will be the embeddings with Positional Encodings [b, seq_len, n_heads, head_dim]
        # The projections will result in the following:
        # K : [b, seq_len, n_heads, head_dim] -> [b, seq_len, n_heads, head_dim]
        # Q : [b, seq_len, n_heads, head_dim] -> [b, seq_len, n_heads, head_dim]
        # V : [b, seq_len, n_heads, head_dim] -> [b, seq_len, n_heads, head_dim]
        self.k_projection = nn.Linear(self.d_head, self.d_head, bias=False)
        self.q_projection = nn.Linear(self.d_head, self.d_head, bias=False)
        self.v_projection = nn.Linear(self.d_head, self.d_head, bias=False)

    def forward(self, k, q, v, mask=None):
        
        # Resize the input embeddings depending the no. of heads
        # Here, k = q = v = Embeddings
        b, seq_len, _ = k.shape 
        k = k.view(b, seq_len, self.n_heads, self.d_head)
        q = q.view(b, seq_len, self.n_heads, self.d_head)
        v = v.view(b, seq_len, self.n_heads, self.d_head)

        # Obtain the k,q and v matrices by projecting the embeddings
        k = self.k_projection(k)
        q = self.q_projection(q)
        v = self.v_projection(v)

        # Perform mat-multiplication between q and k.transpose
        # Shape of k and q => [b, seq_len, n_head, d_head]
        # We need output of the shape [b, n_head, seq_len, seq_len], so we reshape k and q to [b, n_head, seq_len, d_head]
        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        k_transpose = torch.transpose(k, 2, 3)
        qk_matmul_prod = torch.matmul(q, k_transpose) / np.sqrt(self.d_head) # Divide by head dimension
        
        # If a mask has been provided, perform masking, otherwise skip
        if mask is not None:
            qk_matmul_prod.masked_fill_(mask == 0, 1e-19)
        

        # Perform softmax to obtain the probabilities on the dimension of keys (last dimension)
        # => [b, d_head, seq_len, seq_len]
        qk_matmul_prod = softmax(qk_matmul_prod, dim=3)

        # Perform matrix multiplication of this product with the value matrix
        # qk => [b, n_head, seq_len, seq_len]
        # v => [b, seq_len, n_head, head_dim]
        # Therefore, we need to take the transpose of value matrix so that we can get an output of
        # output => [b, n_head, seq_len, head_dim]
        v = v.transpose(1, 2)
        out = torch.matmul(qk_matmul_prod, v)

        # Merge the heads to form the output vector after reshaping
        # out => [b, n_head, seq_len, d_head] to [b, seq_len, n_head, d_head] to [b, seq_len, n_head * d_head]
        out = out.transpose(1,2).contiguous()
        out = out.view(b, seq_len, -1)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed

        # Define the sub-layers
        self.mha = MultiHeadAttention(self.n_heads, self.d_embed)
        self.layer_norm1 = nn.LayerNorm(self.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_embed, self.d_embed * transformer_config['expansion_factor']),
            nn.ReLU(),
            nn.Linear(self.d_embed * transformer_config['expansion_factor'], self.d_embed)
        )
        self.layer_norm2 = nn.LayerNorm(self.d_embed)
    
    def forward(self, x):

        # Received embeddings are passed through the MHA module first 
        op = self.mha(x,x,x, mask=None)

        # Perform LayerNorm1 after adding the output to the input Embeddings
        op_norm1 = self.layer_norm1(op + x)

        # Pass the output to the feedforward network
        ff_out = self.feed_forward(op_norm1)

        # Perform LayerNorm2 after adding output from LayerNorm1
        op_norm2 = self.layer_norm2(ff_out + op_norm1)

        return op_norm2
        
class DecoderBlock(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed

        # Define the sub-layers
        self.masked_mha = MultiHeadAttention(self.n_heads, self.d_embed)
        self.mha = MultiHeadAttention(self.n_heads, self.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_embed, self.d_embed * transformer_config['expansion_factor']),
            nn.ReLU(),
            nn.Linear(self.d_embed * transformer_config['expansion_factor'], self.d_embed)
        )
        self.layer_norm1 = nn.LayerNorm(self.d_embed)
        self.layer_norm2 = nn.LayerNorm(self.d_embed)
        self.layer_norm3 = nn.LayerNorm(self.d_embed)

    def forward(self, x, encoder_k, encoder_v, mask):
        
        # Received embeddings are passed through the Masked-MHA module first with the mask
        op = self.masked_mha(x, x, x, mask=mask)

        # Perform LayerNorm1 after adding the output to the input Embeddings
        op_norm1 = self.layer_norm1(op + x)

        # Pass this output to the MHA module with given encoder keys and encoder vals
        op = self.mha(encoder_k, op_norm1, encoder_v, mask=None)

        # Perform LayerNorm2 after adding output from LayerNorm1
        op_norm2 = self.layer_norm2(op + op_norm1)

        # Perform Feed-Forward on this output
        ff_out = self.feed_forward(op_norm2)

        # Perform LayerNorm3 after adding the output from LayerNorm2
        op = self.layer_norm3(ff_out + op_norm2)

        return op
    
class TransformerDecoder(nn.Module):
    def __init__(self, n_layers=decoder_config['n_layers'], n_heads=decoder_config['n_heads'], d_embed=transformer_config['d_embed']):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.embedding_layer = EmbeddingLayer(vocab_size=data_config['vocab_size'], d_embed=self.d_embed)
        self.position_encoding_layer = PositionalEncodingLayer(d_embed=self.d_embed)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(self.n_heads, self.d_embed) for _ in range(self.n_layers)]
        )
        self.linear = nn.Linear(self.d_embed, data_config['vocab_size'])
    
    def forward(self, input, encoder_k, encoder_v):

        # Create a mask for the inputs to the decoder
        b, seq_len = input.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        # Now, the mask size is [1, seq_len, seq_len], corresponding to each head of the qk matrix of size [b, n_head, seq_len, seq_len] after matmul.
        # Since we have b batches, we need to repeat the same mask for each element in the batch
        mask = repeat(mask, '1 1 s1 s2 -> b 1 s1 s2', b=b)
        
        # Obtain the embeddings and positional encodings from the given input ids
        input_embeddings = self.embedding_layer(input)
        input_embeddings = self.position_encoding_layer(input_embeddings)

        # Iterate through the decoder blocks and process the input embeddings
        out = input_embeddings
        for module in self.decoder_blocks:
            out = module(out, encoder_k, encoder_v, mask)

        # Pass the output through the linear layer to obtain the vocab logits
        linear_out = self.linear(out)

        return linear_out

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers=encoder_config['n_layers'], n_heads=encoder_config['n_heads'], d_embed=transformer_config['d_embed']):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.embedding_layer = EmbeddingLayer(vocab_size=data_config['vocab_size'], d_embed=self.d_embed)
        self.position_encoding_layer = PositionalEncodingLayer(d_embed=self.d_embed)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(self.n_heads, self.d_embed) for _ in range(self.n_layers)]
        )

    def forward(self, x):

        # Obtain the embeddings and positional encodings from the given input ids
        input_embeddings = self.embedding_layer(x)
        input_embeddings = self.position_encoding_layer(input_embeddings)

        # Iterate through the encoder blocks and process the input embeddings
        out = input_embeddings
        for module in self.encoder_blocks:
            out = module(out)
        
        return out

if __name__ == '__main__':
    # Sample Input
    test_input = torch.randint(0,100,(5,10))
    transformer_decoder = TransformerDecoder()
    transformer_encoder = TransformerEncoder()
    encoder_k = encoder_v = transformer_encoder(test_input)
    print(f'Encoder Output: {encoder_k.shape}')
    out = transformer_decoder(test_input, encoder_k, encoder_v)
    print(f'Decoder Output:{out.shape}')
