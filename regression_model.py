import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('position', torch.arange(max_len).unsqueeze(1).float())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.div_term = div_term.unsqueeze(0)

    def forward(self, x):
        # Calculate positional encoding dynamically based on the sequence length of input x
        position = self.position[:x.size(0), :]
        div_term = self.div_term[:, :x.size(1)]
        sinusoid_input = torch.matmul(position, div_term)
        position_encoding = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
        return x + position_encoding.unsqueeze(0).to(x.device)

class RegressionTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, max_sequence_length):
        super(RegressionTransformerNetwork, self).__init__()

        # Embedding layer for the first input
        self.embedding1 = nn.Linear(input_size, hidden_size)
        # Embedding layer for the second input
        self.embedding2 = nn.Linear(input_size, hidden_size)

        # Positional encoding for both inputs
        self.positional_encoding1 = PositionalEncoding(hidden_size, max_sequence_length)
        self.positional_encoding2 = PositionalEncoding(hidden_size, max_sequence_length)

        # Transformer Encoder
        encoder_layers1 = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layers1,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
        )

        # Transformer Encoder
        encoder_layers2 = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layers2,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2):
        # Embedding and positional encoding for the first input
        x1 = self.embedding1(x1)
        x1 = self.positional_encoding1(x1)

        # Embedding and positional encoding for the second input
        x2 = self.embedding2(x2)
        x2 = self.positional_encoding2(x2)

        # Concatenate the two inputs along the sequence length dimension
        #x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0)), dim=0)

        # Transformer Encoder
        #x = x = x.permute(1, 0, 2)#x.transpose(0, 1)
        x1 = self.transformer_encoder1(x1)
        x1 = x1.mean(dim=0)

        x2 = self.transformer_encoder2(x2)
        x2 = x2.mean(dim=0)
        # Output layer
        x1 = self.output_layer(x1)
        x2 = self.output_layer(x2)

        output = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0)),dim=0).mean(dim=0)

        return output.squeeze(1)


