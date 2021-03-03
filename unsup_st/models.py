import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Speech2VecRNN(nn.Module):
    def __init__(self,
                 input_size=13,
                 hidden_size=100,
                 hidden_channels=None,
                 scale_factor=None,
                 mean=None,
                 std=None,
                 additive_margin=None):
        super(Speech2VecRNN, self).__init__()

        self.scale_factor = scale_factor

        self.hidden_size = hidden_size

        self.hidden_channels = hidden_channels or (32, 48)

        self.kernel_size = 5
        self.padding = 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_channels[0], kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_channels[1], out_channels=hidden_size, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU()
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.projection = nn.Linear(hidden_size*2, hidden_size*2)

        self.loss_func = nn.CrossEntropyLoss()

        self.mean = mean
        self.std = std

        self.additive_margin = additive_margin

    def forward(self, xs, xs_len, ys=None, ys_len=None):
        if self.mean is not None:
            xs = (xs - self.mean) / self.std
        if ys is not None:
            if self.mean is not None:
                ys = (ys - self.mean) / self.std

        if self.scale_factor is not None:
            xs = nn.functional.interpolate(xs.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)
            if ys is not None:
                ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)

            xs_len = [l * self.scale_factor for l in xs_len]
            if ys_len is not None:
                ys_len = [l * self.scale_factor for l in ys_len]

        f = lambda x: math.floor((x + 2 * self.padding - (self.kernel_size - 1) - 1) / 2 + 1)
        xs = self.cnn(xs.transpose(1, 2)).transpose(1, 2)
        xs_len = [f(f(f(l))) for l in xs_len]
        xs = nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)

        _, (xs_embed, _) = self.encoder(xs)
        xs_embed = torch.cat((xs_embed[0], xs_embed[1]), dim=1)

        loss = None
        if ys is not None:
            ys = self.cnn(ys.transpose(1, 2)).transpose(1, 2)
            ys_len = [l // 8 for l in ys_len]
            ys = nn.utils.rnn.pack_padded_sequence(ys, ys_len, batch_first=True, enforce_sorted=False)
            _, (ys_embed, _) = self.encoder(ys)
            ys_embed = torch.cat((ys_embed[0], ys_embed[1]), dim=1)
            ys_embed = self.projection(ys_embed)

            batch_size = xs_embed.shape[0]
            pred = torch.mm(xs_embed, ys_embed.transpose(0, 1))
            if self.additive_margin is not None:
                pred -= self.additive_margin * torch.eye(batch_size, device=pred.device)
            gold = torch.arange(start=0, end=batch_size, device=pred.device)
            loss = self.loss_func(pred, gold)

        return loss, xs_embed


class Speech2VecXF(nn.Module):
    def __init__(self,
                 input_size=13,
                 hidden_size=100,
                 layers=2,
                 scale_factor=None,
                 mean=None,
                 std=None,
                 additive_margin=None):
        super(Speech2VecXF, self).__init__()

        self.scale_factor = scale_factor

        self.encoder = Encoder(
            d_input=input_size,
            n_layers=layers,
            n_head=8,
            d_k=hidden_size//8,
            d_v=hidden_size//8,
            d_model=hidden_size,
            d_inner=hidden_size*4
        )

        self.projection = nn.Linear(hidden_size, hidden_size)

        self.loss_func = nn.CrossEntropyLoss()

        self.mean = mean
        self.std = std

        self.additive_margin = additive_margin

    def forward(self, xs, xs_len, ys=None, ys_len=None):
        if self.mean is not None:
            xs = (xs - self.mean) / self.std
        if ys is not None:
            if self.mean is not None:
                ys = (ys - self.mean) / self.std

        if self.scale_factor is not None:
            xs = nn.functional.interpolate(xs.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)
            if ys is not None:
                ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)

            xs_len = [int(l * self.scale_factor) for l in xs_len]
            if ys_len is not None:
                ys_len = [int(l * self.scale_factor) for l in ys_len]

        xs_embed = self.encoder(xs, xs_len)[0]
        xs_embed = xs_embed[:, 0, :]    # take the embeddings for the first token
        # TODO: compare CLS vs MOT

        loss = None
        if ys is not None:
            ys_embed = self.encoder(ys, ys_len)[0]
            ys_embed = ys_embed[:, 0, :]
            ys_embed = self.projection(ys_embed)

            batch_size = xs_embed.shape[0]
            pred = torch.mm(xs_embed, ys_embed.transpose(0, 1))
            if self.additive_margin is not None:
                pred -= self.additive_margin * torch.eye(batch_size, device=pred.device)
            gold = torch.arange(start=0, end=batch_size, device=pred.device)
            loss = self.loss_func(pred, gold)

        return loss, xs_embed
