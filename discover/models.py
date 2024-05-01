
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM
import numpy as np

import os
os.environ['TRANSFORMERS_CACHE'] = '.transformers/'

import random
import math

from role_assignment_functions import *
import config

# Determine whether to use GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Conventions (set in config.py):
# 0 is pad_token
# 1 is sos
# 2 is eos

# Every encoder should take a list of integers as input, and return a vector encoding, or a list of vector encodings.
# Every decoder should take a vector encoding as input, and return a list of logits.

# PADDING FOR VARIABLE LENGTH SEQUENCES IN BATCH
# - Reserve 0 for padding, always

# INDICES VS. VOCABULARY: Models only interface with indices


###############################################################################################
# Sequence-to-sequence model
# Can use any type of encoder and any type of decoder
###############################################################################################
class Seq2Seq(nn.Module):
    def __init__(self, architecture=None, vocab_size=None, n_roles=None, hidden_size=None, emb_size=None, role_emb_size=None, n_layers=1, dropout=0.1, max_length=30, seq2roles=None, nonlinearity="ReLU", uniform_src_length=None, n_head=None, dim_feedforward=None, has_linear_layer=False, tpr_invert_decoder=False, return_bindings=False, aggregation="sum",
            decoder_architecture=None, decoder_vocab_size=None, decoder_n_roles=None, decoder_hidden_size=None, decoder_emb_size=None, decoder_role_emb_size=None, decoder_n_layers=None, decoder_dropout=None, decoder_max_length=None, decoder_seq2roles=None, decoder_nonlinearity=None, uniform_trg_length=None, decoder_n_head=None, decoder_dim_feedforward=None, no_eos=False, bottleneck=False, decoder_has_linear_layer=None, tpr_invert_encoder=False, encoding_is_seq=None,
            encoder=None, decoder=None, **kwargs):
        super(Seq2Seq, self).__init__()

        # If the encoder and decoder differ in any hyperparameters,
        # then the basic hyperparameter name (e.g., 'vocab_size') will
        # be used for the encoder, while the one starting with 'decoder_'
        # (e.g., 'decoder_vocab_size') will be used for the decoder
        # If no 'decoder_' parameter is specified, we default to assuming
        # that the encoder and decoder have the same hyperparameters
        if decoder_architecture is None:
            decoder_architecture = architecture
        if decoder_vocab_size is None:
            decoder_vocab_size = vocab_size
        if decoder_n_roles is None:
            decoder_n_roles = n_roles
        if decoder_hidden_size is None:
            decoder_hidden_size = hidden_size
        # To prevent the decoder from getting the previous output
        # as an input, set decoder_emb_size to 0
        if decoder_emb_size is None:
            decoder_emb_size = emb_size
        if decoder_role_emb_size is None:
            decoder_role_emb_size = role_emb_size
        if decoder_n_layers is None:
            decoder_n_layers = n_layers
        if decoder_dropout is None:
            decoder_dropout = dropout
        if decoder_max_length is None:
            decoder_max_length = max_length
        if decoder_seq2roles is None:
            decoder_seq2roles = seq2roles
        if decoder_nonlinearity is None:
            decoder_nonlinearity = nonlinearity
        if uniform_trg_length is None:
            uniform_trg_length = uniform_src_length
        if decoder_n_head is None:
            decoder_n_head = n_head
        if decoder_dim_feedforward is None:
            decoder_dim_feedforward = dim_feedforward
        if decoder_has_linear_layer is None:
            decoder_has_linear_layer = has_linear_layer


        if tpr_invert_decoder:
            self.decoder = TensorProductDecoder(hidden_size=decoder_hidden_size, n_fillers=decoder_vocab_size, n_roles=decoder_n_roles, filler_dim=decoder_emb_size, role_dim=decoder_role_emb_size, seq2roles=decoder_seq2roles, has_linear_layer=decoder_has_linear_layer, tpr_enc_to_invert=None, encoding_is_seq=encoding_is_seq)
            tpr_dec_to_invert = self.decoder
        else:
            tpr_dec_to_invert = None

        if encoder is not None:
            # Load a preexisting encoder
            self.encoder = encoder
        else:
            # Create a new encoder
            if architecture in ["SRN", "GRU", "LSTM"]:
                self.encoder = EncoderRNN(vocab_size=vocab_size, hidden_size=hidden_size, recurrent_unit=architecture, emb_size=emb_size, n_layers=n_layers, dropout=dropout)
            elif architecture == "MLP":
                self.encoder = EncoderMLP(vocab_size=vocab_size, hidden_size=hidden_size, emb_size=emb_size, nonlinearity=nonlinearity, n_layers=n_layers, dropout=dropout, uniform_src_length=uniform_src_length)
            elif architecture == "Transformer":
                self.encoder = EncoderTransformer(vocab_size=vocab_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout, n_head=n_head, dim_feedforward=dim_feedforward)
            elif architecture == "TPR":
                self.encoder = TensorProductEncoder(hidden_size=hidden_size, n_fillers=vocab_size, n_roles=n_roles, filler_dim=emb_size, role_dim=role_emb_size, seq2roles=seq2roles, has_linear_layer=has_linear_layer, tpr_dec_to_invert=tpr_dec_to_invert, return_bindings=return_bindings, aggregation=aggregation)
            else:
                raise RuntimeError("Unknown encoder architecture: " + architecture)

        if decoder is not None:
            # Load a preexisting decoder
            self.decoder = decoder
        else:
            # Create a new decoder
            if decoder_architecture in ["SRN", "GRU", "LSTM"]:
                self.decoder = DecoderRNN(vocab_size=decoder_vocab_size, hidden_size=decoder_hidden_size, recurrent_unit=decoder_architecture, emb_size=decoder_emb_size, n_layers=decoder_n_layers, dropout=decoder_dropout, max_length=decoder_max_length, no_eos=no_eos)
            elif decoder_architecture == "MLP":
                self.decoder = DecoderMLP(vocab_size=decoder_vocab_size, hidden_size=decoder_hidden_size, nonlinearity=decoder_nonlinearity, n_layers=decoder_n_layers, dropout=decoder_dropout, uniform_trg_length=uniform_trg_length, no_eos=no_eos)
            elif decoder_architecture == "Transformer":
                self.decoder = DecoderTransformer(vocab_size=decoder_vocab_size, hidden_size=decoder_hidden_size, n_layers=decoder_n_layers, dropout=decoder_dropout, n_head=decoder_n_head, dim_feedforward=decoder_dim_feedforward, max_length=decoder_max_length, bottleneck=bottleneck, no_eos=no_eos)
            elif decoder_architecture == "TPR":
                if tpr_invert_encoder:
                    self.decoder = TensorProductDecoder(hidden_size=decoder_hidden_size, n_fillers=decoder_vocab_size, n_roles=decoder_n_roles, filler_dim=decoder_emb_size, role_dim=decoder_role_emb_size, seq2roles=decoder_seq2roles, has_linear_layer=decoder_has_linear_layer, tpr_enc_to_invert=self.encoder, encoding_is_seq=encoding_is_seq)
                elif tpr_invert_decoder:
                    pass
                else:
                    self.decoder = TensorProductDecoder(hidden_size=decoder_hidden_size, n_fillers=decoder_vocab_size, n_roles=decoder_n_roles, filler_dim=decoder_emb_size, role_dim=decoder_role_emb_size, seq2roles=decoder_seq2roles, has_linear_layer=decoder_has_linear_layer, tpr_enc_to_invert=None, encoding_is_seq=encoding_is_seq)
            else:
                raise RuntimeError("Unknown decoder architecture: " + decoder_architecture)

    def forward(self, batch, tf_ratio=0.0):

        # Encode
        output, hidden = self.encoder(batch)

        # Decode
        log_probs = self.decoder(batch, output, hidden, tf_ratio=tf_ratio)
        topv, topi = log_probs.transpose(0,1).topk(1)
        preds = topi.squeeze(2).tolist()

        # preds is a list of shape (batch_size, max_target_seq_length)
        # probs are shape (max_target_seq_length, batch_size, vocab_size)
        return preds, log_probs





###############################################################################################
# Encoder and decoder architectures
###############################################################################################


# Encoder MLP
class EncoderMLP(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, emb_size=None, nonlinearity="ReLU", n_layers=1, dropout=0.0, uniform_src_length=6):
        super(EncoderMLP, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = uniform_src_length

        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)

        # padding_idx makes it so that the pad token has an embedding
        # of 0 and does not have a gradient computed for it
        if emb_size is None:
            self.emb_size = hidden_size
        else:
            self.emb_size = emb_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=config.PAD_TOKEN)


        # We have to use this setattr() approach to defining the
        # model's layers so that they will all be included under
        # model.parameters() when we pass that to the optimizer 
        layer_index = 0
        setattr(self, "layer" + str(layer_index), nn.Linear(self.emb_size*self.max_length, self.hidden_size))
        for i in range(n_layers-1):
            layer_index += 1
            setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.hidden_size))

        if nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()


    def forward(self, batch):

        # batch: dict with keys input_seq, target_seq, input_seq_lengths, target_seq_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length). 
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        input_seq = batch["input_seq"]
        input_seq_lengths = batch["input_lengths"]

        # shape (seq_length, batch_size)
        input_seq = torch.LongTensor(input_seq).transpose(0,1).to(device=device) 

        hiddens = []

        # Embed input sequences
        # shape (seq_length, batch_size, emb_size)
        emb = self.embedding(input_seq)
        emb = self.dropout(emb)
        
        hidden = emb.transpose(0,1).reshape(-1,1,self.emb_size*self.max_length).transpose(0,1)
        hiddens.append(hidden)

        for layer_index in range(self.n_layers):
            hidden = getattr(self,"layer" + str(layer_index))(hidden)
            hidden = self.nonlinearity(hidden)
            hidden = self.dropout(hidden)
            hiddens.append(hidden)
       
        # hiddens is shape (n_layers, batch_size, hidden_size)
        #     hiddens[-1] is the same as hidden  
        # hidden is shape (1, batch_size, hidden_size)
        return hiddens, hidden


# Decoder MLP
class DecoderMLP(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, nonlinearity="ReLU", n_layers=1, dropout=0.0, uniform_trg_length=6, no_eos=True):
        super(DecoderMLP, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_length = uniform_trg_length
        self.vocab_size = vocab_size

        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)
        
        self.no_eos = no_eos
        if not self.no_eos:
            self.max_length = self.max_length + 1

        # We have to use this setattr() approach to defining the
        # model's layers so that they will all be included under
        # model.parameters() when we pass that to the optimizer 
        layer_index = 0
        for i in range(n_layers-1):
            setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.hidden_size))
            layer_index += 1

        setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.vocab_size*self.max_length))

        if nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()


    # Perform the full forward pass
    def forward(self, batch, output, hidden, tf_ratio=0.0):

        for layer_index in range(self.n_layers-1):
            hidden = getattr(self,"layer" + str(layer_index))(hidden)
            hidden = self.nonlinearity(hidden)
            hidden = self.dropout(hidden)
        
        hidden = getattr(self,"layer" + str(self.n_layers-1))(hidden)
        hidden = hidden.transpose(0,1).reshape(-1, self.max_length, self.vocab_size).transpose(0,1)
        decoder_outputs = F.log_softmax(hidden, dim=2)
        
        # shape (max_target_seq_length, batch_size, vocab_size)
        return decoder_outputs




# Encoder RNN
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, recurrent_unit=None, emb_size=None, n_layers=1, dropout=0.0):
        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.recurrent_unit = recurrent_unit
        
        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)

        if emb_size is None:
            self.emb_size = hidden_size
        else:
            self.emb_size = emb_size


        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
        if recurrent_unit == "SRN":
            self.rnn = nn.RNN(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)
        elif recurrent_unit == "GRU":
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)
        elif recurrent_unit == "LSTM":
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)
        else:
            raise RuntimeError("Unknown recurrent unit: " + recurrent_unit)

    # Creates the initial hidden state
    def init_hidden(self, batch_size):
        if self.recurrent_unit == "SRN" or self.recurrent_unit == "GRU":
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=device)
        elif self.recurrent_unit == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=device))

        return hidden


    # For succesively generating each new output and hidden layer
    def forward(self, batch):

        # batch: dict with keys input_seq, target_seq, input_seq_lengths, target_seq_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length). 
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        input_seq = batch["input_seq"]
        input_seq_lengths = batch["input_lengths"]

        # shape (seq_length, batch_size)
        input_seq = torch.LongTensor(input_seq).transpose(0,1).to(device=device) 

        batch_size = input_seq.shape[1]
        hidden = self.init_hidden(batch_size)

        # Embed input sequences
        # shape (seq_length, batch_size, emb_size)
        emb = self.embedding(input_seq)
        emb = self.dropout(emb)


        # Pack the padded inputs
        emb = nn.utils.rnn.pack_padded_sequence(emb, input_seq_lengths, enforce_sorted=False)

        # Pass through RNN
        output, hidden = self.rnn(emb, hidden)
        
        # Undo the packing
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)


        # output is shape (max_input_seq_length, batch_size, hidden_size)
        #     output[i] corresponds to input_seq[i]  
        # hidden is shape (n_layers, batch_size, hidden_size)
        #     hidden[-1] is the hidden state for the top layer (closest to the output)
        return output, hidden


# Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, recurrent_unit=None, emb_size=None, n_layers=1, dropout=0.0, max_length=30, no_eos=False):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_length = max_length
        self.recurrent_unit = recurrent_unit
        self.vocab_size = vocab_size

        if emb_size == 0:
            self.emb_size = 1
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
            self.embedding.weight = torch.nn.Parameter(torch.zeros_like(self.embedding.weight))
            self.embedding.weight.requires_grad = False
        else:
            if emb_size is None:
                self.emb_size = hidden_size
            else:
                self.emb_size = emb_size
       
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        
        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)
        
        if recurrent_unit == "SRN":
            self.rnn = nn.RNN(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)
        elif recurrent_unit == "GRU":
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)
        elif recurrent_unit == "LSTM":
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropoutp)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

        # If True, the model does not have to predict an EOS token.
        # Instead, we stop it at the correct point.
        self.no_eos = no_eos

    # Perform the full forward pass
    def forward(self, batch, output, hidden, tf_ratio=0.0):

        # batch: dict with keys input_seq, target_seq, input_seq_lengths, target_seq_lengths
        # target_seq is a list of shape = (batch_size, max_seq_length).
        # target_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # target_seq is already formatted as numerical indices for the tokens
        # seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        input_seq = batch["input_seq"]
        target_seq = batch["target_seq"]
        input_seq_lengths = batch["input_lengths"]
        target_seq_lengths = batch["target_lengths"]
 
        # shape (seq_length, batch_size)
        input_seq = torch.LongTensor(input_seq).transpose(0,1).to(device=device)

        # Whether to use teacher forcing
        use_tf = True if random.random() < tf_ratio else False

        if use_tf:
            # Add SOS tokens
            if self.no_eos:
                target_seq = [[config.SOS_TOKEN] + seq[:-1] for seq in target_seq]
            else:
                target_seq = [[config.SOS_TOKEN] + seq for seq in target_seq]
                target_seq_lengths = [length + 1 for length in target_seq_lengths]

            # shape (seq_length, batch_size)
            target_seq = torch.LongTensor(target_seq).transpose(0,1).to(device=device)

            # Embed target sequences (which act as
            # inputs when using teacher forcing)
            emb = self.embedding(target_seq)
            emb = self.dropout(emb)

            # Pack the padded inputs
            emb = nn.utils.rnn.pack_padded_sequence(emb, target_seq_lengths, enforce_sorted=False)

            # Pass through RNN
            output, hidden = self.rnn(emb, hidden)
    
            # Undo the packing
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

            # Convert logits to log probabilities
            decoder_outputs = F.log_softmax(self.out(output), dim=2)

        else:
            if not self.training and not self.no_eos:
                # If we aren't telling the model how long the output should be
                output_length = self.max_length
            else:
                # shape (seq_length, batch_size)
                target_seq = torch.LongTensor(target_seq).transpose(0,1).to(device=device)
                output_length = target_seq.shape[0]

            batch_size = input_seq.shape[1]
            decoder_input = torch.LongTensor([config.SOS_TOKEN] * batch_size).to(device=device)
            decoder_outputs = []

            # Keeps track of whether we have generated an EOS token for
            # every sequence in the batch. If so, we can stop decoding.
            finished = torch.BoolTensor([False] * batch_size).to(device=device)
            
            for di in range(output_length):
                # Embed the input (just one token)
                emb = self.embedding(decoder_input).unsqueeze(0)
                emb = self.dropout(emb)

                # Pass this token through the RNN
                output, hidden = self.rnn(emb, hidden)

                # Get the log probabilities
                output = F.log_softmax(self.out(output[0]), dim=1)
                decoder_outputs.append(output.unsqueeze(0)) 

                # Determine what the predicted output is to serve as
                # the input at the next timestep
                topv, topi = output.data.topk(1)
                decoder_input = topi.view(-1).to(device=device)

                # Update the list of which sequences have had an EOS
                # token generated
                now_finished = decoder_input == config.EOS_TOKEN
                finished[now_finished] = True

                if all(finished) and not self.training:
                    break
                
            decoder_outputs = torch.cat(decoder_outputs, 0)
        
        # shape (max_target_seq_length, batch_size, vocab_size)
        return decoder_outputs



# From official PyTorch implementation
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Encoder
class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, n_layers=1, dropout=0.0, n_head=4, dim_feedforward=None):
        super(EncoderTransformer, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        
        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.positional_encoder = PositionalEncoding(self.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

    # For succesively generating each new output and hidden layer
    def forward(self, batch):

        # batch: dict with keys input_seq, target_seq, input_seq_lengths, target_seq_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length). 
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        input_seq = batch["input_seq"]
        input_seq_lengths = batch["input_lengths"]

        # shape (seq_length, batch_size)
        input_seq = torch.LongTensor(input_seq).transpose(0,1).to(device=device) 
        
        # Specifies which values should be ignored by the attention
        # because they are padding tokens
        src_key_padding_mask = (input_seq == config.PAD_TOKEN).transpose(0,1)

        # Embed input sequences
        # shape (seq_length, batch_size, hidden_size)
        emb = self.embedding(input_seq)
        emb = self.positional_encoder(emb)
        emb = self.dropout(emb)

        memory = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)

        # memory is shape (seq_length, batch_size, hidden_size)
        return memory, memory[0].unsqueeze(0)


# Transformer Decoder
class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size=None, hidden_size=None, n_layers=1, dropout=0.0, n_head=4, dim_feedforward=None, max_length=None, bottleneck=False, no_eos=False):
        super(DecoderTransformer, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.bottleneck = bottleneck
        
        self.dropoutp = dropout
        self.dropout = nn.Dropout(self.dropoutp)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.positional_encoder = PositionalEncoding(self.hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

        # If True, the model does not have to predict an EOS token.
        # Instead, we stop it at the correct point.
        self.no_eos = no_eos

    # For succesively generating each new output and hidden layer
    def forward(self, batch, memory, output, tf_ratio=0.0):

        # batch: dict with keys input_seq, target_seq), (input_seq_lengths, target_seq_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length). 
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        input_seq = batch["input_seq"]
        target_seq = batch["target_seq"]
        input_seq_lengths = batch["input_lengths"]
        target_seq_lengths = batch["target_lengths"]

        # shape (seq_length, batch_size)
        input_seq = torch.LongTensor(input_seq).transpose(0,1).to(device=device)

        # Specifies which values in the memory should be ignored by the attention
        # because they are padding tokens
        if self.bottleneck:
            # Only looking at the first token of the input. Previously done via an
            # attention mask, but that doesn't work when swapping in a TPE encoder
            memory = output
            memory_key_padding_mask = None

        else:
            # Attend to all input elements
            memory_key_padding_mask = (input_seq == config.PAD_TOKEN).transpose(0,1)

        # Whether to use teacher forcing
        use_tf = True if random.random() < tf_ratio else False

        if use_tf:
            # Add SOS tokens
            if self.no_eos:
                
                # As an input to the model, we remove the last element, since the
                # inputs are shifted over by one relative to the outputs (so the
                # last output never serves as an input)
                target_seq = [[config.SOS_TOKEN] + seq[:-1] for seq in target_seq]
            else:
                target_seq = [[config.SOS_TOKEN] + seq for seq in target_seq]
                target_seq_lengths = [length + 1 for length in target_seq_lengths]

            # shape (seq_length, batch_size)
            target_seq = torch.LongTensor(target_seq).transpose(0,1).to(device=device)

            # Specifies which values in the target should be ignored by the attention
            # because they are padding tokens
            tgt_key_padding_mask = (target_seq == config.PAD_TOKEN).transpose(0,1)

            # This mask ensures that each word only attends to words on its left
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_key_padding_mask.shape[1]).to(device=device)

            # Embed target sequences (which act as
            # inputs when using teacher forcing)
            emb = self.embedding(target_seq)
            emb = self.positional_encoder(emb)
            emb = self.dropout(emb)

            output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

            output = F.log_softmax(self.out(output), dim=2)
        else:
            if not self.training and not self.no_eos:
                # If we aren't telling the model how long the output should be
                output_length = self.max_length
            else:
                # shape (seq_length, batch_size)
                target_seq = torch.LongTensor(target_seq).transpose(0,1).to(device=device)
                output_length = target_seq.shape[0]

            batch_size = input_seq.shape[1]
            decoder_input_list = [[config.SOS_TOKEN]] * batch_size
            decoder_outputs = []

            # Keeps track of whether we have generated an EOS token for
            # every sequence in the batch. If so, we can stop decoding.
            finished = torch.BoolTensor([False] * batch_size).to(device=device)

            for di in range(output_length):
                decoder_input = torch.LongTensor(decoder_input_list).to(device=device)
                
                # Specifies which values in the target should be ignored by the attention
                # because they are padding tokens
                tgt_key_padding_mask = (decoder_input == config.PAD_TOKEN)

                # This mask ensures that each word only attends to words on its left
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_key_padding_mask.shape[1]).to(device=device)
                
                # Embed the input (the SOS token plus the whole prefix decoded so far)
                emb = self.embedding(decoder_input).transpose(0,1)
                emb = self.positional_encoder(emb)
                emb = self.dropout(emb)

                output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

                # Get the log probabilities
                output = F.log_softmax(self.out(output[-1]), dim=1)
                decoder_outputs.append(output.unsqueeze(0))

                # Determine what the predicted output is, to serve as
                # the input at the next timestep
                topv, topi = output.data.topk(1)
                next_input = topi.view(-1).tolist()
                decoder_input_list = [decoder_input_list[index] + [next_input[index]] for index in range(len(decoder_input_list))]

                # Update the list of which sequences have had an EOS
                # token generated
                now_finished = next_input == config.EOS_TOKEN
                finished[now_finished] = True

                if all(finished) and not self.training:
                    break

            decoder_outputs = torch.cat(decoder_outputs, 0)
            output = decoder_outputs

        # shape (max_target_seq_length, batch_size, vocab_size)
        return output


class OuterProduct(nn.Module):
    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, input1, input2):
        einsum = torch.einsum('blf,blr->blfr', (input1, input2))
        outputs = einsum.view(einsum.shape[0], einsum.shape[1], -1)

        summed_outputs = torch.sum(outputs, dim=1).unsqueeze(0)

        return outputs.transpose(0,1), summed_outputs

class AveragedOuterProduct(nn.Module):
    def __init__(self):
        super(AveragedOuterProduct, self).__init__()

    def forward(self, input1, input2):
        einsum = torch.einsum('blf,blr->blfr', (input1, input2))
        outputs = einsum.view(einsum.shape[0], einsum.shape[1], -1)

        summed_outputs = torch.sum(outputs, dim=1).unsqueeze(0)

        mask = (outputs != 0).float()
        mask = torch.sum(mask, dim=1).unsqueeze(0)
        mean_outputs = summed_outputs / mask

        return outputs.transpose(0,1), mean_outputs

# Defines the tensor product, used in tensor product representations
class SumFlattenedOuterProduct(nn.Module):
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()
           
    def forward(self, input1, input2):
        sum_outer_product = torch.bmm(input1.transpose(1,2), input2)
        flattened_sum_outer_product = sum_outer_product.view(sum_outer_product.size()[0],-1).unsqueeze(0)
        return None, flattened_sum_outer_product

# Tensor Product Encoder
class TensorProductEncoder(nn.Module):
    def __init__(self, hidden_size=None, n_fillers=None, n_roles=None, filler_dim=None, role_dim=None, seq2roles=None, has_linear_layer=True, tpr_dec_to_invert=None, return_bindings=False, aggregation="sum", role_learning=False, role_assigner_kwargs=None):
        super(TensorProductEncoder, self).__init__()
	
        self.role_learning = role_learning
        self.hidden_size = hidden_size

        # Vocab size for the fillers and roles
        self.n_fillers = n_fillers
        self.n_roles = n_roles

        # Embedding size for fillers and roles
        self.filler_dim = filler_dim
        self.role_dim = role_dim

        # Embeddings for fillers and roles
        # The padding_idx means that the padding token, 0, will be embedded
        # as a vector of all zeroes that get no gradient
        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim, padding_idx=config.PAD_TOKEN)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim, padding_idx=config.PAD_TOKEN)

        # Function that takes in the batch and returns
        # the role ids for the sequences in the batch
        self.seq2roles = seq2roles

        if self.role_learning:
            assert role_assigner_kwargs is not None
            self.role_assigner = RoleLearner(**role_assigner_kwargs)

        self.return_bindings = return_bindings

        # Create a layer that will bind together the fillers
        # and roles and then aggregate the filler/role pairs.
        # The default is to use the tensor product as the binding
        # operation and elementwise sum as the aggregation operation.
        # These are implemented together with SumFlattenedOuterProduct()
        if aggregation == "sum":
            if return_bindings:
                self.bind_and_aggregate_layer = OuterProduct()
            else:
                self.bind_and_aggregate_layer = SumFlattenedOuterProduct()
        elif aggregation == "mean":
            self.bind_and_aggregate_layer = AveragedOuterProduct()

        # Final linear layer that takes in the tensor product representation
        # and outputs an encoding of size self.hidden_size
        self.has_linear_layer = has_linear_layer
        if self.has_linear_layer:
            self.output_layer = nn.Linear(self.filler_dim * self.role_dim, self.hidden_size)

        if tpr_dec_to_invert is not None:
            self.role_embedding.weight[1:] = nn.Parameter(torch.pinverse(tpr_dec_to_invert.role_embedding.weight[1:]).transpose(0,1))
            self.filler_embedding.weight = tpr_dec_to_invert.filler_embedding.weight

            if self.has_linear_layer:
                raise RuntimeError("Cannot invert a TensorProductDecoder that has a linear layer")


    # Function for a forward pass through this layer. Takes a list of fillers and 
    # a list of roles and returns a single vector encoding it.
    def forward(self, batch):

        # batch: dict with keys input_seq, target_seq, input_seq_lengths, target_seq_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length).
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # input_seq_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        # Establish what the fillers and roles are
        if "fillers" in batch:
            fillers = batch["fillers"]
            fillers = torch.LongTensor(fillers).to(device=device)
            roles = batch["roles"]
            roles = torch.LongTensor(roles).to(device=device)
            roles = self.role_embedding(roles)
        else:
            fillers, role_lists = self.seq2roles.assign_roles_batch(batch)
            fillers = torch.LongTensor(fillers).to(device=device)
            if self.role_learning:
                roles, role_predictions = self.role_assigner(batch)
                # Store the role predictions for regularization in training.py
                self.role_predictions = role_predictions
            else:
                roles = None
                for role_list in role_lists:

                    sub_roles = torch.LongTensor(role_list).to(device=device)
                    sub_roles = self.role_embedding(sub_roles)

                    if roles is None:
                        roles = sub_roles
                    else:
                        roles = roles + sub_roles

        # Embed the fillers
        fillers = self.filler_embedding(fillers)

        # Bind and aggregate the fillers and roles
        outputs, hidden = self.bind_and_aggregate_layer(fillers, roles)

        # Pass the encoding through the final linear layer
        if self.has_linear_layer:
            hidden = self.output_layer(hidden)

            if self.return_bindings:
                outputs = self.output_layer(outputs)#.transpose(0,1)
 
        # hidden is shape (1, batch_size, hidden_size)
        # outputs is shape (seq_len, batch_size, hidden_size)
        return outputs, hidden

# Tensor Product Decoder
class TensorProductDecoder(nn.Module):
    def __init__(self, hidden_size=None, n_fillers=None, n_roles=None, filler_dim=None, role_dim=None, seq2roles=None, has_linear_layer=False, tpr_enc_to_invert=None, encoding_is_seq=False):
        super(TensorProductDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.n_fillers = n_fillers
        self.n_roles = n_roles

        self.filler_dim = filler_dim
        self.role_dim = role_dim

        # Embeddings for fillers and roles
        # The padding_idx means that the padding token, 0, will be embedded
        # as a vector of all zeroes that get no gradient
        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim, padding_idx=config.PAD_TOKEN)

        self.seq2roles = seq2roles

        self.has_linear_layer = has_linear_layer
        if self.has_linear_layer:
            self.pre_linear = None
            self.input_layer = nn.Linear(self.hidden_size, self.filler_dim*self.role_dim)

        if tpr_enc_to_invert is not None:
            role_emb_matrix = self.role_embedding.weight.detach()
            role_emb_matrix[3:] = torch.pinverse(tpr_enc_to_invert.role_embedding.weight[3:]).transpose(0,1)
            self.role_embedding.load_state_dict({'weight':role_emb_matrix})
            self.filler_embedding.weight = nn.Parameter(tpr_enc_to_invert.filler_embedding.weight.detach())

            if self.has_linear_layer:
                self.pre_linear = nn.Parameter(-1*tpr_enc_to_invert.output_layer.bias)
                self.input_layer.bias = nn.Parameter(torch.zeros_like(self.input_layer.bias))
                self.input_layer.weight = nn.Parameter(torch.pinverse(tpr_enc_to_invert.output_layer.weight))


        self.encoding_is_seq = encoding_is_seq

    def forward(self, batch, output, hidden, tf_ratio=None):

        if self.encoding_is_seq:
            hidden = torch.sum(output, dim=0).unsqueeze(0)

        if self.has_linear_layer:
            if self.pre_linear is not None:
                hidden = hidden + self.pre_linear
            hidden = self.input_layer(hidden)

        # Reshape the hidden state into a matrix
        # to be viewed as a TPR
        hidden = hidden.transpose(0,1).transpose(1,2)
        hidden = hidden.view(-1,self.filler_dim,self.role_dim)

        # Determine the roles to be unbound from
        if "roles" in batch:
            # This form of passing in the roles
            # just includes one role scheme
            roles = batch["roles"]
        else:
            _, roles = self.seq2roles.assign_roles_batch(batch)

            # Having the role scheme be a summation of multiple schemes is only supported
            # for the TensorProductEncoder. For the decoder, we always assume that the list
            # of role schemes only has one element - hence the use of [0]
            roles = roles[0]
        
        roles = torch.LongTensor(roles).to(device=device)

        # Get the role unbinding vectors, and multiply them
        # by the TPR to get the guess for the filler embedding
        roles_emb = self.role_embedding(roles)
        filler_guess = torch.bmm(roles_emb, hidden.transpose(1,2))

        # Find the distance between the filler embedding guess
        # and all actual filler vectors, then use those distance
        # as logits that are the input to the NLLLoss
        filler_guess_orig_shape = filler_guess.shape
        filler_guess = filler_guess.reshape(-1, filler_guess_orig_shape[-1]).unsqueeze(1)
        comparison = self.filler_embedding.weight.unsqueeze(0)


        dists = (filler_guess - comparison).pow(2).sum(dim=2).pow(0.5).reshape(filler_guess_orig_shape[0], filler_guess_orig_shape[1], -1)
      
        
        dists = -1*dists.transpose(0,1)

        dists = F.log_softmax(dists, dim=2)
        return dists


class RoleLearner(nn.Module):
    def __init__(
            self,
            num_roles,
            encoding_model,
            role_embedding_dim,
            softmax_roles=True,
            center_roles_at_one=True,
            snap_one_hot_predictions=False,
            relative_role_prediction_function=None,
    ):
        super(RoleLearner, self).__init__()

        self.encoding_model = encoding_model
        self.num_roles = num_roles
        self.snap_one_hot_predictions = snap_one_hot_predictions
        self.softmax_roles = softmax_roles
        if softmax_roles:
            logging.info("Use softmax for role predictions")
            # The output of role_weight_predictions is shape
            # (batch_size, sequence_length, num_roles)
            # We want to softmax across the roles so set dim=2
            self.softmax = nn.Softmax(dim=2)

        self.relative_role_prediction_function = relative_role_prediction_function
        # TODO this 768 should be dynamic based on encoding_model or at least passed in
        encoding_model_dim = 768
        if self.relative_role_prediction_function is None:
            self.role_weight_predictions = nn.Linear(encoding_model_dim, num_roles)
        elif self.relative_role_prediction_function == 'concat':
            self.role_weight_predictions = nn.Linear(encoding_model_dim*2, num_roles)
        elif self.relative_role_prediction_function == 'elementwise':
            self.role_weight_predictions = nn.Linear(encoding_model_dim, num_roles)
        elif self.relative_role_prediction_function == 'concat+elementwise':
            self.role_weight_predictions = nn.Linear(encoding_model_dim*3, num_roles)
        else:
            raise Exception('relative_role_prediction_function {} is not supported'.format(self.relative_role_prediction_function))

        self.role_embedding_dim = role_embedding_dim
        self.role_matrix = nn.Parameter(torch.zeros(self.num_roles, self.role_embedding_dim))  # role embeddings
        nn.init.xavier_uniform_(self.role_matrix, gain=1.414)

        self.center_roles_at_one = center_roles_at_one

    def forward(self, batch):
        """
        :return: A tensor of size (batch_size, sequence_length, role_embedding_dim) with the role
            embeddings for the input filler_tensor.
        """

        encodings, _ = self.encoding_model(batch)

        if self.relative_role_prediction_function is None:
            pass
        else:
            input_seq_tensor = torch.zeros(len(batch['input_lengths']), max(batch['input_lengths']), dtype=torch.long)
            mask = torch.arange(max(batch['input_lengths'])) < torch.tensor(batch['input_lengths'])[:, None]
            input_seq_tensor[mask] = torch.tensor(np.concatenate(batch['input_seq']))
            mask_id = self.encoding_model.mask_id
            mask_token_index = input_seq_tensor == mask_id
            mask_encodings = encodings[mask_token_index]
            mask_encodings = mask_encodings[:, None, :].expand(mask_encodings.shape[0], encodings.shape[1],
                                                               mask_encodings.shape[1])
            if self.relative_role_prediction_function == 'concat':
                # Expand the mask encodings to the proper shape to concat with encodings
                encodings = torch.cat((encodings, mask_encodings), dim=-1)
            elif self.relative_role_prediction_function == 'elementwise':
                encodings = torch.mul(encodings, mask_encodings)
            elif self.relative_role_prediction_function == 'concat+elementwise':
                combined = torch.mul(encodings, mask_encodings)
                encodings = torch.cat((encodings, mask_encodings, combined), dim=-1)
            else:
                raise Exception(
                    'relative_role_prediction_function {} is not supported'.format(self.relative_role_prediction_function))

        role_predictions = self.role_weight_predictions(encodings)

        if self.softmax_roles:
            role_predictions = self.softmax(role_predictions)
        # role_predictions is size (batch_size, sequence_length, num_roles)

        # Normalize the embeddings. This is important so that role attention is not overruled by
        # embeddings `` with different orders of magnitude.
        role_matrix = self.role_matrix / torch.norm(self.role_matrix, dim=1, keepdim=True)
        # role_matrix is size (num_roles, role_embedding_dim)

        # During evaluation, we want to snap the role predictions to a one-hot vector
        if self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(role_predictions, 2),
                                                         self.num_roles)
            roles = torch.matmul(one_hot_predictions, self.role_matrix)
        else:
            roles = torch.matmul(role_predictions, self.role_matrix)

        # roles is size (batch_size, sequence_length, role_embedding_dim)
        if self.center_roles_at_one:
            roles = roles + 1

        return roles, role_predictions

    @staticmethod
    def get_regularization_loss(role_predictions,
                                softmax_roles,
                                one_hot_temperature=1,
                                one_hot_regularization_weight=1,
                                l2_norm_regularization_weight=1,
                                unique_role_regularization_weight=1,
                                ):
        batch_size = role_predictions.shape[0]

        if softmax_roles:
            # For RoleLearningTensorProductEncoder, we encourage one hot vector weight predictions
            # by regularizing the role_predictions by `w * (1 - w)`
            one_hot_reg = torch.sum(role_predictions * (1 - role_predictions))
        else:
            one_hot_reg = torch.sum((role_predictions ** 2) * (1 - role_predictions) ** 2)
        one_hot_loss = one_hot_temperature * one_hot_reg / batch_size

        if softmax_roles:
            l2_norm = -torch.sum(role_predictions * role_predictions)
        else:
            l2_norm = (torch.sum(role_predictions ** 2) - 1) ** 2
        l2_norm_loss = one_hot_temperature * l2_norm / batch_size

        # We also want to encourage the network to assign each filler in a sequence to a
        # different role. To encourage this, we sum the vector predictions across a sequence
        # (call this vector w) and add `(w * (1 - w))^2` to the loss function.
        exclusive_role_vector = torch.sum(role_predictions, 1)
        unique_role_loss = one_hot_temperature * torch.sum(
            (exclusive_role_vector * (1 - exclusive_role_vector)) ** 2) / batch_size
        return one_hot_regularization_weight * one_hot_loss, \
               l2_norm_regularization_weight * l2_norm_loss, \
               unique_role_regularization_weight * unique_role_loss





# BERT encoder
class BERTEncoder(nn.Module):
    def __init__(self, bert_model=None, bert_tokenizer=None, layer=None, mask=False, indices=False, name="bert"):
        super(BERTEncoder, self).__init__()

        self.bert_tokenizer = bert_tokenizer
        if name == "roberta":
            self.mask_id = bert_tokenizer.encode("<mask>")[1]
        else:
            self.mask_id = bert_tokenizer.encode("[MASK]")[1]
        self.bert_model = bert_model

        self.bert_model.eval()

        self.layer = layer
        self.mask = mask
        self.indices = indices
        self.name = name


    def forward(self, batch):
        if self.name == "roberta":
            if self.mask:
                input_sentences = [s.replace("[MASK]", "<mask>") for s in batch["masked_sentence"]]
            else:
                input_sentences = [s.replace("[MASK]", "<mask>") for s in batch["sentence"]]
        else:
            if self.mask:
                input_sentences = batch["masked_sentence"]
            else:
                input_sentences = batch["sentence"]

        encoding = self.bert_tokenizer(input_sentences, padding="longest")

        if "token_type_ids" in encoding:
            bert_outputs = self.bert_model(input_ids=torch.LongTensor(encoding["input_ids"]).to(device), attention_mask=torch.LongTensor(encoding["attention_mask"]).to(device), token_type_ids=torch.LongTensor(encoding["token_type_ids"]).to(device), output_hidden_states=True)
        else:
            bert_outputs = self.bert_model(input_ids=torch.LongTensor(encoding["input_ids"]).to(device), attention_mask=torch.LongTensor(encoding["attention_mask"]).to(device), output_hidden_states=True)

        bert_hidden = bert_outputs["hidden_states"]

        layer_embs = bert_hidden[self.layer].detach()

        if self.mask:
            indices = [enc.index(self.mask_id) for enc in encoding["input_ids"]]
        elif self.indices:
            # Adding 1 because of CLS token at the start
            indices = [ind + 1 for ind in batch["word_index"]]

        word_embs = []


        for sentence_index, layer_emb in enumerate(layer_embs):

            if self.indices and len(layer_emb) != 2 + len(input_sentences[sentence_index].split()):
                logging.info("Some words are split into multiple tokens: Word indices may be off")
                # Ah yes, 15/0
                15/0

            word_emb = layer_emb[indices[sentence_index]].detach()
            word_embs.append(word_emb.detach().unsqueeze(0))

        hidden = torch.cat(word_embs, 0)
        hidden = hidden.unsqueeze(0)

        # output is shape (max_input_seq_length, batch_size, hidden_size)
        # hidden is shape (1, batch_size, hidden_size)
        return layer_embs, hidden



# BERT decoder
class BERTDecoder(nn.Module):
    def __init__(self, bert_model=None, layer=None, name="bert"):
        super(BERTDecoder, self).__init__()

        self.bert_model = bert_model

        self.bert_model.eval()

        self.layer = layer

        self.name = name

    def forward(self, batch, output, hidden, tf_ratio=None):
        if self.name == "bert":
            for i in range(self.layer, 12):
                output = self.bert_model.bert.encoder.layer[i](output)[0]
            output = self.bert_model.cls(output)
        elif self.name == "roberta":
            for i in range(self.layer, 12):
                output = self.bert_model.roberta.encoder.layer[i](output)[0]
            output = self.bert_model.lm_head(output)
        elif self.name == "distilbert":
            for i in range(self.layer, 6):
                output = self.bert_model.distilbert.transformer.layer[i](output, attn_mask=torch.ones(output.shape[0], output.shape[1]))[0]

            output = self.bert_model.vocab_transform(output)
            output = self.bert_model.vocab_layer_norm(output)
            output = self.bert_model.vocab_projector(output)
        elif self.name == "albert":
            for i in range(self.layer, 12):
                output = self.bert_model.albert.encoder.albert_layer_groups[0](output, head_mask=[None for _ in range(12)])[0]
            output = self.bert_model.predictions(output)

        output = F.log_softmax(output, dim=-1)

        return output


# BERT encoder/decoder
class BERTEncoderDecoder(nn.Module):
    def __init__(self, layer=None, mask=False, indices=False, name="bert"):
        super(BERTEncoderDecoder, self).__init__()

        if name == "bert":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
            self.mask_id = self.bert_tokenizer.encode("<mask>")[1]
        elif name == "distilbert":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
            self.bert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(device)
            self.mask_id = self.bert_tokenizer.encode("[MASK]")[1]
        elif name == "roberta":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
            self.bert_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
            self.mask_id = self.bert_tokenizer.encode("[MASK]")[1]
        elif name == "albert":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_fast=True)
            self.bert_model = AlbertForMaskedLM.from_pretrained('albert-base-v2').to(device)
            self.mask_id = self.bert_tokenizer.encode("[MASK]")[1]


        self.bert_model.eval()

        self.layer = layer
        self.mask = mask
        self.indices = indices

        self.encoder = BERTEncoder(bert_model=self.bert_model, bert_tokenizer=self.bert_tokenizer, layer=self.layer, mask=self.mask, indices=self.indices, name=name)
        self.decoder = BERTDecoder(bert_model=self.bert_model, layer=self.layer, name=name)


    def forward(self, batch, decode=False):
        output, hidden = self.encoder(batch)
        log_probs = self.decoder(batch, output, hidden)

        return _, log_probs





















if __name__ == "__main__":
    src_idx2token = {}
    role_token2idx = {}
    for i in range(10):
        src_idx2token[i] = i
        role_token2idx["0_LTR" + str(i)] = i

    ltr_seq2roles = RoleAssigner("ltr", src_idx2token=src_idx2token)
    ltr_seq2roles.filler_token2idx = src_idx2token
    ltr_seq2roles.role_token2idx = role_token2idx

    enc = TensorProductEncoder(hidden_size=19, n_fillers=10, n_roles=10, filler_dim=3, role_dim=5, has_linear_layer=True, seq2roles=ltr_seq2roles)

    batch = {}
    batch["input_seq"] = [[2,1,2,3]]
    batch["input_lengths"] = [4]

    print(enc(batch))

    batch = {}
    batch["input_seq"] = [[2,1,2,3], [3,3,3,3]]
    batch["input_lengths"] = [4,4]


    print(enc(batch))

