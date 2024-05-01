

import argparse

discover_parser = argparse.ArgumentParser(add_help=False)

discover_parser.add_argument("--json_args", help="json file of saved model configuration to load", type=str, default=None)

# Arguments defining the architecture
discover_parser.add_argument("--architecture", help="model type (SRN, GRU, LSTM)", type=str, default=None)
discover_parser.add_argument("--hidden_size", help="hidden size", type=int, default=120)
discover_parser.add_argument("--emb_size", help="embedding size", type=int, default=10)
discover_parser.add_argument("--role_emb_size", help="role embedding size", type=int, default=None)
discover_parser.add_argument("--role_scheme", help="role scheme", type=str, default=None)
discover_parser.add_argument("--n_layers", help="number of layers", type=int, default=1)
discover_parser.add_argument("--max_length", help="maximum sequence length", type=int, default=10)
discover_parser.add_argument("--nonlinearity", help="nonlinearity to apply between MLP layers", type=str, default="ReLU")
discover_parser.add_argument("--n_head", help="number of attention heads", type=int, default=4)
discover_parser.add_argument("--dim_feedforward", help="size of a Transformer's feedforward layer", type=int, default=None)
discover_parser.add_argument("--has_linear_layer", help="in a TPR encoder, whether there is a linear layer at the end", action='store_true')
discover_parser.add_argument("--aggregation", help="aggregation function for a TPR encoder", type=str, default="sum")
discover_parser.add_argument("--tpr_invert_decoder", help="whether the TPR encoder should be created by inverting its decoder", action='store_true')
discover_parser.add_argument("--return_bindings", help="whether the TPR encoder should return the individual filler-role bindings, as well as the sum of the filler-role bindings", action='store_true')
discover_parser.add_argument("--no_train_encoder", help="don't train the encoder; just stick with its randomly-initialized values", action='store_true')

discover_parser.add_argument("--decoder_architecture", help="decoder model type, if different from encoder (SRN, GRU, LSTM)", type=str, default=None)
discover_parser.add_argument("--decoder_hidden_size", help="decoder hidden size, if different from encoder", type=int, default=None)
discover_parser.add_argument("--decoder_emb_size", help="decoder embedding size, if different from encoder", type=int, default=None)
discover_parser.add_argument("--decoder_role_emb_size", help="decoder role embedding size, if different from encoder", type=int, default=None)
discover_parser.add_argument("--decoder_role_scheme", help="role scheme for the decoder, if different from the encoder", type=str, default=None)
discover_parser.add_argument("--decoder_n_layers", help="number of layers in the decoder, if different from the encoder", type=int, default=None)
discover_parser.add_argument("--decoder_max_length", help="maximum sequence length for the decoder, if different from the encoder", type=int, default=None)
discover_parser.add_argument("--decoder_nonlinearity", help="nonlinearity to apply between MLP decoder layers, if different from the encoder", type=str, default=None)
discover_parser.add_argument("--decoder_n_head", help="number of attention heads for the decoder, if different from the encoder", type=int, default=None)
discover_parser.add_argument("--decoder_dim_feedforward", help="size of a decoder Transformer's feedforward layer, if different from the encoder", type=int, default=None)
discover_parser.add_argument("--bottleneck", help="have the Transformer decoder only attend to the first element of the input", action='store_true')
discover_parser.add_argument("--decoder_has_linear_layer", help="in a TPR decoder, whether there is a linear layer at the start", action='store_true')
discover_parser.add_argument("--tpr_invert_encoder", help="whether the TPR decoder should be created by inverting its encoder", action='store_true')
discover_parser.add_argument("--encoding_is_seq", help="whether the encoding used by the model is a sequence of vectors (as opposed to a single vector). E.g., Transformers use a sequence of vectors; an RNN without attention uses a single vector", action='store_true')
discover_parser.add_argument("--no_train_decoder", help="don't train the decoder; just stick with its randomly-initialized values", action='store_true')


# DISCOVER arguments
discover_parser.add_argument("--discover_hidden_size", help="DISCOVER hidden size", type=int, default=60)
discover_parser.add_argument("--discover_emb_size", help="DISCOVER model filler embedding size", type=int, default=None)
discover_parser.add_argument("--discover_role_emb_size", help="DISCOVER model role embedding size", type=int, default=None)
discover_parser.add_argument("--discover_role_scheme", help="role scheme for a DISCOVER model", type=str, default=None)
discover_parser.add_argument("--discover_filler_role_list_from_file", help="load the fillers and roles from a file", action='store_true')
discover_parser.add_argument("--discover_filler_min_count", help="the minimum number of occurrences that a filler needs to have in a .filler_list file to be included in the filler vocab (i.e., not UNKed)", type=int, default=0)
discover_parser.add_argument("--discover_filler_file_int", help="the fillers loaded from a file should be viewed as ints", action='store_true')
discover_parser.add_argument("--discover_role_min_count", help="the minimum number of occurrences that a role needs to have in a .role_list file to be included in the role vocab (i.e., not UNKed)", type=int, default=0)
discover_parser.add_argument("--discover_has_linear_layer", help="in a TPR encoder, whether there is a linear layer at the end", action='store_true')
discover_parser.add_argument("--discover_init_filler_embedding", help="in a TPR encoder, initialize its filler embeddings with the target model's filler embeddings", action='store_true')
discover_parser.add_argument("--discover_no_train_filler_embedding", help="in a TPR encoder, don't train its filler embeddings", action='store_true')
discover_parser.add_argument("--linreg_truncate_filler_examples", help="number of examples to use in computing the SVD of the filler embeddings", type=int, default=1000)
discover_parser.add_argument("--linreg_truncate_role_examples", help="number of examples to use in computing the SVD of the role embeddings", type=int, default=1000)
discover_parser.add_argument("--linreg_full_rank", help="don't do any rank reduction in a linear regression model", action='store_true')

# Arguments defining the training
discover_parser.add_argument("--patience", help="number of checkpoints without improvement to wait before early stopping", type=int, default=5)
discover_parser.add_argument("--learning_rate_decay_patience", help="when the number of learning rate decays is greater than this, halt training", type=int, default=2)
discover_parser.add_argument("--n_epochs", help="maximum number of training epochs", type=int, default=1000)
discover_parser.add_argument("--eval_every", help="number of batches to train on in between each evaluation", type=int, default=300)
discover_parser.add_argument("--training_size", help="number of training examples to use", type=int, default=None)
discover_parser.add_argument("--dropout", help="dropout percentage", type=float, default=0.1)
discover_parser.add_argument("--decoder_dropout", help="dropout percentage for the decoder, if different from the encoder", type=float, default=None)
discover_parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
discover_parser.add_argument("--no_train", help="don't train, just evaluate", action='store_true')
discover_parser.add_argument("--input_sos", help="add an SOS token at the start of the input sequence", action='store_true')
discover_parser.add_argument("--no_eos", help="don't append the EOS token to target sequences", action='store_true')
discover_parser.add_argument("--get_padding_loss", help="backpropagate loss for padding tokens", action='store_true')
discover_parser.add_argument("--task_type", help="type of task (seq2seq, encoding, encoding_seq)", type=str, default=None)
discover_parser.add_argument("--task", help="training task. Note: only used in analogies; for training models, the task is specified by providing input/output pairs", type=str, default=None)
discover_parser.add_argument("--lr_to_tpe", help="convert linear regression model to a Tensor Product Encoder", action='store_true')

# Evaluations to run after doing DISCOVER
discover_parser.add_argument("--eval_swapping", help="run evaluations of swapping accuracy and agreement", action='store_true')
discover_parser.add_argument("--eval_nearest_neighbor_test", help="run nearest neighbor evaluation among just the test set elements", action='store_true')
discover_parser.add_argument("--eval_nearest_neighbor_all", help="run nearest neighbor evaluation with all datasets (training, validation, and test) as candidates", action='store_true')
discover_parser.add_argument("--eval_tvd", help="run total variation distance evaluation", action='store_true')
discover_parser.add_argument("--eval_explained_variance", help="run explained variance evaluation", action='store_true')

# Padding the data
discover_parser.add_argument("--pad_src_left", help="pad source sequences on the left, instead of the right", action='store_true')
discover_parser.add_argument("--pad_trg_left", help="pad target sequences on the left, instead of the right", action='store_true')
discover_parser.add_argument("--uniform_src_length", help="sequence length for all source sequences", type=int, default=None)
discover_parser.add_argument("--uniform_trg_length", help="sequence length for all target sequences", type=int, default=None)

# Directories
discover_parser.add_argument("--data_dir", help="data directory", type=str, default="data/")
discover_parser.add_argument("--weight_dir", help="model weight directory", type=str, default="weights/")
discover_parser.add_argument("--log_dir", help="directory for logs", type=str, default="logs/")

# Data loading
discover_parser.add_argument("--headers", help="input files have headers", action='store_true')
discover_parser.add_argument("--batch_size", help="batch size", type=int, default=32)
discover_parser.add_argument("--batches_per_buffer", help="number of batches to store in the buffer", type=int, default=1000)
discover_parser.add_argument("--shuffle", help="whether to shuffle batches within the buffer", action="store_true")
discover_parser.add_argument("--stream", help="use streaming in loading the data", action="store_true")
discover_parser.add_argument("--encode_on_the_fly", help="in running a DISCOVER model, generate the target encodings on the fly", action="store_true")

# BERT-related arguments
discover_parser.add_argument("--bert_layer", help="BERT layer from which the target encodings are drawn", type=int, default=0)
discover_parser.add_argument("--filler_indices", help="the fillers are BERT token indices", action="store_true")
discover_parser.add_argument("--bert_mask", help="we are extracting BERT embeddings of masked tokens", action="store_true")
discover_parser.add_argument("--bert_indices", help="we are extracting BERT embeddings of tokens at specified indices", action="store_true")
discover_parser.add_argument("--input_seq_from_bert", help="the input sequence is the tokens derived from BERT", action="store_true")

# Miscellaneous
discover_parser.add_argument("--random_seed", help="random seed", type=int, default=None)
discover_parser.add_argument("--dataset_prefix", help="prefix for the training, validation, and test dataset", type=str, default=None)
discover_parser.add_argument("--dataset_prefix_to_encode", help="prefix for the datasets to generate encodings for (if different from dataset_prefix)", type=str, default=None)
discover_parser.add_argument("--analogy_prefix", help="prefix for analogy file", type=str, default=None)
discover_parser.add_argument("--analogy_scale", help="assume that aggregation was mean-based, not sum-based, and scale analogies accordingly", action="store_true")
discover_parser.add_argument("--model_name", help="name for saving model weights", type=str, default=None)
discover_parser.add_argument("--encoder_model_name", help="name for an encoder to load (when different from model_name)", type=str, default=None)
discover_parser.add_argument("--decoder_model_name", help="name for a decoder to load (when different from model_name)", type=str, default=None)
discover_parser.add_argument("--memoize", help="use memoization inside the role assigner", action="store_true")

discover_parser.add_argument("--prnt", help="assume that aggregation was mean-based, not sum-based, and scale analogies accordingly", action="store_true")

# ROLE arguments
discover_parser.add_argument("--role_learning", help="Enable the role learning module", action="store_true")
discover_parser.add_argument("--num_roles", help="The number of learnable roles", type=int, default=20)
discover_parser.add_argument("--softmax_roles", help="Whether to apply softmax after role predicitons", action="store_false")
discover_parser.add_argument("--center_roles_at_one", help="Whether to center the role vectors at 1", action="store_false")
discover_parser.add_argument("--role_learning_hyper", help="How much to modulate the one-hot regularization by", type=float, default=1.)
discover_parser.add_argument("--relative_role_prediction_function", help="The function used to combined encodings before role prediction. One of [None, concat, elementwise, concat+elementwise]", type=str, default=None)

