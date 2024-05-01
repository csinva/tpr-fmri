
import os
import logging
import numpy as np
import json

from models import *
from token_dictionary import *
from utils import *
from training import *
from evaluation import *
from role_assignment_functions import *
import config

import argparse
from discover_argparser import discover_parser


###################################################################
# Establish training arguments
###################################################################

parser = argparse.ArgumentParser(parents=[discover_parser], description="Arguments governing model architectures and training")
args = parser.parse_args()

# Determine whether to use GPU or CPU 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# If applicable, load arguments from a saved file
if args.json_args is not None:
    with open(args.json_args, 'rt') as f:
        copy_args = argparse.Namespace()
        copy_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=copy_args)

if not args.no_train:
    # Create the model name by adding a numerical suffix
    # to the end of the provided model name.
    # This ensures that the name is new and doesn't overwrite
    # an old model.
    model_name = args.model_name
    model_index = 0
    args.model_name = model_name + "_" + str(model_index)
    while args.model_name + ".log" in os.listdir(args.log_dir):
        model_index += 1
        args.model_name = model_name + "_" + str(model_index)

    # Set the random seed
    if args.random_seed is None:
        args.random_seed = model_index

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

# Set some Transformer-specific arguments
if (args.architecture == "Transformer" and not args.bottleneck) or (args.decoder_architecture == "Transformer" and not args.bottleneck):
    args.encoding_is_seq = True

if args.architecture == "Transformer" and args.dim_feedforward is None:
    args.dim_feedforward = args.hidden_size * 4
elif args.decoder_architecture == "Transformer" and args.decoder_dim_feedforward is None:
    if args.decoder_hidden_size is not None:
        args.decoder_dim_feedforward = args.decoder_hidden_size * 4
    else:
        args.decoder_dim_feedforward = args.hidden_size * 4

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + ".log")])

###################################################################
# Load the data
###################################################################

# Create dictionaries for converting tokens to numerical IDs
src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = src_trg_dicts_from_file_prefix(args.dataset_prefix, data_dir=args.data_dir, headers=args.headers)

args.vocab_size = src_vocab_size
args.decoder_vocab_size = trg_vocab_size

if args.architecture == "MLP" or args.decoder_architecture == "MLP":
    if args.uniform_src_length is None:
        args.uniform_src_length = max_src_length
    if args.uniform_trg_length is None:
        args.uniform_trg_length = max_trg_length

logging.info(args)

# Create dataloaders
train_set = DataLoader(args.data_dir + args.dataset_prefix + ".train", src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, dataset_size=args.training_size, **args.__dict__)
valid_set = DataLoader(args.data_dir + args.dataset_prefix + ".valid", src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, **args.__dict__)
test_set = DataLoader(args.data_dir + args.dataset_prefix + ".test", src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, **args.__dict__)

if args.dataset_prefix + ".gen" in os.listdir(args.data_dir):
    gen_set = DataLoader(args.data_dir + args.dataset_prefix + ".gen", src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, **args.__dict__)
else:
    gen_set = None

###################################################################
# Create the model
###################################################################

# Create role assigner, if the architecture is one that uses
# roles (this is uncommon)
if args.role_scheme is not None:
    seq2roles = RoleAssigner(args.role_scheme, uniform_seq_length=args.uniform_src_length, memoize=args.memoize, src_idx2token=src_idx2token, src_token2idx=src_token2idx, pad_src_left=args.pad_src_left)

    seq2roles.update_for_dataset(train_set)
    seq2roles.update_for_dataset(valid_set)
    seq2roles.update_for_dataset(test_set)

    if gen_set is not None:
        seq2roles.update_for_dataset(gen_set)

    args.seq2roles = seq2roles
    args.n_roles = seq2roles.role_vocab_size

if args.decoder_role_scheme is not None:
    decoder_seq2roles = RoleAssigner(args.decoder_role_scheme, uniform_seq_length=args.uniform_trg_length, memoize=args.memoize, src_idx2token=src_idx2token, src_token2idx=src_token2idx, pad_src_left=args.pad_src_left)
    
    decoder_seq2roles.update_for_dataset(train_set)
    decoder_seq2roles.update_for_dataset(valid_set)
    decoder_seq2roles.update_for_dataset(test_set)

    if gen_set is not None:
        decoder_seq2roles.update_for_dataset(gen_set)

    args.decoder_seq2roles = decoder_seq2roles
    args.decoder_n_roles = decoder_seq2roles.role_vocab_size


model = Seq2Seq(**args.__dict__).to(device=device)

# Handle special cases where the encoder is frozen and untrained
if args.no_train_encoder:
    if args.encoder_model_name is not None:
        model.encoder.load_state_dict(torch.load(args.weight_dir + args.encoder_model_name + ".weights"))

    if args.tpr_invert_encoder:
        model.decoder = TensorProductDecoder(hidden_size=model.decoder.hidden_size, n_fillers=model.decoder.n_fillers, n_roles=model.decoder.n_roles, filler_dim=model.decoder.filler_dim, role_dim=model.decoder.role_dim, has_linear_layer=model.decoder.has_linear_layer, seq2roles=model.decoder.seq2roles, tpr_enc_to_invert=model.encoder)

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    
# Handle special cases where the decoder is frozen and untrained
if args.no_train_decoder:
    for name, param in model.decoder.named_parameters():
        param.requires_grad = False


###################################################################
# Train the model
###################################################################

if args.no_train:
    if args.encoder_model_name is not None:
        model.encoder.load_state_dict(torch.load(args.weight_dir + args.encoder_model_name + ".weights"))
    if args.tpr_invert_encoder:
        model.decoder = TensorProductDecoder(hidden_size=model.decoder.hidden_size, n_fillers=model.decoder.n_fillers, n_roles=model.decoder.n_roles, filler_dim=model.decoder.filler_dim, role_dim=model.decoder.role_dim, has_linear_layer=model.decoder.has_linear_layer, seq2roles=model.decoder.seq2roles, tpr_enc_to_invert=model.encoder)

    if args.model_name + ".weights" not in os.listdir(args.weight_dir):
        torch.save(model.state_dict(), args.weight_dir + args.model_name + ".weights")

else:
    # Train the model
    args.stopping_criterion = "loss"
    args.task_type = "seq2seq"

    train(model=model, training_set=train_set, valid_set=valid_set, **args.__dict__)


# Load best saved weights
model.load_state_dict(torch.load(args.weight_dir + args.model_name + ".weights"))


logging.info("Parameter count:" + str(count_params(model)))

###################################################################
# Evaluate
###################################################################

# Evaluate on the test set
test_acc, _ = evaluate(model, test_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, pad_trg_left=args.pad_trg_left, prnt=args.prnt)
logging.info("Test accuracy:" + str(test_acc))

# Evaluate on the gen set, if there is one
if gen_set is not None:
    gen_acc, _ = evaluate(model, gen_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, pad_trg_left=args.pad_trg_left, prnt=args.prnt)
    logging.info("Gen accuracy:" + str(gen_acc))

###################################################################
# Save training arguments (so that we can easily reload the
# same arguments later if needed)
###################################################################
if not args.no_train:
    with open(args.log_dir + args.model_name + ".training_args.json", 'w') as fo:
        args.seq2roles = None
        args.decoder_seq2roles = None
        json.dump(vars(args), fo)



