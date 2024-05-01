
import os
import logging
import json
import numpy as np

os.environ['TRANSFORMERS_CACHE'] = '.transformers/'

from models import *
from token_dictionary import *
from utils import *
from training import *
from evaluation import *
from role_assignment_functions import *

import argparse
from discover_argparser import discover_parser

################################################################################
# Preliminaries
################################################################################

# Check whether we are on a GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load arguments
parser = argparse.ArgumentParser(parents=[discover_parser], description="Arguments governing model architectures and training")

args = parser.parse_args()

random_seed = args.random_seed

if args.json_args is not None:
    with open(args.json_args, 'rt') as f:
        copy_args = argparse.Namespace()
        copy_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=copy_args)

# Do this so that the random seed is not restricted to be the
# same as in the loaded args
args.random_seed = random_seed

args.stopping_criterion = "loss"
if args.encoding_is_seq:
    args.return_bindings = True

model_index = 0
model_name = args.model_name
args.model_name = model_name + "_" + str(model_index)
while args.model_name + ".log" in os.listdir(args.log_dir):
    model_index += 1
    args.model_name = model_name + "_" + str(model_index)

if args.random_seed is None:
    args.random_seed = model_index

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

if args.dataset_prefix_to_encode is None:
    args.dataset_prefix_to_encode = args.dataset_prefix

if args.encoder_model_name is None:
    args.encoder_model_name = args.decoder_model_name

# Even if the target model was trained with an input SOS token,
# we don't include it in the decomposition
args.input_sos = False

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + ".log")])
logging.info(args)

if args.role_learning:
    logging.info("Role learning is turned on")
    args.discover_role_scheme = "bow"

################################################################################
# Load the data
################################################################################

if args.encode_on_the_fly:
    model_prefix = ""
else:
    model_prefix = args.decoder_model_name + "_"

src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = src_trg_dicts_from_file_prefix(model_prefix + args.dataset_prefix, data_dir=args.data_dir, headers=args.headers, only_encodings=(not args.encode_on_the_fly))

logging.info("Tokenizer loaded")

if args.architecture == "MLP" or args.decoder_architecture == "MLP":
    if args.uniform_src_length is None:
        args.uniform_src_length = max_src_length
    if args.uniform_trg_length is None:
        args.uniform_trg_length = max_trg_length

# If we are generating encodings on the fly,
# specify the encoder used to do this (namely,
# the target model's encoder)
if args.encode_on_the_fly:
    file_prefix = ""
    file_suffix = ""
else:
    file_prefix = args.decoder_model_name + "_"
    file_suffix = "_encodings"

# Need to delete this key from args in order to
# pass args to the DataLoaders below
args_minus_role_scheme = dict(args.__dict__)
del args_minus_role_scheme["role_scheme"]

# Create data loaders for the training, validation, and test data
train_set = DataLoader(args.data_dir + file_prefix + args.dataset_prefix_to_encode + ".train" + file_suffix, src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, dataset_size=args.training_size, encoding_model=None, sentence2mask=(args.encode_on_the_fly and args.bert_mask), role_scheme=args.discover_role_scheme, **args_minus_role_scheme)
valid_set = DataLoader(args.data_dir + file_prefix + args.dataset_prefix_to_encode + ".valid" + file_suffix, src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, encoding_model=None, sentence2mask=(args.encode_on_the_fly and args.bert_mask), role_scheme=args.discover_role_scheme, **args_minus_role_scheme)
test_set = DataLoader(args.data_dir + file_prefix + args.dataset_prefix_to_encode + ".test" + file_suffix, src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, encoding_model=None, sentence2mask=(args.encode_on_the_fly and args.bert_mask), role_scheme=args.discover_role_scheme, **args_minus_role_scheme)


if file_prefix + args.dataset_prefix_to_encode + ".gen" + file_suffix in os.listdir(args.data_dir):
    gen_set = DataLoader(args.data_dir + file_prefix + args.dataset_prefix_to_encode + ".gen" + file_suffix, src_token2idx=src_token2idx, trg_token2idx=trg_token2idx, encoding_model=None, sentence2mask=(args.encode_on_the_fly and args.bert_mask), role_scheme=args.discover_role_scheme, **args_minus_role_scheme)
else:
    gen_set = None

logging.info("Dataset loaded")

# For when the target model's encoder and/or decoder uses roles (this is rare)
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
    
    args.decoder_seq2roles= decoder_seq2roles
    args.decoder_n_roles = decoder_seq2roles.role_vocab_size



################################################################################
# Create models (target model and DISCOVER model)
################################################################################

# Create the target model (the model we are trying to approximate)
if "bert" in args.architecture:
    target_model = BERTEncoderDecoder(layer=args.bert_layer, mask=args.bert_mask, indices=args.bert_indices, name=args.architecture)
    mask_id = target_model.mask_id
else:
    target_model = Seq2Seq(**args.__dict__)
    target_model.load_state_dict(torch.load(args.weight_dir + args.decoder_model_name + ".weights", map_location=device))
    mask_id = None

target_model.to(device=device)

# No dropout in the target model
target_model.eval()

logging.info("Target model created")

# If we are generating encodings on the fly,
# specify the encoder used to do this (namely,
# the target model's encoder)
if args.encode_on_the_fly:
    encoding_model = target_model.encoder
    for name, param in encoding_model.named_parameters():
        param.requires_grad = False
    train_set.encoding_model = encoding_model
    valid_set.encoding_model = encoding_model
    test_set.encoding_model = encoding_model

    # Need to reset the datasets so that they
    # now compute the encodings
    train_set.reset(hard_reset=True)
    valid_set.reset(hard_reset=True)
    test_set.reset(hard_reset=True)

    if gen_set is not None:
        gen_set.encoding_model = encoding_model
        gen_set.reset(hard_reset=True)


if args.encode_on_the_fly and args.bert_mask:
    # Specifies whether the fillers are token indices,
    # rather than tokens
    filler_indices = True
else:
    filler_indices = args.filler_indices

if args.discover_filler_role_list_from_file:
    args.discover_filler_role_list_prefix = args.data_dir + file_prefix + args.dataset_prefix_to_encode + "." + args.discover_role_scheme
else:
    args.discover_filler_role_list_prefix = None

# Create the role scheme for the DISCOVER model
discover_seq2roles = RoleAssigner(args.discover_role_scheme, mask_id=mask_id, uniform_seq_length=args.uniform_src_length, memoize=args.memoize, src_idx2token=src_idx2token, src_token2idx=src_token2idx, pad_src_left=args.pad_src_left, filler_indices=filler_indices, filler_role_list_prefix=args.discover_filler_role_list_prefix, filler_min_count=args.discover_filler_min_count, role_min_count=args.discover_role_min_count, filler_file_int=args.discover_filler_file_int)

# Assign fillers and roles to all the sequences in 
# the training, validation, and test sets so that the
# role assigner can build up a list of all observed
# fillers and roles
logging.info("Establishing role scheme")
discover_seq2roles.update_for_dataset(train_set)
discover_seq2roles.update_for_dataset(valid_set)
discover_seq2roles.update_for_dataset(test_set)

if gen_set is not None:
    discover_seq2roles.update_for_dataset(gen_set)

args.discover_seq2roles= discover_seq2roles
n_fillers = discover_seq2roles.filler_vocab_size
args.discover_n_roles = discover_seq2roles.role_vocab_size
logging.info("Role scheme established. Number of roles: " + str(args.discover_n_roles - 3) + " Number of fillers: " + str(n_fillers))

role_assigner_kwargs = {}

# Create the DISCOVER model
role_assigner_kwargs = {}
if args.role_learning:
    role_assigner_kwargs['num_roles'] = args.num_roles
    role_encoding_model = BERTEncoderDecoder(layer=args.bert_layer-1, mask=args.bert_mask, indices=args.bert_indices).encoder
    role_encoding_model.eval()
    for name, param in role_encoding_model.named_parameters():
        param.requires_grad = False
    role_assigner_kwargs['encoding_model'] = role_encoding_model
    role_assigner_kwargs['role_embedding_dim'] = args.discover_role_emb_size
    role_assigner_kwargs['softmax_roles'] = args.softmax_roles
    role_assigner_kwargs['center_roles_at_one'] = args.center_roles_at_one
    role_assigner_kwargs['relative_role_prediction_function'] = args.relative_role_prediction_function

logging.info("Creating TPE")
tpe = TensorProductEncoder(hidden_size=args.discover_hidden_size, n_fillers=n_fillers, n_roles=args.discover_n_roles, filler_dim=args.discover_emb_size, role_dim=args.discover_role_emb_size, seq2roles=discover_seq2roles, has_linear_layer=args.discover_has_linear_layer, return_bindings=args.return_bindings, aggregation=args.aggregation, role_learning=args.role_learning, role_assigner_kwargs=role_assigner_kwargs).to(device=device)

# Initialize the DISCOVER filler embeddings
# with the word embeddings from the target model
if args.discover_init_filler_embedding:
    logging.info("Loading filler embeddings")
    if args.architecture == "TPR":
        tpe.filler_embedding = target_model.encoder.filler_embedding
    elif "bert" in args.architecture:
        tpe_filler_weights = tpe.filler_embedding.weight.detach()
        for idx in discover_seq2roles.filler_idx2token:
            if idx < 3:
                continue

            filler = discover_seq2roles.filler_idx2token[idx]
            if isinstance(filler, int):
                tokens_filler = [filler]
            elif isinstance(filler, str):
                tokens_filler = target_model.bert_tokenizer([filler])["input_ids"][0][1:-1]

            if len(tokens_filler) == 0:
                tokens_filler = tokenizer([filler])["input_ids"][0][:1]

            if args.architecture == "bert":
                bert_emb = target_model.bert_model.bert.embeddings.word_embeddings(torch.LongTensor(tokens_filler).to(device=device)).mean(dim=0)
            elif args.architecture == "roberta":
                bert_emb = target_model.bert_model.roberta.embeddings.word_embeddings(torch.LongTensor(tokens_filler).to(device=device)).mean(dim=0)
            elif args.architecture == "distilbert":
                bert_emb = target_model.bert_model.distilbert.embeddings.word_embeddings(torch.LongTensor(tokens_filler).to(device=device)).mean(dim=0)
            elif args.architecture == "albert":
                bert_emb = target_model.bert_model.albert.embeddings.word_embeddings(torch.LongTensor(tokens_filler).to(device=device)).mean(dim=0)
            tpe_filler_weights[idx] = bert_emb

            tpe.filler_embedding.load_state_dict({'weight':tpe_filler_weights})
    else:
        tpe.filler_embedding = target_model.encoder.embedding

    if args.discover_no_train_filler_embedding:
        for name, param in tpe.filler_embedding.named_parameters():
            param.requires_grad = False



################################################################################
# Run the decomposition
################################################################################

logging.info("Beginning decomposition")
if args.task_type == "seq2seq":
    # Create a seq2seq model by combining a DISCOVER encoder
    # with the target model's (frozen) decoder and then training
    # the whole thing end-to-end
    model = Seq2Seq(**args.__dict__)
    model.load_state_dict(torch.load(args.weight_dir + args.decoder_model_name + ".weights", map_location=device))

    model.encoder = tpe

    model.to(device=device)
    
    for name, param in model.decoder.named_parameters():
        param.requires_grad = False

    train(model=model, training_set=train_set, valid_set=valid_set, save_encoder_only=True, **args.__dict__)

else:
    # Train a DISCOVER encoder on mean squared error with the target encodings
    # This is the "standard" usage of DISCOVER (e.g., what we've done in
    # the TPDN paper and the ROLE paper)
    train(model=tpe, training_set=train_set, valid_set=valid_set, **args.__dict__)

# Load the best saved weights of the DISCOVER encoder
tpe.load_state_dict(torch.load(args.weight_dir + args.model_name + ".weights", map_location=device))

logging.info("TPE parameter count: " + str(count_params(tpe)))

# Subtracting 3 due to the 3 extra ones (EOS/SOS/pad)
logging.info("Number of roles: " + str(args.discover_n_roles - 3))



################################################################################
# Evaluate the quality of the approximation
################################################################################


# Get MSE on test set
if args.encoding_is_seq:
    _, test_loss = evaluate(tpe, test_set, task_type="encoding_seq", no_eos=args.no_eos, criterion=nn.MSELoss(), pad_trg_left=args.pad_trg_left, role_learning_hyper=args.role_learning_hyper, softmax_roles=args.softmax_roles)
else:
    _, test_loss = evaluate(tpe, test_set, task_type="encoding", no_eos=args.no_eos, criterion=nn.MSELoss(), pad_trg_left=args.pad_trg_left, role_learning_hyper=args.role_learning_hyper, softmax_roles=args.softmax_roles)

logging.info("Test MSE:" + str(test_loss))

if gen_set is not None:
    # Get MSE on gen set
    if args.encoding_is_seq:
        _, gen_loss = evaluate(tpe, gen_set, task_type="encoding_seq", no_eos=args.no_eos, criterion=nn.MSELoss(), pad_trg_left=args.pad_trg_left)
    else:
        _, gen_loss = evaluate(tpe, gen_set, task_type="encoding", no_eos=args.no_eos, criterion=nn.MSELoss(), pad_trg_left=args.pad_trg_left)
    logging.info("Gen MSE:" + str(gen_loss))


# Swapping accuracy and swapping agreement
if args.eval_swapping:
    # Create the model for swapping accuracy
    swapping_model = Seq2Seq(**args.__dict__)
    swapping_model.load_state_dict(torch.load(args.weight_dir + args.decoder_model_name + ".weights", map_location=device))

    swapping_model.encoder = tpe
    swapping_model.to(device=device)


    # Evaluate on the test set
    test_acc, _ = evaluate(swapping_model, test_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, pad_trg_left=args.pad_trg_left, role_learning_hyper=args.role_learning_hyper, softmax_roles=args.softmax_roles)
    logging.info("Test accuracy:" + str(test_acc))

    test_agreement, _ = evaluate(swapping_model, test_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, target_model=target_model, pad_trg_left=args.pad_trg_left, role_learning_hyper=args.role_learning_hyper, softmax_roles=args.softmax_roles)

    logging.info("Test agreement:" + str(test_agreement))

    if gen_set is not None:

        # Evaluate on the test set
        gen_acc, _ = evaluate(swapping_model, gen_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, pad_trg_left=args.pad_trg_left)
        logging.info("Gen accuracy:" + str(gen_acc))


        gen_agreement, _ = evaluate(swapping_model, gen_set, task_type="seq2seq", no_eos=args.no_eos, tf_ratio=0.0, target_model=target_model, pad_trg_left=args.pad_trg_left)

        logging.info("Gen agreement:" + str(gen_agreement))

# Nearest neighbor evaluation: For every item in the test set,
# get the DISCOVER encoding. Then, find the DISCOVER encoding's
# nearest neighbor among all of the target encodings across the
# train, validation, and test sets. Check whether the nearest
# neighbor is the right one
if args.eval_nearest_neighbor_all:
    train_set.reset()
    valid_set.reset()
    test_set.reset()

    nn_acc = nearest_neighbor(tpe, test_set, [train_set, valid_set, test_set])
    logging.info("Nearest neighbor accuracy (all): " + str(nn_acc))

    if gen_set is not None:
        train_set.reset()
        valid_set.reset()
        test_set.reset()
        gen_set.reset()

        nn_acc = nearest_neighbor(tpe, test_set, [train_set, valid_set, test_set, gen_set])
        logging.info("Gen nearest neighbor accuracy (all): " + str(nn_acc))

# Nearest neighbor evaluation: For every item in the test set,
# get the DISCOVER encoding. Then, find the DISCOVER encoding's
# nearest neighbor among the target encodings just for the
# test set. Check whether the nearest neighbor is the right one
if args.eval_nearest_neighbor_test:
    test_set.reset()

    nn_acc = nearest_neighbor(tpe, test_set, [test_set])
    logging.info("Nearest neighbor accuracy (test): " + str(nn_acc))

    if gen_set is not None:
        gen_set.reset()
        nn_acc = nearest_neighbor(tpe, gen_set, [gen_set])
        logging.info("Gen nearest neighbor accuracy (gen): " + str(nn_acc))

# Total variation distance
if args.eval_tvd:

    test_set.reset()
    tvd_results = one_minus_tvd(tpe, target_model.decoder, test_set, target_encoder=target_model.encoder, bert_mask=args.bert_mask, bert_indices=args.bert_indices, no_eos=args.no_eos, pad_trg_left=args.pad_trg_left, encoding_is_seq=args.encoding_is_seq) 

    logging.info("1 - TVD: " + str(tvd_results))

    if gen_set is not None:
        gen_set.reset()

        tvd_results = one_minus_tvd(tpe, target_model.decoder, gen_set, target_encoder=target_model.encoder, bert_mask=args.bert_mask, bert_indices=args.bert_indices, no_eos=args.no_eos, pad_trg_left=args.pad_trg_left, encoding_is_seq=args.encoding_is_seq)
        logging.info("1 - TVD (gen): " + str(tvd_results))

# Explained variance
if args.eval_explained_variance:
    test_set.reset()
    r2 = explained_variance(tpe, test_set, encoding_is_seq=args.encoding_is_seq)
    logging.info("R2:" + str(r2))

    if gen_set is not None:
        gen_set.reset()

        r2 = explained_variance(tpe, gen_set, encoding_is_seq=args.encoding_is_seq)
        logging.info("R2 (gen):" + str(r2))


# Log the arguments used in the decomposition
args.seq2roles = None
args.decoder_seq2roles = None
args.discover_seq2roles = None
args.indices2tokens = None
args.decoder_indices2tokens = None
with open(args.log_dir + args.model_name + ".decomposition_args.json", 'w') as fo:
    json.dump(vars(args), fo)


