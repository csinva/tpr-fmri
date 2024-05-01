
import os
import config

from collections import defaultdict

# Initialize a dictionary with tokens for padding,
# start of sentence, and end of sentence
def init_dictionary(default=None):
    if default is not None:
        token2idx = defaultdict(lambda:default)
        idx2token = {}
        idx2token[default] = "UNK"
    else:
        token2idx = {}
        idx2token = {}

    idx2token[config.PAD_TOKEN] = "<PAD>"
    idx2token[config.SOS_TOKEN] = "<SOS>"
    idx2token[config.EOS_TOKEN] = "<EOS>"

    token2idx["<PAD>"] = config.PAD_TOKEN
    token2idx["<SOS>"] = config.SOS_TOKEN
    token2idx["<EOS>"] = config.EOS_TOKEN

    vocab_size = 3

    return token2idx, idx2token, vocab_size

# Have all keys be sorted
# Useful for reloading models without
# having to save all of their token dictionaries
def standardize_dictionary(dictionary):
    preset_tokens = ["<PAD>", "<SOS>", "<EOS>"]
    other_tokens = []

    for token in dictionary:
        if token not in preset_tokens:
            other_tokens.append(token)

    sorted_tokens = sorted(other_tokens)
    all_tokens = preset_tokens + sorted_tokens
    
    token2idx = {}
    idx2token = {}

    for index, token in enumerate(all_tokens):
        token2idx[token] = index
        idx2token[index] = token

    return token2idx, idx2token

# Update a dictionary to include all words from the file fi
def update_dictionaries(src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length, fi, headers=False, src_indices=False):
    fi = open(fi, "r")

    first = True
    for line in fi:

        if first:
            first = False

            if headers:
                parts = line.strip().split("\t")
                if "input_seq" in parts:
                    index_src = parts.index("input_seq")
                elif "sentence" in parts:
                    index_src = parts.index("sentence")
                else:
                    index_src = 0

                if "target_seq" in parts:
                    index_trg = parts.index("target_seq")
                else:
                    index_trg = index_src
                continue
            else:
                index_src = 0
                index_trg = 1

                if len(line.strip().split("\t")) == 1:
                    index_trg = 0

        line_parts = line.strip().split("\t")
        src = line_parts[index_src]
        trg = line_parts[index_trg]

        src_words = src.split()
        if src_indices:
            src_words = [int(x) for x in src_words]
        trg_words = trg.split()

        # Update max lengths
        if len(src_words) > max_src_length:
            max_src_length = len(src_words)
        if len(trg_words) > max_trg_length:
            max_trg_length = len(trg_words)

        for src_word in src_words:
            if src_word not in src_token2idx:
                src_token2idx[src_word] = src_vocab_size
                src_idx2token[src_vocab_size] = src_word
                src_vocab_size += 1


        for trg_word in trg_words:
            if trg_word not in trg_token2idx:
                trg_token2idx[trg_word] = trg_vocab_size
                trg_idx2token[trg_vocab_size] = trg_word
                trg_vocab_size += 1

    fi.close()

    return src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length

# Create dictionaries from the train, validation, and test (and 
# optionally gen) files with the provided prefix
def src_trg_dicts_from_file_prefix(prefix, data_dir="data/", only_encodings=False, headers=False, max_length=None, src_indices=False, src_only=False):
    if only_encodings:
        suffix = "_encodings"
    else:
        suffix = ""

    src_token2idx, src_idx2token, src_vocab_size = init_dictionary()
    trg_token2idx, trg_idx2token, trg_vocab_size = init_dictionary()


    max_src_length = 0
    max_trg_length = 0

    src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = update_dictionaries(src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length, data_dir + prefix + ".train" + suffix, headers=headers, src_indices=src_indices)
    src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = update_dictionaries(src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length, data_dir + prefix + ".valid" + suffix, headers=headers, src_indices=src_indices)
    src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = update_dictionaries(src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length, data_dir + prefix + ".test" + suffix, headers=headers, src_indices=src_indices)

    if prefix + ".gen" + suffix in os.listdir(data_dir):
        src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length = update_dictionaries(src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length, data_dir + prefix + ".gen" + suffix, headers=headers, src_indices=src_indices)

    src_token2idx, src_idx2token = standardize_dictionary(src_token2idx)
    trg_token2idx, trg_idx2token = standardize_dictionary(trg_token2idx)


    return src_token2idx, src_idx2token, src_vocab_size, trg_token2idx, trg_idx2token, trg_vocab_size, max_src_length, max_trg_length








