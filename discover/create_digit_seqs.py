import numpy as np
import random
from random import shuffle

from utils import pairs_to_file

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", help="size of the training set", type=int, default=0)
parser.add_argument("--n_valid", help="size of the validation set", type=int, default=0)
parser.add_argument("--n_test", help="size of the test set", type=int, default=0)
parser.add_argument("--n_gen", help="size of the generalization set", type=int, default=0)
parser.add_argument("--max_seq_length", help="maximum sequence length", type=int, default=6)
parser.add_argument("--min_seq_length", help="minimum sequence length", type=int, default=1)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=10)
parser.add_argument("--withheld_ltr", help="comma-separated list of withheld digit/position pairs; each pair should be digit-hyphen-position. E.g., 0-1,2-3 means to withhold 0 in position 1 and 2 in position 3", type=str, default="")
parser.add_argument("--withheld_rtl", help="comma-separated list of withheld digit/position pairs where positions count from right to left; each pair should be digit-hyphen-position. E.g., 0-1,2-3 means to withhold 0 in position 2nd-from-right and 2 in position 4th-from-right", type=str, default="")
parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default=None)
parser.add_argument("--task", help="task", type=str, default=None)
parser.add_argument("--random_seed", help="random seed", type=int, default=None)
parser.add_argument("--data_dir", help="directory for saving data", type=str, default="data/")
args = parser.parse_args()

if args.random_seed is None:
    args.random_seed = random.randint(0,1000)

random.seed(args.random_seed)
np.random.seed(args.random_seed)


# The task of interleaving a sequence
# E.g., interleaved([1,2,3,4,5,6]) = [1,6,2,5,3,4]
def interleaved(sequence, start_right=False):
    if len(sequence) <= 1:
        return list(sequence)
    else:
        if start_right:
            return [sequence[-1], sequence[0]] + interleaved(sequence[1:-1], start_right=start_right)
        else:
            return [sequence[0], sequence[-1]] + interleaved(sequence[1:-1], start_right=start_right)

# Mapping an input sequence to the output
# predicted by the task
def transform(sequence, task):
    if task == "copy":
        return sequence
    if task == "reverse":
        return sequence[::-1]
    if task == "sort_ascending":
        return sorted(sequence)
    if task == "sort_descending":
        return sorted(sequence)[::-1]
    if task == "interleave":
        return interleaved(sequence)
    if task == "interleave_right":
        return interleaved(sequence, start_right=True)

# Determining whether a sequence is in-distribution based on the 
# withholding specified by args.withheld_ltr
withhold_ltr_dict = {}
if len(args.withheld_ltr) != 0:
    withhold_ltr_pairs = args.withheld_ltr.split(",")
else:
    withhold_ltr_pairs = []

for pair in withhold_ltr_pairs:
    parts = pair.split("-")
    digit = int(parts[0])
    index = int(parts[1])

    if index not in withhold_ltr_dict:
        withhold_ltr_dict[index] = []
    if digit not in withhold_ltr_dict[index]:
        withhold_ltr_dict[index].append(digit)

withhold_rtl_dict = {}
if len(args.withheld_rtl) != 0:
    withhold_rtl_pairs = args.withheld_rtl.split(",")
else:
    withhold_rtl_pairs = []

for pair in withhold_rtl_pairs:
    parts = pair.split("-")
    digit = int(parts[0])
    index = int(parts[1])

    if index not in withhold_rtl_dict:
        withhold_rtl_dict[index] = []
    if digit not in withhold_rtl_dict[index]:
        withhold_rtl_dict[index].append(digit)


def in_distribution(seq):
    for index, elt in enumerate(seq):
        if index in withhold_ltr_dict and elt in withhold_ltr_dict[index]:
            return False

    for index, elt in enumerate(seq[::-1]):
        if index in withhold_rtl_dict and elt in withhold_rtl_dict[index]:
            return False

    return True

def doubly_out_of_distribution(seq):

    for index, elt in enumerate(seq):
        ltr_index = index
        rtl_index = len(seq) - ltr_index - 1

        if ltr_index in withhold_ltr_dict and elt in withhold_ltr_dict[ltr_index]:
            if rtl_index in withhold_rtl_dict and elt in withhold_rtl_dict[rtl_index]:
                return True

    return False

# Creates a list of num_examples_needed examples
# Each example consists of a sequence of digits of
# length seq_length, where each digit is randomly
# drawn from 0 to (vocab_size - 1)
# If ood is False, all examples must be in-distribution
# Else, all examples must be out-of-distribution
# All examples must be unique
def generate_examples(min_seq_length, max_seq_length, vocab_size, num_examples_needed, ood=False, doubly_ood=False):

    list_examples = []
    dict_examples = {}

    num_examples = 0
    while num_examples < num_examples_needed:
        seq_length = min_seq_length + np.random.randint(max_seq_length - min_seq_length + 1)
        seq = tuple(np.random.randint(vocab_size,size=seq_length))
        if seq not in dict_examples:
            if (in_distribution(seq) and not ood and not doubly_ood) or (not in_distribution(seq) and ood) or (doubly_out_of_distribution(seq) and doubly_ood):
            	list_examples.append((seq, transform(seq, args.task))) 
            	dict_examples[seq] = 1
            	num_examples += 1

    return list_examples

# Generate the examples

# We generate the train, validation, and test examples all at once like
# this so that there won't be any differences in length distribution
# across the three sets
iid_examples = generate_examples(args.min_seq_length, args.max_seq_length, args.vocab_size, args.n_train+args.n_valid+args.n_test, ood=False)
shuffle(iid_examples)
train_set = iid_examples[:args.n_train]
valid_set = iid_examples[args.n_train:args.n_train + args.n_valid]
test_set = iid_examples[args.n_train + args.n_valid:]

if args.n_gen > 0:
    if args.withheld_ltr != "" and args.withheld_rtl != "":
        ood_examples = generate_examples(args.min_seq_length, args.max_seq_length, args.vocab_size, args.n_gen, doubly_ood=True)
    else:
        ood_examples = generate_examples(args.min_seq_length, args.max_seq_length, args.vocab_size, args.n_gen, ood=True)
    gen_set = ood_examples

# Save the sequences to files
if args.n_train > 0:
    pairs_to_file(train_set, args.data_dir + args.prefix + ".train")
if args.n_valid > 0: 
    pairs_to_file(valid_set, args.data_dir + args.prefix + ".valid")
if args.n_test > 0:
    pairs_to_file(test_set, args.data_dir + args.prefix + ".test")
if args.n_gen > 0:
    pairs_to_file(gen_set, args.data_dir + args.prefix + ".gen")


with open(args.data_dir + args.prefix + ".dataset_creation_args.json", 'w') as fo:
    json.dump(vars(args), fo)

