import config

import math

from collections import defaultdict

from token_dictionary import *



class RoleAssigner:

    def __init__(self, role_scheme, mask_id=None, uniform_seq_length=None, memoize=True, src_idx2token=None, src_token2idx=None, pad_src_left=None, filler_indices=False, filler_role_list_prefix=None, filler_min_count=0, role_min_count=0, filler_file_int=False):
        # For memoization
        self.seqdict = {}

        self.filler_role_list_prefix = filler_role_list_prefix

        if filler_role_list_prefix is not None:

            # Using EOS token as default because it is not otherwise used for fillers and roles
            self.filler_token2idx, self.filler_idx2token, self.filler_vocab_size = init_dictionary(default=config.EOS_TOKEN)
            self.role_token2idx, self.role_idx2token, self.role_vocab_size = init_dictionary(default=config.EOS_TOKEN)

            fi_filler = open(filler_role_list_prefix + ".filler_list", "r")
            for line in fi_filler:
                parts = line.strip().split("\t")
                filler = parts[0]

                if filler_file_int:
                    filler = int(filler)

                count = int(parts[1])
                if count >= filler_min_count:
                    self.filler_token2idx[filler] = self.filler_vocab_size
                    self.filler_idx2token[self.filler_vocab_size] = filler
                    self.filler_vocab_size += 1

            fi_filler.close()

            fi_role = open(filler_role_list_prefix + ".role_list", "r")
            for line in fi_role:
                parts = line.strip().split("\t")
                role = parts[0]

                count = int(parts[1])
                if count >= role_min_count:
                    self.role_token2idx[role] = self.role_vocab_size
                    self.role_idx2token[self.role_vocab_size] = role
                    self.role_vocab_size += 1
            fi_role.close()

        else:
            self.filler_token2idx, self.filler_idx2token, self.filler_vocab_size = init_dictionary()
            self.role_token2idx, self.role_idx2token, self.role_vocab_size = init_dictionary()

        self.uniform_seq_length = uniform_seq_length
        self.memoize = memoize
        self.src_idx2token = src_idx2token
        self.src_token2idx = src_token2idx
        self.pad_src_left = pad_src_left
        self.filler_indices = filler_indices

        # A list of role assignment functions
        # In most cases this list will just have one element
        # The option for having multiple is so that
        # we can sum together multiple role schemes
        self.seq2roles_list = create_role_function(role_scheme, idx2token=self.src_idx2token, token2idx=self.src_token2idx, mask_id=mask_id)

    # Update the dictionary of fillers based on a sequence of fillers
    def update_fillers(self, filler_seq):
        for filler in filler_seq:
            if filler not in self.filler_token2idx:
                self.filler_token2idx[filler] = self.filler_vocab_size
                self.filler_idx2token[self.filler_vocab_size] = filler
                self.filler_vocab_size += 1

    # Update the dictionary of roles based on a sequence of roles
    def update_roles(self, role_seq):
        for role in role_seq:
            if role not in self.role_token2idx:
                self.role_token2idx[role] = self.role_vocab_size
                self.role_idx2token[self.role_vocab_size] = role
                self.role_vocab_size += 1
   
    # Given a dataset, update the filler and role dictionaries
    # to include all fillers and roles present in the dataset
    def update_for_dataset(self, dataset):

        # If passed a file to use for fillers and roles, use that instead
        if self.filler_role_list_prefix is None:
            dataset.reset()

            original_no_encode = dataset.no_encode
            dataset.no_encode = True

            for index, batch in enumerate(dataset):
                if batch is None:
                    # done with dataset
                    break

                _, _ = self.assign_roles_batch(batch, update_dicts=True)
   
            # Standardize the dictionaries so that, upon reloading, 
            # they have the same mappings
            self.filler_token2idx, self.filler_idx2token = standardize_dictionary(self.filler_token2idx)
            self.role_token2idx, self.role_idx2token = standardize_dictionary(self.role_token2idx)

            dataset.reset()
            dataset.no_encode = original_no_encode
    

    # Assign roles to a single sequence, with a single
    # role assignment function
    def assign_roles(self, seq, batch, index_in_batch, seq2roles):
        if self.memoize:
            if tuple(seq) in self.seqdict:
                return self.seqdict[tuple(seq)]
            else:
                fillers, roles = seq2roles(seq, batch, index_in_batch)
                self.seqdict[tuple(seq)] = (fillers, roles)

                return (fillers, roles)

        else:
            fillers, roles = seq2roles(seq, batch, index_in_batch)
            return (fillers, roles)

    # 1. Assigns roles
    # 2. Converts roles to indices
    # 3. Pads the batch
    def assign_roles_batch(self, batch, update_dicts=False):

        input_seqs = batch["input_seq"]
        input_lengths = batch["input_lengths"]

        # A list of lists of roles
        # There is one list of roles for each role scheme
        # being used (in most cases, just one role scheme)
        all_padded_roles = []

        # Loop over all role assignment functions
        for index, seq2roles in enumerate(self.seq2roles_list):
            
            # For differentiating roles from
            # different role schemes that are
            # being summed
            role_prefix = str(index) + "_"

            unpadded_filler_list = []
            unpadded_role_list = []

            # Loop over all input sequences and their lengths
            for index_in_batch, (seq, length) in enumerate(zip(input_seqs, input_lengths)):
                if not self.filler_indices:
                    input_seq = [self.src_idx2token[x] for x in seq]
                else:
                    input_seq = seq[:]

                if self.pad_src_left:
                    unpadded_fillers, unpadded_roles = self.assign_roles(input_seq[-1*length:], batch, index_in_batch, seq2roles)
                else:
                    unpadded_fillers, unpadded_roles = self.assign_roles(input_seq[:length], batch, index_in_batch, seq2roles)

                # Add prefix specific to this role scheme
                unpadded_roles = [role_prefix + str(role) for role in unpadded_roles]

                if update_dicts:
                    # In the initial pass over a file, we need to
                    # update the dicts to include all the relevant
                    # fillers and roles
                    self.update_fillers(unpadded_fillers)
                    self.update_roles(unpadded_roles)

                # Convert fillers and roles to token ids
                unpadded_fillers = [self.filler_token2idx[filler] for filler in unpadded_fillers]
                unpadded_roles = [self.role_token2idx[role] for role in unpadded_roles]

                unpadded_filler_list.append(unpadded_fillers)
                unpadded_role_list.append(unpadded_roles)
        
            # Pad the fillers and the roles to the length of the
            # longest sequence in the batch
            filler_lengths = [len(filler_seq) for filler_seq in unpadded_filler_list]
            max_filler_length = max(filler_lengths)

            if self.uniform_seq_length is not None:
                max_filler_length = self.uniform_seq_length

            if self.pad_src_left:
                padded_fillers = [[config.PAD_TOKEN]*(max_filler_length-len(unpadded_fillers)) + unpadded_fillers for unpadded_fillers in unpadded_filler_list]
                padded_roles = [[config.PAD_TOKEN]*(max_filler_length-len(unpadded_roles)) + unpadded_roles for unpadded_roles in unpadded_role_list]
            else:
                padded_fillers = [unpadded_fillers + [config.PAD_TOKEN]*(max_filler_length-len(unpadded_fillers)) for unpadded_fillers in unpadded_filler_list]
                padded_roles = [unpadded_roles + [config.PAD_TOKEN]*(max_filler_length-len(unpadded_roles)) for unpadded_roles in unpadded_role_list]

            all_padded_roles.append(padded_roles)

        return padded_fillers, all_padded_roles


# The following 3 functions together cover the task of taking in a 
# role scheme name and returning a function that assigns roles to
# a sequence of fillers.
# We need the 3 functions to cover 2 special cases: "sum" role schmes
# and "combo" role schemes

# Splits into a set of role schemes being summed,
# and creates a role assignment function for each of them
def create_role_function(role_scheme, idx2token=None, token2idx=None, mask_id=None):
    role_schemes_to_sum = role_scheme.split("+")

    seq2roles_list = []

    for sub_role_scheme in role_schemes_to_sum:
        sub_seq2roles = create_role_function_helper(sub_role_scheme, idx2token=idx2token, token2idx=token2idx, mask_id=mask_id)
        seq2roles_list.append(sub_seq2roles)
        
    return seq2roles_list

# Deals with "combo" role schemes, where each role is the 
# concatenation of two roles from two other role schemes
def create_role_function_helper(role_scheme, idx2token=None, token2idx=None, mask_id=None):

    if role_scheme.startswith("combo"):
        role_scheme_parts = role_scheme.split("-")

        # role_scheme_parts[0] is the word "combo"
        role_scheme_1 = role_scheme_parts[1]
        role_scheme_2 = role_scheme_parts[2]

        seq2roles_1 = seq2roles_from_string(role_scheme_1, idx2token=idx2token, token2idx=token2idx, mask_id=mask_id)
        seq2roles_2 = seq2roles_from_string(role_scheme_2, idx2token=idx2token, token2idx=token2idx, mask_id=mask_id)

        seq2roles = create_combo_roles(seq2roles_1, seq2roles_2)

    else:
        seq2roles = seq2roles_from_string(role_scheme, idx2token=idx2token, mask_id=mask_id)

    return seq2roles


# Given a role scheme name (specifically, a normal role
# scheme that is not a sum or a combo), returns a 
# role assignment function
def seq2roles_from_string(role_scheme, idx2token=None, token2idx=None, mask_id=None):

    if role_scheme == "ltr":
        seq2roles = create_ltr_roles()
    elif role_scheme == "rtl":
        seq2roles = create_rtl_roles()
    elif role_scheme == "bi":
        seq2roles = create_bi_roles()
    elif role_scheme == "bow":
        seq2roles = create_bow_roles()
    elif role_scheme == "wickel":
        seq2roles = create_wickel_roles()
    
    elif role_scheme == "svo_v":
        seq2roles = create_svo_roles(subj=False, obj=False)
    elif role_scheme == "svo_sv":
        seq2roles = create_svo_roles(subj=True, obj=False)
    elif role_scheme == "svo_vo":
        seq2roles = create_svo_roles(subj=False, obj=True)
    elif role_scheme == "svo_ava":
        seq2roles = create_svo_roles(subj=True, obj=True, distinct=False)
    elif role_scheme == "svo_svo":
        seq2roles = create_svo_roles(subj=True, obj=True, distinct=True)
    elif role_scheme == "svo_a_v":
        seq2roles = create_svo_roles(subj=False, obj=False, aux=True)
    elif role_scheme == "svo_a_sv":
        seq2roles = create_svo_roles(subj=True, obj=False, aux=True)
    elif role_scheme == "svo_a_vo":
        seq2roles = create_svo_roles(subj=False, obj=True, aux=True)
    elif role_scheme == "svo_a_ava":
        seq2roles = create_svo_roles(subj=True, obj=True, distinct=False, aux=True)
    elif role_scheme == "svo_a_svo":
        seq2roles = create_svo_roles(subj=True, obj=True, distinct=True, aux=True)

    return seq2roles


# The rest of this file defines role assignment functions
# A role assignment function takes in:
#      1. The sequence to be assigned roles to
#      2. The batch of which this sequence is a part
#      3. This sequence's position in the batch
# And it returns:
#      1. The list of fillers
#      2. The list of roles.
# Each function below is a function which returns a role assignment function

# Bag-of-words roles: All fillers have
# the same role
def create_bow_roles():

    def bow_roles(seq, batch, index_in_batch):
        return seq, [0 for _ in seq]

    return bow_roles

# Left-to-right roles: Each filler's role is its position
# in the sequence, counting from left to right
def create_ltr_roles():

    def ltr_roles(seq, batch, index_in_batch):
        return seq, ["LTR" + str(x) for x in list(range(len(seq)))]
    
    return ltr_roles

# Right-to-left roles: Each filler's role is its position
# in the sequence, counting from right to left
def create_rtl_roles():

    def rtl_roles(seq, batch, index_in_batch):
        return seq, ["RTL" + str(x) for x in list(range(len(seq)))[::-1]]
    
    return rtl_roles


# Bidirectional roles: Each filler's role is a 2-tuple of
# its position in the sequence counting from left to right,
# and its position in the sequence countring from right to left
def create_bi_roles():
	
    def bi_roles(seq, batch, index_in_batch):
        ltr = list(range(len(seq)))
        rtl = list(range(len(seq)))[::-1]
        bi = zip(ltr, rtl)
        return seq, bi

    return bi_roles

# Wickelroles: Each filler has a contextual role representing
# the filler before it and the filler after it
def create_wickel_roles():

    # Assign wickelroles to a sequence
    def wickel_roles(seq, batch, index_in_batch):
        prev_num = "#"
		
        wickels = []
        for index, item in enumerate(seq):
            if index == len(seq) - 1:
                next_num = "#"
            else:
                next_num = seq[index + 1]
			
            wickels.append(str(prev_num) + "_" + str(next_num))
            prev_num = item
		
        return seq, wickels

    return wickel_roles

# Roles specific to the case where the sentence is
# of the form "The SUBJECT VERBED the OBJECT"
def create_svo_roles(subj=False, obj=False, distinct=False, aux=False):

    if aux:
        aux_modifier = 1
    else:
        aux_modifier = 0

    def svo_roles(seq, batch, index_in_batch):

        sentence = batch["sentence"][index_in_batch]
        if "masked_sentence" in batch:
            masked_sentence = batch["masked_sentence"][index_in_batch]
        else:
            masked_sentence = sentence

        masked_words = masked_sentence.split()

        fillers = [masked_words[2+aux_modifier]]
        roles = ["verb"]

        if subj:
            fillers.append(masked_words[1])
            if distinct:
                roles.append("subj")
            else:
                roles.append("arg")

        if obj:
            fillers.append(masked_words[4+aux_modifier])
            if distinct:
                roles.append("obj")
            else:
                roles.append("arg")
        
        return fillers, roles
    
    return svo_roles


def create_combo_roles(seq2roles_1, seq2roles_2):

    def combo_roles(seq, batch, index_in_batch):
        # We assume that both role schemes use the same fillers
        fillers, roles_1 = seq2roles_1(seq, batch, index_in_batch)
        _, roles_2 = seq2roles_2(seq, batch, index_in_batch)

        roles = zip(roles_1, roles_2)

        return fillers, roles

    return combo_roles



