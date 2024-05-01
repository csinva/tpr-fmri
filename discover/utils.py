
import random
import torch
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# Key arguments:
# - stream: If true, only load part of the file at a time.
#           If false, load the whole file from the start.
# - encoding_model: If included, use this model to compute
#                   the encodings to be approximated by DISCOVER
#                   as you go along. Otherwise, use pre-computed encodings.
#                   (not relevant if you're training a target model, rather
#                   than a DISCOVER model)
class DataLoader():

    def __init__(self, filename, batch_size=None, stream=False, batches_per_buffer=0, shuffle=False, headers=False, role_scheme=None, src_token2idx=None, trg_token2idx=None, filler_token2idx=None, role_token2idx=None, pad_src_left=False, pad_trg_left=False, uniform_src_length=None, uniform_trg_length=None, input_sos=False, encoding_is_seq=False, filler_indices=False, dataset_size=None, encoding_model=None, sentence2mask=False, input_seq_from_bert=False, no_encode=False, **kwargs):
        
        self.filename = filename
        self.batch_size = batch_size
        self.stream = stream
        self.batches_per_buffer = batches_per_buffer
        self.buffer_size = batches_per_buffer*batch_size

        if not self.stream:
            # So that we read lines until the file ends
            self.buffer_size = math.inf

        self.shuffle = shuffle
        self.headers = headers
        self.role_scheme = role_scheme
        self.src_token2idx = src_token2idx
        self.trg_token2idx = trg_token2idx
        self.filler_token2idx = filler_token2idx
        self.role_token2idx = role_token2idx
        self.pad_src_left = pad_src_left
        self.pad_trg_left = pad_trg_left
        self.uniform_src_length = uniform_src_length
        self.uniform_trg_length = uniform_trg_length
        self.input_sos = input_sos
        self.encoding_is_seq = encoding_is_seq
        self.filler_indices = filler_indices
        self.dataset_size = dataset_size
        self.encoding_model = encoding_model
        self.sentence2mask = sentence2mask
        self.input_seq_from_bert = input_seq_from_bert
        self.done_with_file = False

        self.no_encode = no_encode
        

        self.fi = open(self.filename, "r")

        self.index2header = {}
        self.header2index = {}
        if headers:
            header_list = self.fi.readline().strip().split("\t")
            for index, header in enumerate(header_list):
                self.index2header[index] = header
                self.header2index[header] = index
        elif self.sentence2mask:
            self.index2header[0] = "sentence"
            self.index2header[1] = "const"
            self.index2header[2] = "dep"

            self.header2index["sentence"] = 0
            self.header2index["const"] = 1
            self.header2index["dep"] = 2

        else:
            self.index2header[0] = "input_seq"
            self.index2header[1] = "target_seq"
            self.index2header[2] = "encoding"

            self.header2index["input_seq"] = 0
            self.header2index["target_seq"] = 1
            self.header2index["encoding"] = 2

        # List of batches that have been loaded
        self.current_batches = []
        
        # Position in the list of batches
        self.pointer = 0

        # Number of examples in the file that
        # have been read
        self.line_in_file = 0

        # If we're streaming, this loads the buffer
        # Else, it loads the whole file
        self.reload_buffer()

        if not self.stream:
            # We're done with the file, so don't 
            # need to keep it open
            self.fi.close()

    # Starting a new epoch
    def reset(self, hard_reset=False):
        
        if self.stream or hard_reset:
            # Need to reopen the file and start
            # from the top
            self.fi.close()

            self.fi = open(self.filename, "r")
            self.done_with_file = False
            if self.headers:
                # We have already read the headers in __init__, so 
                # no need to reprocess them
                _ = self.fi.readline()

            self.current_batches = []
            self.reload_buffer()
        
            self.pointer = 0
            self.line_in_file = 0

            if not self.stream:
                self.fi.close()
        else:
            # No need to reopen the file; instead,
            # just shuffle the batches and reset the
            # pointer to 0
            if self.shuffle:
                random.shuffle(self.current_batches)

            self.pointer = 0

    # Load the buffer
    # If we are not streaming, this loads the whole file
    def reload_buffer(self):

        # Batches on the buffer
        self.current_batches = []

        # Current position in current_batches
        self.pointer = 0

        lines_read = 0

        examples = []

        # If we're not streaming, self.buffer_size is infinity, so
        # we keep reading until the file is done
        while lines_read < self.buffer_size:
            line = self.fi.readline()
            lines_read += 1
            self.line_in_file += 1

            # Stop reading new examples because we've hit
            # the desired dataset size
            if self.dataset_size is not None and self.line_in_file > self.dataset_size:
                break

            # We've hit the end of the file
            if line == "":
                break
           
            line_parts = line.strip().split("\t")
            example = {}

            for index, elt in enumerate(line_parts):
                header = self.index2header[index]

                if header == "input_seq":
                    src = elt.split()
                    if self.filler_indices:
                        src = [int(x) for x in src]
                    if self.input_sos:
                        src = ["<SOS>"] + src

                    src = [self.src_token2idx[token] for token in src]
                    example[header] = src
                    example["input_sentence"] = elt

                elif header == "target_seq":
                    trg = elt.split()
                    trg = [self.trg_token2idx[token] for token in trg]
                    example[header] = trg
                    example["target_sentence"] = elt

                elif header == "encoding":
                    encoding = elt.split()

                    if self.encoding_is_seq:
                        encoding = " ".join(encoding)
                        encoding = encoding.split("/")
                        encoding = [[float(x) for x in elt.split()] for elt in encoding]
                    else:
                        encoding = [float(x) for x in encoding]

                    example[header] = encoding

                elif header == "word_index":
                    example[header] = int(elt)

                else:
                    example[header] = elt

            if ("input_seq" not in example or "target_seq" not in example) and "sentence" in example and not self.input_seq_from_bert:
                src = example["sentence"].split()
                if self.filler_indices:
                    src = [int(x) for x in src]
                if self.input_sos:
                    src = ["<SOS>"] + src

                src = [self.src_token2idx[token] for token in src]
                if "input_seq" not in example:
                    example["input_seq"] = src
                if "target_seq" not in example:
                    example["target_seq"] = src

            if self.role_scheme in example:
                fillers_and_roles = example[self.role_scheme]
                fillers, roles = fillers_and_roles.split("/")

                fillers = [self.filler_token2idx[filler] for filler in fillers.split()]
                roles = [self.role_token2idx[role] for role in roles.split()]

                example["fillers"] = fillers
                example["roles"] = roles

            examples.append(example)


        if self.sentence2mask:
            self.current_batches = mask_batchify(examples, self.batch_size)
        else:
            self.current_batches = batchify(examples, self.batch_size, pad_src_left=self.pad_src_left, pad_trg_left=self.pad_trg_left, uniform_src_length=self.uniform_src_length, uniform_trg_length=self.uniform_trg_length)

        if self.shuffle:
            random.shuffle(self.current_batches)

    def __iter__(self):
        return self

    def __next__(self):

        if self.pointer == len(self.current_batches):
            if self.stream:
                self.reload_buffer()
            else:
                # Indicates that we have finished an epoch 
                # inside a non-streaming DataLoader
                raise StopIteration

        # Indicates that we have finished an epoch inside
        # a streaming DataLoader
        if len(self.current_batches) == 0:
            raise StopIteration
        else:
            to_return = self.current_batches[self.pointer]

            # For computing encodings on the fly
            if self.encoding_model is not None and not self.no_encode:
                if self.encoding_is_seq:
                    encoding, _ = self.encoding_model(to_return)
                else:
                    _, encoding = self.encoding_model(to_return)

                to_return["encoding"] = encoding.detach()

            if self.sentence2mask or self.input_seq_from_bert:

                if self.encoding_model.name == "roberta":
                    if "masked_sentence" in to_return:
                        input_seq = self.encoding_model.bert_tokenizer([s.replace("[MASK]", "<mask>") for s in to_return["masked_sentence"]])["input_ids"]
                    else:
                        input_seq = self.encoding_model.bert_tokenizer([s.replace("[MASK]", "<mask>") for s in to_return["sentence"]])["input_ids"]
                else:
                    if "masked_sentence" in to_return:
                        input_seq = self.encoding_model.bert_tokenizer(to_return["masked_sentence"])["input_ids"]
                    else:
                        input_seq = self.encoding_model.bert_tokenizer(to_return["sentence"])["input_ids"]

                to_return["input_seq"] = input_seq
                to_return["target_lengths"] = [len(seq) for seq in to_return["input_seq"]]
                to_return["input_lengths"] = to_return["target_lengths"]
            
            self.pointer += 1

            return to_return


# Convert a dataset into batches
def batchify(dataset, batch_size, pad_src_left=False, pad_trg_left=False, uniform_src_length=None, uniform_trg_length=None):
    batched = []

    if len(dataset) == 0:
        return batched

    key_for_counting = "input_seq"
    if "input_seq" not in dataset[0]:
        key_for_counting = "sentence"


    this_batch = {}
    for key in dataset[0]:
        this_batch[key] = []

    for pair in dataset:
        for key in pair:
            if isinstance(pair[key], int):
                this_batch[key].append(pair[key])
            else:
                this_batch[key].append(pair[key][:])
            
        if len(this_batch[key_for_counting]) == batch_size:
            batched.append(pad_batch(this_batch, pad_src_left=pad_src_left, pad_trg_left=pad_trg_left, uniform_src_length=uniform_src_length, uniform_trg_length=uniform_trg_length))

            this_batch = {}
            for key in dataset[0]:
                this_batch[key] = []

    if len(this_batch[key_for_counting]) > 0:
        batched.append(pad_batch(this_batch, pad_src_left=pad_src_left, pad_trg_left=pad_trg_left, uniform_src_length=uniform_src_length, uniform_trg_length=uniform_trg_length))

    return batched

def max_length(sentence_list):
    lengths = [len(sentence.split()) for sentence in sentence_list]

    return max(lengths)

def mask_at_index(sentence, i):
    words = sentence.split()
    words[i] = "[MASK]"

    new_sentence = " ".join(words)

    return new_sentence

def maskify(example, i):
    new_batch = {}
    for key in example:
        new_batch[key] = example[key]
    
    sentence = example["sentence"]
    new_batch["masked_sentence"] = mask_at_index(sentence, i)

    return new_batch


# Convert a dataset into batches
def mask_batchify(dataset, batch_size):
    batched = []

    if len(dataset) == 0:
        return batched

    masked = []

    for example in dataset:
        sentence = example["sentence"]

        for i in range(len(sentence.split())):
            masked.append(maskify(example, i))

    random.shuffle(masked)

    this_batch = {}
    for key in masked[0]:
        this_batch[key] = []

    for example in masked:
        for key in example:
            this_batch[key].append(example[key])

        if len(this_batch["sentence"]) == batch_size:
            batched.append(this_batch)
            this_batch = {}

            for key in masked[0]:
                this_batch[key] = []


    if len(this_batch["sentence"]) != 0:
        batched.append(this_batch)

    return batched


# Add padding to input and target sequences to make
# all lengths uniform within a batch
# Also pad the filler and role sequences, if relevant
def pad_batch(batch, pad_src_left=False, pad_trg_left=False, uniform_src_length=None, uniform_trg_length=None):
    new_batch = {}
    
    if "input_seq" in batch:
        src = batch["input_seq"]
        batch_size = len(src)

        src_lengths = [len(seq) for seq in src]
        if uniform_src_length is None:
            src_length = max(src_lengths)
        else:
            src_length = uniform_src_length


        new_src = [[0]*src_length for seq in src]
        for index, seq in enumerate(src):
            if pad_src_left:
                new_src[index][-1*src_lengths[index]:] = seq
            else:
                new_src[index][:src_lengths[index]] = seq

        new_batch["input_seq"] = new_src
        new_batch["input_lengths"] = src_lengths


    if "target_seq" in batch:
        trg = batch["target_seq"]
        batch_size = len(trg)

        trg_lengths = [len(seq) for seq in trg]
        if uniform_trg_length is None:
            trg_length = max(trg_lengths)
        else:
            trg_length = uniform_trg_length

        new_trg = [[0]*trg_length for seq in trg]
        for index, seq in enumerate(trg):
            if pad_trg_left:
                new_trg[index][-1*trg_lengths[index]:] = seq
            else:
                new_trg[index][:trg_lengths[index]] = seq
   
        new_batch["target_seq"] = new_trg
        new_batch["target_lengths"] = trg_lengths

    if "encoding" in batch:
        if isinstance(batch["encoding"][0][0], float):
            # Encoding is a single vector
            new_batch["encoding"] = torch.FloatTensor(batch["encoding"]).unsqueeze(0).to(device=device)
        else:
            # Encoding is a list of vectors
            new_encoding = []
            for elt in batch["encoding"]:
                while len(elt) < src_length:
                    zeroes = [0]*len(batch["encoding"][0][0])
                    elt.append(zeroes) 
                new_encoding.append(elt)
            new_batch["encoding"] = torch.FloatTensor(new_encoding).transpose(0,1).to(device=device)

    if "fillers" in batch:
        filler_lengths = [len(seq) for seq in batch["fillers"]]
        filler_length = max(filler_lengths)

        new_fillers = [[0]*filler_length for seq in batch["fillers"]]
        for index, seq in enumerate(batch["fillers"]):
            new_fillers[index][:filler_lengths[index]] = seq

        new_batch["fillers"] = new_fillers


    if "roles" in batch:
        role_lengths = [len(seq) for seq in batch["roles"]]
        role_length = max(role_lengths)

        new_roles = [[0]*role_length for seq in batch["roles"]]
        for index, seq in enumerate(batch["roles"]):
            new_roles[index][:role_lengths[index]] = seq

        new_batch["roles"] = new_roles

    for key in batch:
        if key not in new_batch:
            new_batch[key] = batch[key][:]

    return new_batch


# Convert a file into sequences of token ids
def tokenize_format_analogies(fi, src_token2idx, input_sos=False, headers=False):
    fi = open(fi, "r")
    quartets = []

    first = True
    for line in fi:

        if first:
            first = False

            if headers:
                line_parts = line.strip().split("\t")
                index2header = {}

                for index, line_part in enumerate(line_parts):
                    index2header[index] = line_part

                continue
            else:
                index2header[0] = "seq0"
                index2header[1] = "seq1"
                index2header[2] = "seq2"
                index2header[3] = "seq3"

        line_parts = [seq.split() for seq in line.strip().split("\t")]
        if input_sos:
            for index, elt in enumerate(line_parts):
                line_parts[index] = ["<SOS>"] + line_parts[index]
        
        quartet = [[src_token2idx[token] for token in line_part] for line_part in line_parts[:-1]]
        indices = [int(x) for x in line_parts[-1]]

        example = {}
        for index, seq in enumerate(quartet):
            example[index2header[index]] = seq

        example["indices"] = indices

        quartets.append(example)


    return quartets

# Converts a file into batches of lines
def simple_batchify(filename, batch_size):
    fi = open(filename)
    batches = []

    current_batch = []

    for line in fi:
        current_batch.append(line.strip())
        if len(current_batch) == batch_size:
            batches.append(current_batch[:])
            current_batch = []

    if len(current_batch) != 0:
        batches.append(current_batch[:])

    return batches

# simple_batchify, but where the file is
# the output of parsing
def parse_batchify(filename, batch_size):
    fi = open(filename)
    batches = []

    current_batch = [[], [], []]

    for line in fi:
        parts = line.strip().split("\t")
        current_batch[0].append(parts[0])
        current_batch[1].append(parts[1])
        current_batch[2].append(parts[2])

        if len(current_batch[0]) == batch_size:
            batches.append(current_batch[:])
            current_batch = [[], [], []]

    if len(current_batch) != 0:
        batches.append(current_batch[:])

    return batches


# Return the number of parameters
# in a PyTorch model
def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())

    return total_params



# pairs is a list of pairs of lists
# fo is the name of an output fille
# This prints pairs to fo: one pair
#    per line, with a tab between
#    the first and second elements
#    of the pair, and with the sequences
#    space-delimited
def pairs_to_file(pairs, fo):
    fo = open(fo, "w")
    fo.write("input_seq\ttarget_seq\n")
    for pair in pairs:
        fo.write(" ".join([str(x) for x in pair[0]]) + "\t" + " ".join([str(x) for x in pair[1]]) + "\n")

# Same as pairs_to_file, but with 4 sequences (for analogies)
def quartets_to_file(quartets, fo):
    fo = open(fo, "w")
    fo.write("seq0\tseq1\tseq2\tseq3\tindices\n")
    for quartet in quartets:
        fo.write("\t".join([" ".join([str(x) for x in quartet[0][index]]) for index in range(4)]) + "\t" + " ".join([str(y) for y in quartet[1]]) + "\n")








