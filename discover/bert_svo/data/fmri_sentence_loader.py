

import numpy as np
import joblib


for split in ["valid", "test", "train"]:
    fi_sentence = open("/Users/tommccoy/Documents/tommccoy/Documents/GitHub/tpr-fmri/data/synthetic/simple_sentences." + split, "r")

    first = True
    sentences = []
    for line in fi_sentence:
        if first:
            first = False
            continue
        sentence = line.strip().split("\t")[0]
        sentences.append(sentence)


    fo = open("fmri_simple." + split + "_encodings", "w")

    fo.write("input_seq\ttarget_seq\tencoding\n")

    array = joblib.load('/Users/tommccoy/Downloads/embs/embs_simple_sentences.' + split)

    for index, sentence in enumerate(sentences):
        emb = array[index].tolist()

        fo.write(sentence + "\t" + sentence + "\t" + " ".join([str(x) for x in emb]) + "\n")







