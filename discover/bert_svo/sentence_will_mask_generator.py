
from random import shuffle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default="data/sentences_will_mask")
args = parser.parse_args()

nouns_sg = ["professor", "student", "president","judge","senator","secretary","doctor","lawyer","scientist","banker","tourist","manager","artist","author","actor","athlete", "teacher", "engineer", "accountant", "architect", "chef", "journalist", "photographer", "farmer", "ambassador", "astronaut", "astronomer", "blacksmith", "baker", "barber", "biologist", "butler", "chemist", "composer", "cartoonist", "coach", "captain", "carpenter", "dancer", "director", "drummer", "detective", "explorer", "economist", "editor", "governor", "gardener", "illustrator", "intern", "inventor", "journalist", "linguist", "manager", "magician", "mayor", "miner", "mathematician", "musician", "novelist", "nurse", "painter", "philosopher", "physicist", "politician", "programmer", "pilot", "poet", "reporter", "referee", "sailor", "student", "spy", "translator", "treasurer", "technician", "tutor", "umpire", "violinist", "writer", "librarian"]


sentences = []

for subj in nouns_sg:
    for dobj in nouns_sg:
        sentence = " ".join(["the", subj, "will", "[MASK]", "the", dobj, "."])
        sentences.append(sentence)

shuffle(sentences)
count = len(sentences)

train_set = sentences[:int(count*0.8)]
valid_set = sentences[int(count*0.8):int(count*0.9)]
test_set = sentences[int(count*0.9):]

fo_train = open(args.prefix + ".train", "w")
fo_valid = open(args.prefix + ".valid", "w")
fo_test = open(args.prefix + ".test", "w")

fo_train.write("sentence\tword_index\n")
for line in train_set:
    fo_train.write(line + "\t3\n")

fo_valid.write("sentence\tword_index\n")
for line in valid_set:
    fo_valid.write(line + "\t3\n")

fo_test.write("sentence\tword_index\n")
for line in test_set:
    fo_test.write(line + "\t3\n")








