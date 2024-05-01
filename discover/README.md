DISCOVER

DISsecting COmpositionality in VEctor Representations

# Requirements

0. Create a python venv to run code in:
```
# We used python 3.8.1
python -m venv .venv
```

1. Activate the venv
```
source .venv/bin/activate
```

2. Install requirements
```
pip install -U pip setuptools wheel
pip install torch
pip install numpy
pip install transformers
```

3. Prepare the data for the decomposition: I downloaded the embedding files that Chandan shared and then used `bert_svo/data/fmri_sentence_loader.py` to process them into the format that the DISCOVER code looks for. This format involves a file with 3 columns separated by tabs. The first line of the file should be a header line (with headers "input_seq", "target_seq", and "encoding"), and then the rest of the lines should have those three values. "input_seq" is the sentence whose encoding we are analyzing; "target_seq" is not relevant for these experiments, so I just made it a copy of "input_seq"; and "encoding" is the encoding. Note that I did not upload the data files to GitHub due to their size.

4. I then ran the DISCOVER analysis process using the Python commands in `bert_svo/tpr_fmri_bow_768.scr` and `bert_svo/tpr_fmri_ltr_768.scr`




