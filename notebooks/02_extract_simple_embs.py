import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import sys
from tpr.models.fmri_module import fMRIModule


if __name__ == '__main__':
    mod = fMRIModule(checkpoint="facebook/opt-30b")
    subject = 'UTS03'
    for sentence_file in ['simple_sentences.test', 'simple_sentences.valid', 'simple_sentences.train']:
        sentence_file_full = join('..', 'data', 'synthetic', sentence_file)
        df = pd.read_csv(sentence_file_full, sep='\t')
        texts = df['sentence'].values.tolist()
        out_file = join('..', 'data', 'synthetic', f'embs_{sentence_file}')
        if not os.path.exists(out_file):
            print(f'Extracting embeddings for {sentence_file}')
            embs = mod._get_embs(texts)
            joblib.dump(embs, out_file)
        # out_file = join('..', 'data', 'synthetic',
        #                 f'embs_fmri_{subject}_{sentence_file}')
        # if not os.path.exists(out_file):
        #     print(f'Extracting fMRI embeddings for {sentence_file}')
        #     embs_fmri = mod(texts, subject)
        #     joblib.dump(embs_fmri, out_file)
