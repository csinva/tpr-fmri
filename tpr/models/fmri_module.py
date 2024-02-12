from copy import deepcopy
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import numpy.random
import torch
import os.path
import torch.cuda
from os.path import dirname, join
import pickle as pkl
import imodelsx.util
import imodelsx
from spacy.lang.en import English
import sklearn.preprocessing
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import datasets
from typing import List, Union
import logging
import pandas as pd
from collections import defaultdict

# required saved files
from tpr.config import SAVE_DIR_FMRI
VOXEL_SELECTIVITY_JBL = join(SAVE_DIR_FMRI, "voxel_lists",
                             "{subject}_voxel_selectivity.jbl")
ENCODING_WEIGHTS_JBL = join(SAVE_DIR_FMRI, "{model_dir}",
                            "model_weights", "wt_{subject}.jbl")
PREPROC_JBL = join(SAVE_DIR_FMRI, "{model_dir}", "preproc.pkl")
CORRS_JBL = join(SAVE_DIR_FMRI, "{model_dir}", "voxel_performances",
                 "{subject}_voxel_performance.jbl")
ROIS_ANAT_JBL = join(SAVE_DIR_FMRI, "voxel_rois",
                     "voxel_anat_rois", "{subject}_voxel_anat_rois.jbl")
ROIS_FUNC_JBL = join(SAVE_DIR_FMRI, "voxel_rois",
                     "voxel_func_rois", "{subject}_voxel_func_rois.jbl")


class fMRIModule:
    def __init__(
        self,
        voxel_num_best: int = 0,
        subject: str = "UTS01",
        checkpoint="facebook/opt-30b",
    ):
        """
        Params
        ------
        voxel_num_best: int
            Which voxel to predict (0 for best-predicted voxel, then 1, 2, ...1000)
        """

        # load llm model & tokenizer
        assert checkpoint in ["facebook/opt-30b",
                              "decapoda-research/llama-30b-hf"]
        self.checkpoint = checkpoint
        self.model_dir = {
            "facebook/opt-30b": "opt_model",
            "decapoda-research/llama-30b-hf": "llama_model",
        }[checkpoint]
        if checkpoint == "decapoda-research/llama-30b-hf":
            self.tokenizer = LlamaTokenizer.from_pretrained(self.checkpoint)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, device_map="auto", torch_dtype=torch.float16
        )

        # load fmri-specific stuff
        self._init_fmri(subject)
        self._init_fmri_voxel(voxel_num_best, subject)

    def _init_fmri(self, subject: str):
        print("initializing fmri...")

        # load voxel indexes
        self.voxel_idxs = joblib.load(
            VOXEL_SELECTIVITY_JBL.format(subject=subject))

        # load weights
        weights = joblib.load(ENCODING_WEIGHTS_JBL.format(
            model_dir=self.model_dir, subject=subject))
        self.weights = weights[:, self.voxel_idxs]
        self.preproc = pkl.load(
            open(PREPROC_JBL.format(model_dir=self.model_dir), "rb"))
        self.ndel = 4

        # load corrs
        self.corrs = joblib.load(CORRS_JBL.format(
            model_dir=self.model_dir, subject=subject
        ))
        if self.checkpoint == "decapoda-research/llama-30b-hf":
            self.corrs = self.corrs[0]

    def _init_fmri_voxel(self, voxel_num_best: Union[int, np.ndarray[int]], subject: str):
        if isinstance(voxel_num_best, np.ndarray):
            voxel_num_best = voxel_num_best.astype(int)
        self.voxel_num_best = voxel_num_best
        self.subject = subject

        # load corr performance
        if isinstance(voxel_num_best, int):
            self.corr = self.corrs[self.voxel_idxs[voxel_num_best]]

    def _get_embs(self, X: List[str]):
        """
        Returns
        -------
        embs: np.ndarray
            (n_examples, 7168)
        """
        embs = []
        layer = {
            "facebook/opt-30b": 33,
            "decapoda-research/llama-30b-hf": 18,
        }[self.checkpoint]
        for i in tqdm(range(len(X))):
            text = self.tokenizer.encode(X[i])
            inputs = {}
            inputs["input_ids"] = torch.tensor([text]).int()
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape)

            # Ideally, you would use downsampled features instead of copying features across time delays
            emb = (
                list(self.model(**inputs, output_hidden_states=True)
                     [2])[layer][0][-1]
                .cpu()
                .detach()
                .numpy()
            )
            embs.append(emb)
        return np.array(embs)

    def __call__(self, X: List[str], return_all=False) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        self.voxel_num_best may be a list, in which case it will return a 2d array (len(X), len(self.voxel_num_best))
        """
        # get opt embeddings
        embs = self._get_embs(X)
        torch.cuda.empty_cache()

        # apply StandardScaler (pre-trained)
        embs = self.preproc.transform(embs)

        # apply fMRI transform
        embs_delayed = np.hstack([embs] * self.ndel)
        preds_fMRI = embs_delayed @ self.weights

        if return_all:
            return preds_fMRI  # self.weights was already restricted to top voxels
        else:
            pred_voxel = preds_fMRI[
                :, np.array(self.voxel_num_best).astype(int)
            ]  # select voxel (or potentially many voxels)
            return pred_voxel


VOXELS_IDXS_DICT = {
    subject: joblib.load(VOXEL_SELECTIVITY_JBL.format(subject=subject))
    for subject in ["UTS01", "UTS02", "UTS03"]
}


def convert_module_num_to_voxel_num(module_num: int, subject: str):
    return VOXELS_IDXS_DICT[subject][module_num]


def get_roi(voxel_num_best: int = 0, roi_type: str = "anat", subject: str = "UTS01"):
    if roi_type == "anat":
        rois = joblib.load(ROIS_ANAT_JBL.format(subject=subject))
    elif roi_type == "func":
        rois = joblib.load(ROIS_FUNC_JBL.format(subject=subject))
    voxel_idxs = joblib.load(VOXEL_SELECTIVITY_JBL.format(subject=subject))
    voxel_idx = voxel_idxs[voxel_num_best]
    return rois.get(str(voxel_idx), "--")


if __name__ == "__main__":
    # mod = fMRIModule(
    # voxel_num_best=[1, 2, 3], checkpoint="decapoda-research/llama-30b-hf"
    # )
    # X = ["I am happy", "I am sad", "I am angry"]
    # print(X[0][:50])
    # resp = mod(X[:3])
    # print(resp.shape)
    # print(resp)

    for subj in ["UTS01", "UTS02", "UTS03"]:
        print(
            joblib.load(
                f"/home/chansingh/mntv1/deep-fMRI/rj_models/llama_model/voxel_performances/{subj}_voxel_performance.jbl"
            )[0].mean()
        )
