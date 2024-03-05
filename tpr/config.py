# get path to repo
import os
import os.path
from os.path import dirname, join
import sys
path_to_repo = dirname(dirname(os.path.abspath(__file__)))

SAVE_DIR_FMRI = join(path_to_repo, 'fmri_voxel_data')
SAVE_DIR_LINEAR_WEIGHTS = '/home/chansingh/mntv1/deep-fMRI/rj_models/'
