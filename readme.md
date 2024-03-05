# Setup
- clone the repo and run `pip install -e .`, resulting in a package named `tpr` that can be imported
- download the linear encoding weights
  - OPT: download the weights [here](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230422424869) and move to the folder `tpr-embeddings/fmri_voxel_data/llama_model/model_weights`
    - rename the weights in that folder to `wt_UTS01.jbl`, `wt_UTS01.jbl`, `wt_UTS03.jbl`
  - LLaMA: download the weights [here](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230422427269) and move to the folder `tpr-embeddings/fmri_voxel_data/llama_model/model_weights`
    - rename the weights in that folder to `wt_UTS01.jbl`, `wt_UTS01.jbl`, `wt_UTS03.jbl`
- if everything is set up properly, you should be able to run the [notebooks/01_module_example.ipynb](notebooks/01_module_example.ipynb) notebook without any issues

# Organization
- `data`: contains text and scripts for text to evaluate the models on
  - `data/fmri`: shows a sample test story of the type that the models were trained on
- `voxel_data`: contains metadata on the fMRI experiments
- `tpr`: contains main code for modeling (e.g. model architecture)
- `notebooks`: experiments in jupyter notebooks

# Reference
This repo copies a lot of code from [encoding-model-scaling-laws](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main), which is the repo for the paper "Scaling laws for language encoding models in fMRI" ([antonello, vaidya, & huth, 2023](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main?tab=readme-ov-file)). See the cool results there! It also copies a lot of code from the repo for [SASC](https://github.com/microsoft/automated-explanations/tree/main).

 
