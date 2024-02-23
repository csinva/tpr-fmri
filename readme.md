This repo copies a lot of code from [encoding-model-scaling-laws](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main), which is the repo for the paper "Scaling laws for language encoding models in fMRI" ([antonello, vaidya, & huth, 2023](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main?tab=readme-ov-file)). See the cool results there! It also copies a lot of code from the repo for [SASC](https://github.com/microsoft/automated-explanations/tree/main).

 
# Organization
- clone and run `pip install -e .`, resulting in a package named `tpr` that can be imported
- `data`: contains scripts and generated text for synthetic experiments
- `tpr`: contains main code for modeling (e.g. model architecture)
- `notebooks`: experiments in jupyter notebooks