This document contains a brief summary of how the 9th position in 
[Mafat Radar Challenge](https://competitions.codalab.org/competitions/25389#learn_the_details) 
was obtained.

# Summary

 - Ensemble of 55 models (11 * 5 CV folds)
 - EfficientNet B1, B2, B3, B4 and B5
 - Main data augmentation techniques
   - Rolling on X an Y axis (np.roll)
   - Horizontal/Vertical flips
   - SpecAugment (Time and Frequency Masks)
   - Stratified sampling (p=0.8 for training data p=0.2 for auxiliary data)
 - Centered spectrogram based on doppler burst
 
# Data preprocessing

Data preprocessing can be found on [this notebook](mafat_radar_challenge/notebooks/MAFAT_merge_datasets.ipynb). 
This notebook was used to obtain the spectrograms and CSVs used for training, 
validation and testing.
 - Join Training, Auxiliary Experiment and Auxiliary Synthetic spectrograms
 - Center them using `doppler burst` information.
 - Divide them in 5 training and 5 validation sets.
 - Center Auxiliary Background, Full Public and Private test sets.

# Training

Configuration files for training the 55 models and the correspondent 
data augmentation used can be found [here](experiments/ensemble_one).
On each configuration file you can see the difference between each 
model: architecture, loss, etc.

```
$ conda activate mafat_radar_challenge
$ mafat_radar_challenge train -c <path to config file>
```

# SWA

SWA was applied on:

 - Best epoch model and model of next epoch
 - Best epoch model and models of next two epochs

 For instance, if best model with respect to validation set was obtained on epoch 10,
 model averaging of epochs 10-11 and epochs 10-11-12 were done.

[This notebook](mafat_radar_challenge/notebooks/MAFAT_swa_ensembling.ipynb) was used 
for this purpose. Only when the SWA increased performance on the validation set, the 
resulting "weight-averaged" model was used.

# TTA (Test time augmentation)

Test Time Augmentation was used on inference. Inferences were obtained on original image, 
horizontally flipped image and vertically flipped image. Then, the inferences were
averaged. Notebook used for the final inference on the Private dataset is
[this notebook](mafat_radar_challenge/notebooks/MAFAT_ensemble_one.ipynb).

