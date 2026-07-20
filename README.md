# IDH Mutation Prediction
 A collection of experiments on predicting IDH mutation status in glioma from multimodal MRI (T1, T1CE, T2, FLAIR), with an emphasis on generalizing across hospital sites and handling missing modalities.

## Datasets
- **UTSW Glioma dataset** (~622 subjects)
- **TCGA-LGG + TCGA-GBM**
- **UCSF-PDGM**
Used together to test cross-site generalization rather than just single-site accuracy. See DATASETS/ for loading/splits.

## Approach
Backbone: 3D ResNet-18 for volumetric classification over the four MRI modalities
Cross-site harmonization: N4ITK bias field correction + Nyúl intensity normalization, applied before training to reduce scanner/protocol shift between sites (see Normalisation_testing/)
Robustness to missing modalities:
Non-local self-attention blocks
A modality-dropout curriculum during training (randomly withholding modalities so the model doesn't over-rely on any one)
Continual learning via Elastic Weight Consolidation (EWC), tested in CONTINUAL LEARNING/


