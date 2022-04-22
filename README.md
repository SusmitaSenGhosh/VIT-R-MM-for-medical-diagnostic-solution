# VIT-R-MM-for-medical-diagnostic-solution
This repository Keras implementation of the experimets conducted in '**Vision Transformer with Reversed Order Residual Blocks and Multi-Layer Perceptron Mixer for Image based Diagnostic Solution**'. Codes are verified on python3.8.5 with tensorflow version '2.4.1'. Other dependencies are NumPy, cv2, sklearn, matplotlib, random, os, etc.

**Data Resources:**

1. Colorectal Histology: https://www.kaggle.com/kmader/colorectal-histology-mnist
2. ISIC18: https://challenge2018.isic-archive.com/task3/
3. CBIS_DDSM: https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm
4. Chestxray: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
5. Fundus: https://github.com/deepdrdoc/DeepDRiD
6. PBC: https://data.mendeley.com/datasets/snkd93bnjr/1

**Data preparation:** For data preparation use data_prep.py.

**Training and Evaluation:**
1. ViT/ViT-R/ViT-MM/ViT-R-MM/ResNet50/ResNet-ViT/ResNet-ViT-R-MM: Execute train_and_test_models.py
2. Aux-ResNet-ViT-R-MM: Refer to train_and_test_models.py
3. To plot ACSF curve for training and testing of ViT and ViT-R-MM use ViT_ViTRMM_graph.py.
4. For visual interpretation of ViT/ViT-R-MM, ResNet50/ResNet-ViT/ResNet-ViT-R-MM and Aux-ResNet-ViT-R-MM use view_gradients_and_attention_maps.py, view_gradients_and_attention_maps_hybrid.py and view_gradients_and_attention_maps_aux.py respectively.
