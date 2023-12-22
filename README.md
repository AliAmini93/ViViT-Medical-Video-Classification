# ViViT Medical Video Classification

## Overview

This repository contains the complete code for setting up, training, and evaluating the ViViT model on a selected video dataset from MedMNIST. It showcases the model's ability to process and learn from complex video data, providing an example of applying machine learning techniques in the field of medical video analysis. A pre-trained model is not used, and the training process starts from scratch.

## Key Features

- **ViViT Model Implementation:** Utilizes the ViViT architecture, a transformer-based model designed for handling video data, to perform classification tasks on medical videos.
- **MedMNIST Dataset:** Leverages the specialized datasets provided by MedMNIST, designed for biomedical image(2D,3D) classification tasks. The MedMNIST dataset is diverse, encompassing various types of medical images, including but not limited to organ images, dermoscopic images, and blood cell microscopy images. It offers a comprehensive platform for medical image classification with multiple classes, each representing a distinct medical condition or type. MedMNIST v2 enhances this benchmark by offering a large-scale, lightweight dataset for both 2D and 3D biomedical image classification, making it more versatile and applicable to a wider range of medical imaging tasks.

## Configuration and Setup

### Dataset Configuration
- **Dataset:** OrganMNIST3D, a part of the MedMNIST dataset, specifically designed for 3D organ image classification.
- **Batch Size:** 32 - Determines the number of samples processed before the model is updated.
- **Auto-Tuning:** Enabled for efficient data loading.
- **Volume Shape:** (28, 28, 28, 1) - The shape of the input 3D volumes.
- **Total Classes:** 11 - The number of distinct classes in the dataset.

### Model Hyperparameters and Architecture
- **Learning Rate:** 1e-4 - The initial learning rate for the optimizer.
- **Weight Decay Rate:** 1e-5 - Used for regularization to prevent overfitting.
- **Training Epochs:** 60 - The total number of passes through the training dataset.
- **Tubelet Embedding Patch Dimensions:** (8, 8, 8) - The size of each patch extracted from the input volumes for processing by the ViViT model.
- **Patch Count:** Computed based on the volume shape and patch dimensions.

### ViViT Architecture Specifications
- **Normalization Epsilon:** 1e-6 - A small constant is added to the denominator to improve numerical stability in layer normalization.
- **Embedding Dimension:** 128 - The size of the output space for embeddings.
- **Attention Heads:** 8 - The number of attention heads in the transformer layers.
- **Transformer Layers:** 8 - The number of layers in the ViViT model.

This configuration sets the stage for training a robust and effective model capable of handling the complexities of medical video data. The chosen parameters are optimized to balance performance and computational efficiency.

## Tubelet Embedding

In Vision Transformers (ViTs), an image is divided into patches, which are then spatially flattened in a process known as tokenization. For a video, one can repeat this process for individual frames. **Uniform frame sampling**, as suggested by the authors, is a tokenization scheme in which frames are sampled from the video clip and undergo simple ViT tokenization.

| ![uniform frame sampling](https://i.imgur.com/aaPyLPX.png) |
| :--: |
| Uniform Frame Sampling [Source](https://arxiv.org/abs/2103.15691) |

**Tubelet Embedding** differs in terms of capturing temporal information from the video. First, volumes are extracted from the video — these volumes contain patches of the frame and the temporal information as well. The volumes are then flattened to build video tokens.

| ![tubelet embedding](https://i.imgur.com/9G7QTfV.png) |
| :--: |
| Tubelet Embedding [Source](https://arxiv.org/abs/2103.15691) |


