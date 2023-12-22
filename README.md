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

## Positional Encoding

### Concept
In the context of transformers, it's crucial to provide the model with some information about the order or position of the tokens in the sequence. This is where the `PositionalEncoder` layer comes into play. It adds positional information to the encoded video tokens, which is essential for the model to understand the sequence of the input data, especially when dealing with video frames where temporal information is key.

### Implementation in Code
The `PositionalEncoder` class is designed to generate and add positional embeddings to the token sequences.

## Video Vision Transformer (ViViT)

### Overview
The ViViT architecture, a derivative of the Vision Transformer (ViT), is designed to handle the complexities of video data. The authors of ViViT suggest four variants:

- Spatio-temporal attention
- Factorized encoder
- Factorized self-attention
- Factorized dot-product attention

In this project, we focus on implementing the **Spatio-temporal attention** model for its simplicity and effectiveness in capturing both spatial and temporal dimensions of video data. The following implementation is heavily inspired by the [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/) and the [official ViViT repository](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) implemented in JAX.

### Model Implementation

The `build_vivit_model` function constructs the ViViT model with the following key steps:

1. **Input Tensor Initialization:** Defines the input shape of the video data.
2. **Tubelet Embedding:** Generates patches from the input video and projects them into an embedding space using the `TubeletEmbedding` class.
3. **Positional Encoding:** Adds positional information to the patches using the `PositionalEncoder` class.
4. **Transformer Layers:** Constructs several layers of the transformer. Each layer includes:
   - Layer normalization and multi-head self-attention for processing the sequence of embeddings.
   - Residual connections to facilitate deeper architecture without the vanishing gradient problem.
   - Additional layer normalization followed by dense layers for further feature extraction.

5. **Final Representation:** Applies layer normalization and global average pooling to the output of the last transformer layer, preparing it for classification.
6. **Classification Layer:** A dense layer that outputs probabilities for each class.



