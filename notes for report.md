# Image-to-Image Translation with Pix2Pix on the Maps Dataset

Author 1 Name, Author 2 Name

Master in Artificial Intelligence — Computer Vision II, 2025/2026


## Abstract

This work presents an implementation of the Pix2Pix conditional GAN for image-to-image translation on the Maps dataset, where the task is to generate realistic satellite images from geographic label maps. We implement a baseline model following the original architecture proposed by Isola et al. [1], consisting of a U-Net generator with skip connections and a PatchGAN discriminator. We then explore data augmentation as an improvement over the baseline and compare both approaches quantitatively using MAE, PSNR, and SSIM. The code and trained models are publicly available at: [REPO_LINK].


## 1. Introduction

Image-to-image translation refers to the problem of learning a mapping between an input image domain and an output image domain. Pix2Pix [1] addresses this through conditional adversarial training, where a generator learns to produce realistic outputs while a discriminator tries to distinguish generated images from real ones. The adversarial loss is combined with an L1 reconstruction loss to encourage both perceptual quality and pixel-level accuracy.

The Maps dataset [6] contains 500 paired images at 600×600 resolution. Each pair consists of a satellite photograph (left half) and its corresponding geographic label map (right half). The objective is to translate from label maps to realistic satellite images.

This report describes our baseline implementation (Task 1), a data augmentation improvement (Task 2), and the inference demo (Task 3).


## 2. Technical Approach

### 2.1 Data Preparation

The 500 paired images are split into training (70%), validation (15%), and test (15%) sets using a fixed random seed (42) for reproducibility. Each image is split along its center into the target satellite image and the input label map. During preprocessing, images are resized to 600×600 and normalized to the range [-1, 1].

### 2.2 Generator Architecture

The generator follows a U-Net encoder-decoder structure with skip connections. The encoder consists of six downsampling blocks, each using a 4×4 convolution with stride 2, batch normalization (except the first block), and LeakyReLU (slope 0.2). The feature progression is: 64 → 128 → 256 → 512 → 512 → 512.

A bottleneck layer applies a 3×3 convolution with ReLU at the deepest level.

The decoder mirrors the encoder with six upsampling blocks using 4×4 transposed convolutions, batch normalization, and ReLU. The three deepest decoder blocks apply dropout (p=0.5) for regularization. Skip connections concatenate encoder features to decoder features at each spatial resolution, preserving fine-grained details that would otherwise be lost through downsampling. The final layer is a 3×3 convolution followed by Tanh activation, producing output in [-1, 1].

### 2.3 Discriminator Architecture

The discriminator is a PatchGAN that classifies overlapping patches of the image as real or fake, rather than producing a single scalar output. It receives the concatenation of the input label map and the target/generated image (6 channels total). The network applies four convolutional blocks with 4×4 kernels: three with stride 2 for downsampling and one with stride 1, followed by a final 4×4 convolution that outputs a single-channel feature map. Batch normalization is used in all blocks except the first, and LeakyReLU (slope 0.2) is the activation throughout. For a 600×600 input, the output is a spatial map where each value corresponds to a local region of the input.

### 2.4 Training Procedure

Both networks are trained with Adam (lr = 2×10⁻⁴, β₁ = 0.5, β₂ = 0.999). The total generator loss combines the adversarial loss (BCE with logits) and the L1 reconstruction loss weighted by λ = 100, following the original paper:

    L_G = L_GAN + 100 · L_L1

The discriminator loss is the average of the real and fake classification losses:

    L_D = 0.5 · (L_real + L_fake)

Training runs for up to 100 epochs with early stopping (patience = 8 epochs) based on validation L1 loss. The batch size is 8 and the best checkpoint is saved based on the lowest validation L1.

### 2.5 Task 2: Data Augmentation

For the improvement, we apply random horizontal flipping with probability 0.5 during training. The flip is applied jointly to both the label map and the target image to preserve spatial correspondence. All other hyperparameters remain identical to the baseline to isolate the effect of augmentation. The augmented model is trained and evaluated separately, with its own checkpoint directory.

### 2.6 Evaluation Metrics

We evaluate on the held-out test set (75 images) using three metrics:
- **MAE** (Mean Absolute Error): average pixel-level difference between generated and real images.
- **PSNR** (Peak Signal-to-Noise Ratio): measures reconstruction quality in dB; higher is better.
- **SSIM** (Structural Similarity Index): captures perceptual similarity considering luminance, contrast, and structure; range [0, 1], higher is better.

### 2.7 Inference Demo (Task 3)

A command-line inference script accepts a trained checkpoint and either a single image or a folder of images. It supports both paired input (where the label half is extracted automatically) and unpaired input (standalone label maps). The output is a side-by-side visualization showing the input, the generated image, and optionally the ground truth. A Jupyter notebook demo is also provided for interactive visualization.

Public repository: [REPO_LINK]


## 3. Results

### 3.1 Quantitative Results

<!-- FILL AFTER TRAINING: replace the placeholder values below with actual metrics -->

| Model             | MAE    | PSNR   | SSIM   |
|-------------------|--------|--------|--------|
| Baseline          | X.XXXX | XX.XX  | X.XXXX |
| + Augmentation    | X.XXXX | XX.XX  | X.XXXX |

<!-- END FILL -->

The augmented model is expected to show improvements due to the increased effective training set diversity from horizontal flipping. Since geographic features in satellite imagery do not have a strong horizontal orientation bias, this augmentation is a reasonable transformation that does not introduce unrealistic training samples.

### 3.2 Qualitative Results

<!-- FILL: include 2-3 side-by-side images (input / generated / ground truth) from the test set for both baseline and augmented models. Describe what you observe: sharpness, color accuracy, artifacts, etc. -->

### 3.3 Training Behavior

<!-- FILL: briefly describe how training loss evolved, at which epoch early stopping triggered, any observations about generator vs discriminator balance -->


## 4. Conclusion

We implemented a Pix2Pix model for translating geographic label maps to satellite images on the Maps dataset. The baseline follows the standard U-Net + PatchGAN architecture from [1] with a combined adversarial and L1 loss. As an improvement, we applied horizontal flip augmentation during training, which increases the effective diversity of the training data without altering the model architecture or loss function.

<!-- FILL: 1-2 sentences summarizing actual results once metrics are available -->

The code, trained models, and demo are available at [REPO_LINK].
