# SimCLR Transfer Learning for Medical Image Classification

**Author:** Elshaday Desalegn  
**Date:** January 5, 2025

## Overview

This project investigates how well SimCLR self-supervised representations pretrained on natural images transfer to medical image classification.

Using a ResNet-50 model pretrained on CIFAR-100, three experiments were conducted to compare partial fine-tuning and linear evaluation on medical and non-medical datasets.

The main research question was whether representations learned from natural images can generalize to grayscale medical images such as chest X-rays.

## Datasets

**1. Chest X-Ray Pneumonia (Medical)**
- Classes: NORMAL, PNEUMONIA
- Training: 2,413
- Validation: 269
- Testing: 624

**2. Cats vs Dogs (Non-medical)**
- Classes: Cat, Dog
- Training: 4,694
- Validation: 522
- Testing: 624

## Model and Training

- **Architecture:** ResNet-50
- **Pretrained model:** [SimCLR on CIFAR-100](https://huggingface.co/edadaltocg/resnet50_simclr_cifar100)
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Batch size:** 64
- **Epochs:** 6
- **Loss:** Class-weighted cross-entropy for medical experiments

## Experiments and Results

| Experiment | Strategy | Test Accuracy |
|---|---|---:|
| Medical | Partial fine-tuning | **84.29%** |
| Medical | Linear evaluation | 80.45% |
| Cats vs Dogs | Linear evaluation | 80.93% |

### Key Findings

- Partial fine-tuning achieved the highest accuracy on the medical dataset.
- Linear evaluation showed similar performance across medical and non-medical datasets.
- Fine-tuning improved medical classification accuracy by 3.84 percentage points.
- The fine-tuned model achieved 0.99 pneumonia recall but only 0.58 normal recall, showing a bias toward pneumonia predictions.

## Conclusion

The experiments demonstrate that SimCLR representations pretrained on natural images can transfer to medical image classification, even when the target images are grayscale.

Allowing limited adaptation of the pretrained backbone improved overall performance. However, class-specific evaluation revealed that high accuracy does not necessarily mean balanced classification performance.

## Technologies

- Python
- PyTorch
- SimCLR
- ResNet-50
- Hugging Face
- Scikit-learn

