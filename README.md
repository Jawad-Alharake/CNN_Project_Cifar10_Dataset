# CIFAR-10 Image Classification: Custom CNN vs. Transfer Learning

## Project Scope
This project evaluates and compares the performance of a Custom Convolutional Neural Network (built from scratch) against a pre-trained MobileNetV2 model using Transfer Learning. Both models are trained and evaluated on the **CIFAR-10** dataset, which consists of 60,000 color images across 10 distinct classes.

The primary objective is to analyze the trade-offs between computational cost, training time, parameter efficiency, and predictive accuracy when utilizing custom baseline architectures versus established, pre-trained feature extractors.

---

## Workspace Directory Structure

```text
├── dataset.py                   # Data ingestion, augmentation, and 128x128 DataLoader pipeline
├── models.py                    # PyTorch architecture blueprints (Custom CNN & MobileNetV2)
├── train.py                     # Main execution engine, training loops, and evaluation metrics
├── data/                        # Directory automatically created to store the raw CIFAR-10 dataset
├── dataset_samples.png          # Generated visualization of augmented training data
├── custom_cnn_from_scratch_history.png          # Generated loss/accuracy curves for Custom CNN
├── custom_cnn_from_scratch_confusion_matrix.png # Generated heatmap of Custom CNN predictions
├── transfer_learning_mobilenetv2_history.png    # Generated loss/accuracy curves for MobileNetV2
├── transfer_learning_mobilenetv2_confusion_matrix.png # Generated heatmap of MobileNetV2 predictions
└── README.md                    # Project documentation

```

---

## Requirements & Installation

This project is built using Python and PyTorch. To run the pipeline, ensure the following dependencies are installed:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```

---

## Dataset Handling & Preprocessing

This project utilizes the standard `torchvision.datasets.CIFAR10` module to download and load the dataset directly from its official, hardcoded source. The `dataset.py` pipeline dynamically handles the extraction, splitting, and continuous on-the-fly augmentation of the dataset into memory-efficient batches.

**Performance Optimization (Resolution Scaling):**
Raw CIFAR-10 images are 32x32 pixels. Instead of upscaling to MobileNetV2's traditional 224x224 (which is computationally heavy and causes severe image blurring), the data pipeline dynamically resizes all images to **128x128**. This provides a massive reduction in CPU/RAM usage while retaining clean edge data for the models to learn from.

---

## Architectures & Training Methodology

### 1. Custom CNN (Baseline Model)

A lightweight convolutional neural network built from scratch. It utilizes a pure "vanilla" feature extraction pipeline (no intermediate batch normalization or spatial dropout) consisting of 3 Convolutional/MaxPooling blocks.

* **Global Average Pooling (GAP):** Replaces standard spatial flattening to drastically reduce parameter count and prevent overfitting.
* **Pyramidal Classification Head:** A dual-layer dense network with stepped dropout (45% -> 25%) to safely funnel features to the final 10 classes.
* **Optimization:** Adam optimizer (`lr=0.001`).

### 2. Transfer Learning (MobileNetV2)

Utilizes Google's `MobileNet_V2` pre-trained on ImageNet. The core feature extraction layers are frozen, while the final classification head is replaced. Thanks to its native GAP layer, MobileNetV2 dynamically accepts our optimized 128x128 tensors without dimension mismatch errors.

* **Optimization:** Adam optimizer (`lr=0.0001`) applied *only* to the new classification head.
* **Safety Lock:** Batch Normalization (`BatchNorm2d`) layers in the frozen backbone are programmatically forced into `.eval()` mode during training to prevent the corruption of pre-trained ImageNet statistics.

### Pipeline Features (`train.py`)

* **Early Stopping:** Halts training if Validation Loss fails to improve by a delta of 0.001 for 5 consecutive epochs.
* **Headless Matplotlib Engine:** `matplotlib.use('Agg')` is enforced to prevent GUI event loop crashes during automated training.
* **Windows "Caffeine" Mode:** Utilizes `ctypes` to temporarily override Windows OS power settings, preventing sleep/hibernation during long, unattended CPU training cycles.

---

## Final Performance Metrics

The master pipeline was executed sequentially on a standard laptop CPU. Total execution time for the entire script was **~100 minutes**.

| Metric | Custom CNN (From Scratch) | MobileNetV2 (Transfer Learning) |
| --- | --- | --- |
| **Total Epochs Run** | 25 Epochs (Early Stopped) | 20 Epochs (Completed) |
| **Total Training Time (CPU)** | ~22 minutes | ~76 minutes |
| **Final Test Accuracy** | **76%** | **87%** |
| **Macro Average F1-Score** | 0.76 | 0.87 |
| **Best Class Performance** | Automobile (90%) | Automobile (93%) |
| **Weakest Class Performance** | Cat (61%) | Cat (77%) |

### Key Takeaways

1. **The Power of the 128x128 Optimization:** By shrinking the pipeline from 224x224 down to 128x128, the Custom CNN was able to execute epochs in just ~53 seconds on a CPU, establishing a highly accurate 76% baseline in just 22 minutes.
2. **Transfer Learning Dominance:** MobileNetV2 achieved a staggering **82.5% validation accuracy in its very first epoch**. By skipping the foundational learning phase entirely, it peaked at an impressive 87% accuracy, fully justifying its heavier ~76-minute computational footprint.
3. **The Persistent CIFAR-10 Challenge:** Despite wildly different architectures, both models struggled with the exact same classes: Cats and Dogs. This highlights a fundamental limitation of the dataset, distinguishing between small furry mammals at low resolutions is a universally difficult computer vision task, whereas rigid geometric objects (Automobiles, Trucks, Ships) consistently score 90%+.

---

## How to Run

1. Clone this repository.
2. Ensure you have the required packages installed.
3. Run the master pipeline from your terminal:

```bash
python train.py
```

4. The script will automatically download the data, train both models sequentially, and output all `.png` visual artifacts directly to your root folder.
