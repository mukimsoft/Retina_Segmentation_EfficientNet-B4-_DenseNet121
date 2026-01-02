# Diabetic Retinopathy Segmentation

Colab-friendly PyTorch notebook that trains EfficientNet-B4 U-Net and DenseNet121 U-Net models on the merged IDRiD Segmentation + Localization dataset. The workflow pulls data directly from Kaggle, performs on-the-fly augmentations, trains both models, and logs Dice plus pixel-accuracy metrics along with visual artifacts.

## Project Structure

- retina_segmentation_EfficientNet_B4__DenseNet121.ipynb &mdash; main notebook containing data download, preprocessing, training loop, evaluation utilities, and visualization helpers.
- Training.png, DenseNet121 Prediction.png, Best Accuracy.png &mdash; exported screenshots highlighting logs, sample predictions, and comparison metrics.

## Dataset

- Source: [IDRiD Segmentation + Localization](https://www.kaggle.com/datasets/dankok/diabetic-retinopathy-image-dataset)
- Access: Uses KaggleHub/opendatasets for direct download (requires Kaggle API credentials).
- Merge Strategy: Combines segmentation and localization images/masks; discovers mask channel names automatically.

## Data Pipeline

1. Resize inputs to 384x384 RGB and stack all detected mask channels.
2. Albumentations augmentations for the training split:
   - Horizontal flip (p=0.5)
   - Vertical flip (p=0.2)
   - ShiftScaleRotate (±5% shift, ±10% scale, ±15° rotate)
   - RandomBrightnessContrast (p=0.3)
3. Normalize with A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0).
4. Custom IDRiDSegmentationDataset returns tensors ready for PyTorch DataLoader (num_workers=0 for Colab stability).

## Model & Training Details

| Setting | Value |
| --- | --- |
| Architectures | EfficientNet-B4 U-Net, DenseNet121 U-Net (segmentation_models_pytorch) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Epochs | 5 |
| Batch Size | 4 (tunable depending on GPU RAM) |
| Loss | BCEWithLogitsLoss |
| Metrics | Dice coefficient, pixel accuracy |
| Extras | Mixed precision (if available), cosine LR scheduler hook ready |

During training the notebook logs per-epoch train/val loss, Dice, accuracy, and keeps the best checkpoints for each model.

## Results

| Model | Best Accuracy (%) | Best Dice (%) |
| --- | --- | --- |
| EfficientNet-B4 U-Net | 99.9 | 0.21 |
| DenseNet121 U-Net | 99.9 | 0.31 |

Visual inspection of prediction overlays (see DenseNet121 Prediction.png) shows crisp lesion masks despite class imbalance.

## Usage

1. Open the notebook in Google Colab (GPU runtime recommended).
2. Run the setup cell to install dependencies and authenticate with Kaggle.
3. Execute the data prep, training, and evaluation cells sequentially.
4. Use the comparison table/bar chart plus the prediction viewer cells to explore results.

## Roadmap

- Add EfficientNet-B3 and U-Net++ variants for broader benchmarking.
- Experiment with longer training schedules and focal loss for better Dice.
- Export ONNX/torchscript checkpoints for deployment.

## License

Released under the MIT License (see LICENSE).
