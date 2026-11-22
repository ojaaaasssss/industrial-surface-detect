# Industrial Surface Defect Inspection Platform

A Python + PyTorch **binary image classification** prototype. Upload an image of a manufactured surface and receive a **defective / non-defective** prediction, confidence and defect probability. Includes CNN training, a Streamlit GUI, CLI inference, held-out test metrics, and synthetic sample data generation.

## 1. Setup (Python 3.10–3.12 recommended)

Open a terminal in this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Quick demonstration

```bash
streamlit run app.py
```

The ZIP already includes `models/defect_cnn.pt` and `outputs/metrics.json`, trained on bundled synthetic images. To retrain from scratch, run `python train.py --data data/demo --epochs 25`. Training saves the checkpoint and metrics. Open the local URL Streamlit prints (usually http://localhost:8501). Upload any image from `data/demo/defective` or `data/demo/non_defective` to try the UI. **For a fair demonstration of generalization, use images not included in the training split; sample folder images may have been used during training.**

CLI prediction:

```bash
python predict.py --image data/demo/defective/defective_000.png
```

## 3. Use real industrial images

Replace `data/demo` with your dataset folder, e.g.:

```
data/real/
  defective/
    defect_001.jpg
    ...
  non_defective/
    good_001.jpg
    ...
```

Train with `python train.py --data data/real --epochs 20`. The program uses a 70/15/15 stratified random train/validation/test split, saves the checkpoint with lowest validation loss, and evaluates the held-out test partition. Each class needs at least five images; substantially more is recommended. **For production studies, split by part, batch, camera or production run before training to avoid leakage.** The provided random split is for classroom demonstrations only.

## 4. Architecture

Input RGB image → resize 128×128 → normalize → four Conv2D + BatchNorm + ReLU + MaxPool blocks (16, 32, 64, 128 channels) → global average pooling → dropout → linear layer → sigmoid defect probability. BCEWithLogitsLoss and Adam optimize the model. Augmentation is used only for training. Threshold 0.5 by default.

## 5. Evaluation and limitations

`outputs/metrics.json` reports test accuracy, defective-class F1, precision/recall, confusion matrix and training history. The UI shows confidence and defect probability. **Synthetic demonstration images are not evidence of real-world performance.** A pre-trained **synthetic-demo** checkpoint and generated example images are bundled, so the UI works immediately after installing dependencies. No real industrial dataset or industrially validated weights are bundled. This is image-level classification only: it does **not** draw bounding boxes, perform segmentation, or support real-time production-line integration. It is not a safety-critical quality-control system.

## 6. Evaluation talking points

**Problem:** Manual quality inspection can be inconsistent and slow.

**Solution:** CNN learns image features and predicts whether the surface appears defective.

**Training:** Labeled images, 70/15/15 split, training-only augmentation, binary cross-entropy loss, Adam optimizer and validation checkpoint selection.

**Output:** Defect probability, predicted class, confidence; held-out test accuracy, F1 and confusion matrix.

**Future scope:** Real industrial datasets (e.g., NEU surface defects or MVTec AD subject to licenses), batch-aware splitting, defect localization with YOLO/segmentation, calibrated thresholds, FPS/latency tests and camera integration.

## 7. Automated tests

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

- `ModuleNotFoundError`: install requirements in the active environment.
- `No trained model`: run the two quick-demonstration training commands.
- `torch` install fails: see PyTorch's official installation instructions for your OS and Python version.
- CPU training may take several minutes depending on hardware; reduce `--epochs` to 3 for a quick check.
