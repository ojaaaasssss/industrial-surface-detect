# Evaluation Day — 2-minute presentation

Good morning. My project is the **Industrial Surface Defect Inspection Platform**, a deep-learning system that automatically classifies product surface images as defective or non-defective.

In manufacturing, surface scratches and cracks can reduce product quality. Manual inspection is time-consuming and can be inconsistent. My solution uses a convolutional neural network, or CNN, to extract patterns directly from labeled surface images.

The application is built with Python, PyTorch and Streamlit. Images are resized to 128 by 128 pixels and normalized. The CNN uses four convolution blocks followed by global average pooling and a binary classification output. I train it using binary cross-entropy loss and the Adam optimizer. The dataset is divided into training, validation and testing partitions.

In the demonstration, I upload a surface image. The model returns whether the product is defective, together with its confidence and estimated defect probability. I also evaluate predictions using accuracy, precision, recall, F1-score and a confusion matrix.

**Important qualification:** The bundled example dataset is synthetically generated for a functional demonstration. Real manufacturing accuracy has not been established. The next step is to train and test on real, independently collected industrial surface images.

## Likely viva questions

**Why CNN?** It learns local image patterns such as edges, lines and textures through convolution filters.

**What is sigmoid?** It converts a single output logit into a number between zero and one, interpreted here as defect probability.

**Why binary cross-entropy?** It measures how well a binary probability prediction matches the labeled outcome.

**Why train/validation/test?** Training updates weights; validation selects the best checkpoint; test estimates performance on held-out images.

**What is overfitting?** When the model memorizes training patterns but performs poorly on unseen data. Augmentation, dropout and validation help detect/reduce it.

**What is F1?** Harmonic mean of precision and recall; useful when both missed defects and false alarms matter.

**Does it detect the location of a scratch?** No. This version only predicts image-level defect presence. Bounding boxes/segmentation are future enhancements.

**Is it production-ready?** No. It requires real industrial data, leakage-safe validation, calibration, and robustness/latency tests.
