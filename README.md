# Alzheimer's Classifier

This project contains a deep learning model to classify Alzheimer's disease stages from brain MRI images.
access the pre-print of the paper here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5013497.

## Features

- Image preprocessing and augmentation
- CNN model built with TensorFlow/Keras
- Model training and evaluation
- Accuracy and loss visualization

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   Open `Alzheimer's_Classifier.ipynb` in Jupyter or Colab.

3. Or run the script:
   ```bash
   python Alzheimers_Classifier.py
   ```

## Data

The project assumes MRI data is available in a suitable format (e.g. folders per class). Place data in the `data/` directory.

## Output

Model checkpoints and metrics can be saved to the `model/` directory.

---
