# Emoji Classification Project

## Overview
This project focuses on **emoji classification** using machine learning. The notebook provides a framework for building and training a deep learning model to classify emojis into predefined categories.

## Goal
The primary objective is to:
- Load and preprocess emoji images (handling RGBA, grayscale, and RGB formats)
- Build a neural network model for emoji classification
- Train the model on labeled emoji data
- Make predictions on test emojis
- Generate submission predictions for evaluation

## Project Structure
- `emoji-sandbox-submission.ipynb` - Main Jupyter notebook containing the complete workflow

## Notebook Sections

### 1. **Imports**
Essential libraries for image processing, data handling, and deep learning:
- NumPy and Pandas for data manipulation
- PIL and scikit-image for image processing
- TensorFlow/Keras for neural network modeling
- Matplotlib for visualization

### 2. **Functions**
Key helper functions:
- `load_single_image(path)` - Preprocesses images by converting them to RGB, normalizing pixel values, and handling different image formats
- `imageLoader(files, labels, batch_size)` - Generator function for loading images in batches for efficient model training

### 3. **Dataset**
- Loads training images and labels from CSV
- Loads test images for prediction
- Creates data structures for batch processing

### 4. **Training**
- Identifies unique emoji labels from training data
- Implements prediction logic (currently using random predictions as a baseline)
- Generates submission CSV with predictions

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare your dataset with the following structure:
   ```
   data/
   ├── train/
   │   ├── emoji_1.png
   │   ├── emoji_2.png
   │   └── ...
   ├── test/
   │   ├── emoji_1.png
   │   ├── emoji_2.png
   │   └── ...
   ├── train_labels.csv
   ```

2. Update the `PATH` variable in the notebook to point to your dataset directory

3. Run the notebook cells sequentially to:
   - Load and preprocess images
   - Create batch generators
   - Train your model
   - Generate predictions
   - Export submission CSV

### Output
The notebook generates `submission.csv` containing emoji IDs and their predicted labels.

## Key Features
- **Flexible Image Preprocessing** - Handles multiple image formats (RGBA, grayscale, RGB)
- **Batch Processing** - Efficient loading of large datasets
- **Modular Design** - Easy to modify preprocessing and model architecture
- **Submission Ready** - Generates output in the required format

## Notes
- The current implementation includes a baseline random prediction model - you should implement your actual model architecture
- Images are normalized to float32 values in the range [0, 1]
- The batch generator yields preprocessed images and their corresponding labels
