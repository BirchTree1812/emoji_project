# 🎨 Emoji Classification Challenge

A deep learning solution for classifying emoji images by their visual style/vendor using Convolutional Neural Networks (CNN).

## 📋 Overview

This project identifies the origin/vendor of emoji images based on their distinctive visual characteristics. The model can distinguish between emojis from:

- 🍎 **Apple**
- 📘 **Facebook**
- 🔍 **Google**
- 💬 **Messenger**
- 📱 **Samsung**
- 💚 **WhatsApp**

## ✨ Features

- **Memory-Optimized Architecture**: Designed to run on systems with limited RAM
- **Data Generators**: Loads images on-the-fly instead of storing all in memory
- **Lightweight CNN**: ~200K parameters for efficient training
- **Built-in Data Augmentation**: Rotation, flipping, and other transforms
- **Comprehensive Evaluation**: Confusion matrix, classification report, and visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- At least 2GB free RAM
- ~500MB disk space for dependencies

### Installation

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **For Windows users**: Enable long paths (required for TensorFlow)
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then restart your computer.

### Dataset Structure

Organize your data as follows:
```
emoji/
├── train/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── test/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── train_labels.csv
└── emoji-sandbox-submission.ipynb
```

**train_labels.csv format:**
```csv
Id,Label
1,apple
2,google
...
```

## 📊 Model Architecture

### Lightweight CNN Design

```
Input (72x72x3)
    ↓
Data Augmentation (RandomFlip, RandomRotation)
    ↓
Conv2D(16) → BatchNorm → MaxPool
    ↓
Conv2D(32) → BatchNorm → MaxPool
    ↓
Conv2D(64) → BatchNorm → MaxPool
    ↓
Dense(128) → Dropout(0.5)
    ↓
Dense(64) → Dropout(0.3)
    ↓
Output (6 classes, Softmax)
```

**Total Parameters**: ~200,000 (60% smaller than standard architectures)

## 💾 Memory Optimization

This project is optimized for low-memory systems:

| Feature | Memory Savings |
|---------|----------------|
| Data Generators | ~3GB saved (no full dataset in RAM) |
| Small Batch Size (16) | ~1GB saved |
| Lightweight Model | ~500MB saved |
| Garbage Collection | Additional cleanup |

**Estimated RAM Usage**: 300-500 MB (vs 2-4GB for standard approach)

## 🎯 Usage

### Training

Open `emoji-sandbox-submission.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially.

Key parameters you can adjust:
```python
BATCH_SIZE = 16      # Reduce if running out of memory
EPOCHS = 30          # Increase for better accuracy
target_size = (72, 72)  # Image dimensions
```

### Making Predictions

The notebook generates `submission.csv` with predictions for the test set:
```csv
Id,Label
00001,apple
00002,google
...
```

## 📈 Performance

Expected performance metrics:
- **Training Accuracy**: ~85-95%
- **Validation Accuracy**: ~80-90%
- **Training Time**: ~10-20 minutes (CPU) / ~3-5 minutes (GPU)

## 🛠️ Customization

### Increase Model Capacity
If you have more memory available, you can increase model size:
```python
# In build_cnn_model function
layers.Conv2D(32, ...)  # Change from 16
layers.Conv2D(64, ...)  # Change from 32
layers.Conv2D(128, ...) # Change from 64
```

### Adjust Data Augmentation
```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.2)  # Increase rotation range
layers.RandomZoom(0.1)      # Add zoom augmentation
```

## 📁 Project Structure

```
emoji/
├── emoji-sandbox-submission.ipynb  # Main notebook
├── train/                          # Training images
├── test/                           # Test images
├── train_labels.csv                # Training labels
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
└── submission.csv                  # Generated predictions
```

## 🐛 Troubleshooting

### TensorFlow Import Error
```
ImportError: cannot import name 'pywrap_tensorflow'
```
**Solution**: Reinstall TensorFlow
```bash
pip uninstall tensorflow keras
pip install tensorflow==2.16.2 "numpy>=1.26,<2.0"
```

### Out of Memory
**Solution**: Reduce batch size
```python
BATCH_SIZE = 8  # Or even 4
```

### Long Path Issues (Windows)
**Solution**: Enable long paths and restart
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## 📝 Future Improvements

- [ ] Transfer learning with pre-trained models (MobileNet, EfficientNet)
- [ ] K-fold cross-validation
- [ ] Ensemble methods
- [ ] Advanced augmentation (mixup, cutmix)
- [ ] Hyperparameter optimization (Optuna, KerasTuner)
- [ ] Model export for deployment

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- scikit-learn for preprocessing utilities
- The emoji vendors for their distinctive designs

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Made with ❤️ for emoji classification**
