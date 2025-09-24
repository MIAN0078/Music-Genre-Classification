
# Music Genre Classification using GTZAN Dataset
This project implements a music genre classification system using the GTZAN dataset, exploring both tabular feature-based MLP models and spectrogram-based CNN approaches.

## 📋 Project Overview

The goal of this project is to classify music audio files into 10 different genres using machine learning and deep learning techniques. The implementation includes feature extraction from audio files and building neural network models for classification.

## 🎵 Dataset

The project uses the **GTZAN Dataset** which contains 1000 audio tracks (30 seconds each) across 10 music genres:
- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

Each genre has 100 audio files in WAV format.

## 🏗️ Project Structure

The notebook implements the following workflow:

1. **Data Acquisition**: Download and extract GTZAN dataset
2. **Feature Extraction**: Extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio files
3. **Data Preprocessing**: Label encoding and feature scaling
4. **Model Building**: Implement MLP (Multi-Layer Perceptron) for tabular data
5. **Model Evaluation**: Performance analysis with classification reports and confusion matrices
6. **Visualization**: Training history plots and model performance metrics

## 🔧 Technical Implementation

### Feature Extraction
- **MFCC Features**: 13 MFCC coefficients with mean and standard deviation (26 features total)
- **Audio Processing**: Using librosa library for audio analysis
- **Sampling Rate**: 22050 Hz

### Model Architecture - MLP
Input Layer (26 features)
↓
Dense Layer (256 neurons, ReLU) + Dropout (0.3)
↓
Dense Layer (128 neurons, ReLU) + Dropout (0.3)
↓
Output Layer (10 neurons, Softmax)



### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32
- **Validation Split**: 20%

## 📊 Results

The MLP model achieved:
- **Overall Accuracy**: 71-75% on test data
- **Best Performing Genres**: Classical, Metal, Pop (75-88% precision)
- **Challenging Genres**: Disco, Reggae, Rock (45-55% precision)

### Classification Report Example:
          precision    recall  f1-score   support

   blues       0.74      0.85      0.79        20

   classical 0.85 0.85 0.85 20
country 0.78 0.70 0.74 20
disco 0.53 0.50 0.51 20
hiphop 0.58 0.75 0.65 20
jazz 0.75 0.75 0.75 20
metal 0.88 0.75 0.81 20
pop 0.83 0.95 0.88 20
reggae 0.64 0.45 0.53 20
rock 0.55 0.55 0.55 20

accuracy                           0.71       200



## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or local Python environment

### Required Libraries
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python tqdm



Dataset Setup
Obtain Kaggle API credentials (kaggle.json)

The notebook automatically downloads and extracts the GTZAN dataset

Dataset path: gtzan_dataset/Data/genres_original/

🚀 Usage
Run in Google Colab:

Upload the notebook to Google Colab

Upload your kaggle.json file when prompted

Execute cells sequentially

### Feature Extraction:

The code extracts MFCC features from all audio files

Creates training and testing splits (80-20 split)

Model Training:

The MLP model trains for 30 epochs

Progress and metrics are displayed during training
### Evaluation:
View classification reports and confusion matrices
Analyze training history plots

📈 Model Performance Analysis
The project includes comprehensive visualization:
Training vs Validation accuracy/loss plots
Confusion matrices for model performance analysis
Genre-wise classification metrics

🔮 Future Enhancements
Potential improvements for the project:
Spectrogram-based CNN models for potentially better performance
Data augmentation techniques for audio files
Ensemble methods combining multiple models
Advanced feature extraction (chroma features, spectral contrast)
Hyperparameter tuning for optimized performance

📚 References
GTZAN Dataset: Kaggle Link
Librosa Audio Processing Library
TensorFlow/Keras Documentation
Scikit-learn Documentation
