# 🎯 Habit Performance Analyzer

A complete end-to-end **Deep Learning application** built with PyTorch that analyzes habit and productivity data to predict performance scores. Upload your CSV data and get AI-powered insights instantly!

## 🌟 Features

- **PyTorch Deep Learning Model**: Multi-layer perceptron (MLP) with batch normalization and dropout
- **Flexible Architecture**: Automatically adapts to any number of input features
- **Complete Pipeline**: Training, inference, and web interface
- **Production-Ready**: Modular code with proper preprocessing and error handling
- **Interactive Web App**: Beautiful Streamlit interface for easy usage
- **Visualizations**: Interactive charts showing performance distributions
- **Export Results**: Download predictions as CSV

## 📁 Project Structure

```
habit_analyzer/
├── model.py                    # PyTorch model architecture
├── train.py                    # Training pipeline
├── inference.py                # Prediction pipeline
├── app.py                      # Streamlit web interface
├── generate_sample_data.py     # Sample data generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── sample_training_data.csv    # (Generated) Sample dataset
├── model.pth                   # (Generated) Trained model weights
└── scaler.pkl                  # (Generated) Feature scaler
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)

```bash
# Create a sample dataset for testing
python generate_sample_data.py
```

This creates `sample_training_data.csv` with 1000 rows of realistic habit data including:
- Sleep hours
- Exercise minutes
- Water intake
- Meditation time
- Screen time
- Social interactions
- Productivity tasks
- Stress levels
- Meal quality
- Breaks taken

### 3. Train the Model

```bash
# Train on the sample data (or your own CSV)
python train.py sample_training_data.csv
```

**What happens during training:**
- Loads CSV data (last column = target)
- Handles missing values automatically
- Standardizes features using StandardScaler
- Trains for 100 epochs with validation
- Saves `model.pth` and `scaler.pkl`

**Training output:**
```
Loading data from sample_training_data.csv...
Dataset shape: (1000, 11)
Features shape: (1000, 10)
Target shape: (1000,)

Using device: cpu
Model created with 10 input features

Starting training for 100 epochs...
Epoch [1/100] | Train Loss: 245.1234 | Val Loss: 238.5678
Epoch [10/100] | Train Loss: 89.4567 | Val Loss: 91.2345
...
Epoch [100/100] | Train Loss: 12.3456 | Val Loss: 15.6789

Training completed! Best validation loss: 14.2345
```

### 4. Run the Web App

```bash
# Launch Streamlit interface
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 5. Make Predictions

**Option A: Web Interface**
1. Open the Streamlit app
2. Upload your CSV file (without target column)
3. Click "Analyze Performance"
4. View results and download predictions

**Option B: Command Line**
```bash
# Make predictions via CLI
python inference.py new_data.csv output_predictions.csv
```

## 📊 Data Format Requirements

Your CSV file should:
- Have **numerical features only** (or properly encoded categorical features)
- Match the **same number of features** as training data
- **Not include** the target column (performance score)
- Have column headers

### Example Training Data Format:
```csv
feature1,feature2,feature3,...,target_score
10.5,5.2,3.1,...,75.5
8.3,4.7,2.9,...,68.2
```

### Example Inference Data Format:
```csv
feature1,feature2,feature3,...
10.5,5.2,3.1,...
8.3,4.7,2.9,...
```

## 🏗️ Model Architecture

The model uses a flexible Multi-Layer Perceptron (MLP):

```
Input Layer (dynamic size)
    ↓
Linear(input_size, 128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(128, 64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64, 32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(32, 1) → Output (Performance Score)
```

**Key Features:**
- **Adaptive Input**: Works with any number of features
- **Batch Normalization**: Faster training and better generalization
- **Dropout**: Prevents overfitting (30% dropout rate)
- **Xavier Initialization**: Optimal weight initialization
- **MSE Loss**: Suitable for regression tasks
- **Adam Optimizer**: Adaptive learning rate

## 🔧 Configuration

### Model Hyperparameters (in `model.py`):
```python
hidden_sizes = [128, 64, 32]  # Hidden layer dimensions
dropout_rate = 0.3            # Dropout probability
```

### Training Hyperparameters (in `train.py`):
```python
epochs = 100                  # Number of training epochs
learning_rate = 0.001         # Adam optimizer learning rate
batch_size = 32               # Training batch size
test_size = 0.2              # Validation split (20%)
```

## 📈 Using Your Own Data

### Step 1: Prepare Your CSV
Ensure your CSV has:
- Numerical features in all columns except the last
- Last column = target score (for training)
- No missing headers

### Step 2: Train on Your Data
```bash
python train.py your_data.csv
```

### Step 3: Create Test Data
Remove the target column from new samples:
```python
import pandas as pd

df = pd.read_csv('your_new_data.csv')
df_test = df.drop('target_column', axis=1)
df_test.to_csv('test_data.csv', index=False)
```

### Step 4: Get Predictions
```bash
python inference.py test_data.csv results.csv
```

## 🎨 Web App Features

The Streamlit interface includes:
- ✅ **File Upload**: Drag-and-drop CSV upload
- 📊 **Data Preview**: View uploaded data before analysis
- 🔍 **Validation**: Automatic feature count checking
- 📈 **Statistics**: Min, max, mean, std deviation of scores
- 📉 **Visualizations**: Distribution and box plots
- 💾 **Export**: Download results as CSV

## 🛠️ Advanced Usage

### Custom Model Architecture
Edit `model.py` to change the network:
```python
model = HabitPerformanceModel(
    input_size=input_size,
    hidden_sizes=[256, 128, 64, 32],  # Add more layers
    dropout_rate=0.4                   # Increase dropout
)
```

### Batch Predictions in Python
```python
from inference import PerformancePredictor
import pandas as pd

# Load predictor
predictor = PerformancePredictor()

# Load your data
df = pd.read_csv('my_data.csv')

# Get predictions
predictions = predictor.predict(df)

# Add to dataframe
df['Performance_Score'] = predictions
```

### Model Evaluation
```python
import torch
from train import load_and_preprocess_data

# Load test data
train_loader, val_loader, scaler, input_size = load_and_preprocess_data('data.csv')

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
# ... add your evaluation code
```

## 🐛 Troubleshooting

### "Model not found" error
- Run `python train.py <data.csv>` first to train the model

### "Column mismatch" error
- Ensure your test CSV has the same number of features as training data
- Remove the target column from test data

### "Module not found" error
- Install dependencies: `pip install -r requirements.txt`

### Low prediction accuracy
- Train on more data (1000+ samples recommended)
- Increase epochs: modify `epochs=200` in `train.py`
- Check data quality and feature relevance

## 📝 Code Quality

✅ **Production-ready features:**
- Comprehensive error handling
- Input validation
- Detailed logging
- Type hints (where applicable)
- Modular design
- Extensive comments
- Reusable components

## 🤝 Contributing

To extend this project:
1. Add new features in `model.py`
2. Enhance preprocessing in `train.py`
3. Improve visualizations in `app.py`
4. Add evaluation metrics in `inference.py`

## 📄 License

This project is open-source and available for educational and commercial use.

## 🎓 Learning Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Streamlit Docs**: https://docs.streamlit.io/
- **scikit-learn**: https://scikit-learn.org/

## 🚀 Deployment

### Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Visit https://streamlit.io/cloud
3. Connect your repository
4. Deploy!

### Deploy to Heroku:
```bash
# Add Procfile
echo "web: streamlit run app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review code comments
- Experiment with the sample data

---

**Built with ❤️ using PyTorch and Streamlit**

Happy Analyzing! 🎯