# Personality Predictor - AI Classification Project

A complete machine learning project that predicts personality types (Extrovert vs Introvert) based on behavioral patterns and social preferences.

## 📋 Project Overview

This project uses machine learning to analyze behavioral data and predict whether someone is an **Extrovert** or **Introvert** with **91% accuracy**. The model analyzes 7 key behavioral indicators to make its prediction.

### 🎯 Key Features
- **High Accuracy**: 91% prediction accuracy using Random Forest algorithm
- **7 Behavioral Indicators**: Time alone, social events, friends circle, etc.
- **Interactive Web App**: Easy-to-use Streamlit interface
- **Complete ML Pipeline**: From data collection to deployment
- **Detailed Analysis**: Feature importance and confidence scores

## 📁 Project Structure

```
AIClass/
├── personality-predictor.ipynb    # 📓 Complete ML development notebook
├── personality-predictor.py       # 🐍 Streamlit web application
├── main.py                       # 🚀 Alternative entry point
├── requirements.txt              # 📦 Python dependencies
├── pyproject.toml               # ⚙️ UV project configuration
├── README.md                    # 📖 This file
├── models/                      # 🤖 Trained model files
│   ├── personality_model.pkl
│   ├── personality_scaler.pkl
│   ├── personality_label_encoder.pkl
│   └── feature_engineering.py
└── uv.lock                     # 🔒 Dependency lock file
```

## 🚀 Quick Start Guide

### Prerequisites
You need these installed on your computer:
- **Python 3.13+** ([Download here](https://www.python.org/downloads/))
- **UV Package Manager** ([Install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **VS Code** (Optional, for viewing notebooks) ([Download here](https://code.visualstudio.com/))

### Step 1: Clone or Download the Project
```bash
# If you have git installed
git clone <repository-url>
cd AIClass

# Or download the ZIP file and extract it
```

### Step 2: Install Dependencies
This project uses **UV** (a fast Python package manager). Don't worry if you haven't used it before!

```bash
# Install all required packages automatically
uv sync

# Alternative: If you prefer pip
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
# Using UV (recommended)
uv run streamlit run personality-predictor.py

# Using regular Python
python -m streamlit run personality-predictor.py
```

### Step 4: Open in Browser
- The app will automatically open in your browser
- If not, go to: `http://localhost:8501`
- Fill in your behavioral preferences and get your personality prediction!

## 📓 Understanding the Development Process

### The Jupyter Notebook (`personality-predictor.ipynb`)

This file contains the **complete machine learning development process**. Think of it as a digital lab notebook where we:

#### What's a Jupyter Notebook?
- An **interactive document** that combines code, explanations, and visualizations
- Like a Word document, but you can run Python code inside it
- Perfect for data science and machine learning experiments

#### What's Inside the Notebook?
The notebook is organized into 8 major sections:

1. **📊 Problem Definition & Data Collection**
   - Downloaded personality dataset from Kaggle
   - 2,512 records of behavioral data
   - 7 behavioral features + personality labels

2. **🧹 Data Preprocessing & Cleaning**
   - Checked for missing values and duplicates
   - Cleaned and validated data quality
   - Prepared data for machine learning

3. **🔍 Exploratory Data Analysis (EDA)**
   - Created visualizations to understand patterns
   - Compared Extrovert vs Introvert behaviors
   - Discovered key behavioral differences

4. **⚙️ Feature Engineering**
   - Created new features from existing data
   - **Social-Solitary Balance**: Most important feature (54.6% importance)
   - Scaled data for optimal model performance

5. **🤖 Model Development & Evaluation**
   - Tested 4 different algorithms:
     - Random Forest (Winner! 91% accuracy)
     - Gradient Boosting
     - Logistic Regression
     - Support Vector Machine
   - Used cross-validation for reliable results

6. **💾 Model Deployment Preparation**
   - Saved trained model and components
   - Created prediction pipeline
   - Built input validation system

7. **🎯 Prediction System & Testing**
   - Interactive prediction interface
   - Comprehensive testing with examples
   - Real-time personality assessment

8. **📋 Project Summary & Next Steps**
   - Performance metrics and conclusions
   - Future improvement suggestions

### How to View the Notebook

#### Option 1: VS Code (Recommended)
1. Open VS Code
2. Install the "Jupyter" extension
3. Open `personality-predictor.ipynb`
4. You can view all the code and run cells interactively

#### Option 2: Jupyter Lab/Notebook
```bash
# Install Jupyter
uv add jupyter

# Start Jupyter
uv run jupyter lab
# or
uv run jupyter notebook
```

#### Option 3: Online Viewers
- Upload to [Google Colab](https://colab.research.google.com/)
- View on [GitHub](https://github.com) (static view only)

## 🎨 Using the Streamlit App

### What is Streamlit?
**Streamlit** is a Python framework that turns data scripts into interactive web applications. No web development experience needed!

### App Features
1. **📝 Input Form**: Enter your behavioral preferences
2. **🎯 Prediction**: Get your personality type prediction
3. **📊 Confidence Score**: See how confident the model is
4. **📈 Feature Analysis**: Understand what influenced the prediction
5. **💡 Interpretation**: Get insights about your behavioral patterns

### Input Parameters
- **Time Alone**: Hours spent alone daily (0-11)
- **Social Events**: Events attended monthly (0-10)
- **Going Outside**: Times per week (0-7)
- **Friends Circle**: Number of close friends (0-15)
- **Social Posts**: Posts per week (0-10)
- **Stage Fear**: Do you have public speaking anxiety? (Yes/No)
- **Drained After Social**: Do you feel tired after socializing? (Yes/No)

## 🔧 Technical Details

### Technology Stack
- **Python 3.13**: Programming language
- **UV**: Fast dependency management
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Web application framework
- **Jupyter**: Interactive development environment

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 91%
- **Cross-Validation**: 5-fold validation
- **Key Feature**: Social-Solitary Balance (54.6% importance)
- **Training Data**: 2,009 samples
- **Test Data**: 503 samples

### Model Features
The model analyzes these behavioral patterns:
1. **Social Engagement Score**: Combines outward social behaviors
2. **Introversion Tendency**: Measures preference for solitude
3. **Social-Solitary Balance**: The difference between the two above
4. **Individual Behavioral Metrics**: Each input parameter

## 🚨 Troubleshooting

### Common Issues

#### UV Not Found
```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

#### Port Already in Use
```bash
# Use a different port
uv run streamlit run personality-predictor.py --server.port 8502
```

#### Module Not Found Errors
```bash
# Reinstall dependencies
uv sync --reinstall
```

#### Can't Open Notebook in VS Code
1. Install the "Jupyter" extension in VS Code
2. Install the "Python" extension in VS Code
3. Restart VS Code

### Getting Help
- Check the error message carefully
- Make sure you're in the correct directory
- Ensure all dependencies are installed
- Try restarting your terminal/command prompt

## 📚 Learning Resources

### New to Python?
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python Tutorials](https://realpython.com/)

### New to Machine Learning?
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Learn](https://www.kaggle.com/learn)

### New to Jupyter Notebooks?
- [Jupyter Documentation](https://jupyter.readthedocs.io/)
- [DataCamp Jupyter Tutorial](https://www.datacamp.com/tutorial/tutorial-jupyter-notebook)

### New to UV?
- [UV Documentation](https://docs.astral.sh/uv/)
- [UV Getting Started Guide](https://docs.astral.sh/uv/getting-started/)

## 🎉 What's Next?

After running the app, you can:
1. **Explore the notebook** to understand the ML process
2. **Modify the code** to experiment with different algorithms
3. **Add new features** to improve predictions
4. **Deploy to the cloud** using platforms like Heroku or Streamlit Cloud
5. **Extend the model** to predict more personality traits

## 📄 License

This project is for educational purposes. Feel free to use, modify, and learn from it!

---

**Happy Personality Predicting! 🧠✨**

*If you encounter any issues or have questions, don't hesitate to ask for help.*
