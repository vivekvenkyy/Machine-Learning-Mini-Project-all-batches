# 📊 HistGradientBoostingRegressor Interactive Demo
Live Link: https://regressncompare.streamlit.app/

A comprehensive Streamlit web application that demonstrates how **HistGradientBoostingRegressor** works and compares it with other popular regression models.

## 🌟 Features

### 📚 Educational Content
- **Detailed Model Explanations**: Learn how HistGradientBoostingRegressor works with histogram binning and gradient boosting
- **Algorithm Architecture**: Visual breakdown of the boosting process
- **Hyperparameter Guide**: Understand key parameters and tuning strategies
- **Model Comparisons**: Side-by-side comparison with LinearRegression, RandomForest, and XGBoost

### 📊 Dataset Support
- **Built-in Datasets**: 
  - California Housing (regression)
  - Diabetes (regression)
- **CSV Upload**: Upload your own datasets with automatic preprocessing
- **Optional**: Hugging Face datasets integration (install `datasets` package)

### 🔧 Automatic Preprocessing
- **Smart Feature Detection**: Automatically identifies numerical and categorical columns
- **Missing Value Handling**: Imputation with median (numerical) and mode (categorical)
- **Feature Encoding**: One-hot encoding for categorical features
- **Feature Scaling**: StandardScaler for numerical features
- **Pipeline Architecture**: Complete preprocessing + model pipelines

### 🎯 Model Training & Evaluation
- **Multiple Models**: Train and compare up to 4 regression models simultaneously
- **Comprehensive Metrics**: R², MSE, MAE, RMSE for both training and test sets
- **Train-Test Split**: Configurable test size via sidebar slider
- **Overfitting Detection**: Automatic calculation of train-test performance gaps

### 📈 Rich Visualizations
- **Feature Importance**: Bar charts showing top influential features
- **Predicted vs Actual**: Scatter plots with perfect prediction line
- **Residual Plots**: Analyze prediction errors and model bias
- **Partial Dependence Plots**: Understand feature effects on predictions
- **Model Comparison Charts**: Side-by-side performance metrics

### 💾 Export Capabilities
- **Download Metrics**: Export comparison results as CSV
- **Save Models**: Download trained models as .pkl files for deployment
- **Timestamped Files**: Automatic naming with date/time stamps

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)

## 📖 How to Use

### 1. Select Dataset
- Choose from built-in datasets or upload your own CSV file
- For CSV uploads, select the target column you want to predict

### 2. Configure Settings
- Adjust test set size (10-40%)
- Select which models to train and compare
- All settings available in the left sidebar

### 3. Explore Tabs

#### Tab 1: Model Explanation & Visualization
- Learn about HistGradientBoostingRegressor
- Understand algorithm architecture
- Compare with other models
- Review hyperparameter guidelines

#### Tab 2: Training & Evaluation
- View dataset preview and statistics
- Click "Train All Models" button
- Select a model for detailed analysis
- Explore feature importance, predictions, and residuals

#### Tab 3: Model Comparison & Insights
- Compare all models side-by-side
- View performance metrics
- Get automatic recommendations
- Download results and trained models

## 📦 Project Structure

```
ML_mini/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computations

### Machine Learning
- **scikit-learn**: ML models and preprocessing
- **xgboost**: XGBoost regressor (optional but recommended)

### Visualization
- **matplotlib**: Charts and graphs
- **seaborn**: Statistical visualizations

### Optional
- **datasets**: Hugging Face datasets (uncomment in requirements.txt)

## 🎓 Key Concepts Explained

### HistGradientBoostingRegressor
- **Histogram Binning**: Groups continuous features into discrete bins (typically 255) for faster training
- **Gradient Boosting**: Sequentially builds decision trees, each correcting errors from previous trees
- **Native Missing Value Support**: Handles NaN values without imputation
- **Categorical Feature Support**: Can process categorical variables directly

### Evaluation Metrics
- **R² Score**: Proportion of variance explained (closer to 1 is better)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

### Overfitting Detection
The app calculates the "Overfit Gap" = R²(train) - R²(test)
- **< 0.05**: Well-generalized model ✅
- **0.05 - 0.10**: Moderate overfitting ⚠️
- **> 0.10**: Significant overfitting ❌

## 🎯 Example Use Cases

1. **Educational**: Learn how gradient boosting works with interactive visualizations
2. **Model Selection**: Compare multiple models to find the best for your dataset
3. **Data Analysis**: Upload CSV data and get instant model performance insights
4. **Prototyping**: Quickly test if regression models work for your problem
5. **Feature Analysis**: Identify which features are most important for predictions

## 🔬 Advanced Features

### Custom Dataset Upload
- CSV files with any number of features
- Automatic handling of mixed data types
- Missing value imputation
- Categorical encoding

### Model Interpretability
- Feature importance rankings
- Partial dependence plots for top features
- Residual analysis for bias detection
- Predicted vs actual scatter plots

### Hyperparameter Information
The app uses optimized default hyperparameters:
- **HistGradientBoosting**: max_iter=100, learning_rate=0.1, max_depth=5
- **RandomForest**: n_estimators=100, max_depth=10
- **XGBoost**: n_estimators=100, learning_rate=0.1, max_depth=5

## 🐛 Troubleshooting

### XGBoost Not Available
If you see a warning about XGBoost:
```bash
pip install xgboost
```

### Hugging Face Datasets
To enable Hugging Face dataset support:
```bash
pip install datasets
```

### Memory Issues with Large Datasets
- Reduce the number of models being trained
- Use a smaller test set size
- Sample your dataset before uploading

### Slow Training
- Tree-based models can be slow on very large datasets
- Try reducing the number of estimators/iterations
- Use HistGradientBoosting for best speed on large data

## 📊 Sample Datasets Included

### California Housing
- **Samples**: 20,640
- **Features**: 8 numerical features
- **Target**: Median house value
- **Use Case**: Real estate price prediction

### Diabetes
- **Samples**: 442
- **Features**: 10 numerical features
- **Target**: Disease progression
- **Use Case**: Medical data regression

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest new features
- Add support for more datasets
- Improve visualizations

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML models from [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/)
- XGBoost from [DMLC](https://github.com/dmlc/xgboost)

## 📧 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the in-app explanations and tooltips
3. Ensure all dependencies are correctly installed

---

**Enjoy exploring regression models! 🚀📊**
