# Breast Cancer Diagnosis Using Machine Learning

A comprehensive machine learning project for diagnosing breast cancer using the Wisconsin Breast Cancer Diagnostic (WDBC) dataset. This project implements and compares multiple classification algorithms with hyperparameter tuning to achieve optimal performance.

## 📊 Project Overview

This project applies various machine learning techniques to predict whether a breast tumor is malignant or benign based on features computed from digitized images of fine needle aspirate (FNA) of breast mass. The project includes data preprocessing, exploratory data analysis, feature engineering with PCA, and model optimization through grid search.

## 🎯 Dataset

**Wisconsin Breast Cancer Diagnostic (WDBC) Dataset**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
- **Features**: 30 real-valued features computed from breast mass cell nuclei images
- **Target**: Binary classification (Malignant or Benign)
- **Instances**: 569 samples

### Features Include:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter² / area - 1.0)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension ("coastline approximation" - 1)

Each feature is computed with mean, standard error, and worst (mean of the three largest values) statistics.

## 🤖 Machine Learning Models

This project implements and compares the following classification algorithms:

1. **Logistic Regression**
   - Regularization: L1 (Lasso) and L2 (Ridge)
   - Solvers: liblinear, saga, lbfgs, newton-cg
   - Hyperparameter tuning for C (regularization strength)

2. **Support Vector Machine (SVM)**
   - Kernels: Linear, RBF (Radial Basis Function), Polynomial
   - Hyperparameter tuning for C and gamma parameters

3. **Decision Tree Classifier**
   - Criterion: Gini impurity and Entropy
   - Hyperparameter tuning for max_depth, min_samples_split, and min_samples_leaf

## 🔧 Methodology

### Data Preprocessing
- **Feature Scaling**: StandardScaler for normalization
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Train-Test Split**: Stratified split to maintain class distribution

### Model Optimization
- **Grid Search Cross-Validation**: 5-fold cross-validation
- **Hyperparameter Tuning**: Systematic search over specified parameter ranges
- **Performance Metrics**: Accuracy, confusion matrix, classification report

## 📈 Results

The project evaluates each model using:
- **Accuracy Score**: Overall classification accuracy
- **Confusion Matrix**: True positives, true negatives, false positives, false negatives
- **Classification Report**: Precision, recall, F1-score for each class
- **Cross-Validation**: 5-fold CV to assess model generalization

## 🚀 Usage

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib seaborn numpy jupyter
```

### Running the Analysis
1. Clone the repository:
```bash
git clone https://github.com/jdwright963/ml_breast_cancer_diagnosis.git
cd ml_breast_cancer_diagnosis
```

2. Open the Jupyter notebook:
```bash
jupyter notebook "Ml_Breast_Cancer_Diagnosis_Report .ipynb"
```

3. Run all cells to:
   - Load and preprocess the dataset
   - Perform exploratory data analysis
   - Train and optimize multiple ML models
   - Compare model performance
   - Visualize results

## 📁 Project Structure

```
ml_breast_cancer_diagnosis/
├── README.md                                    # Project documentation
├── Ml_Breast_Cancer_Diagnosis_Report .ipynb    # Main Jupyter notebook
└── Ml_Breast_Cancer_Diagnosis_Report .pdf      # PDF report of analysis
```

## 👥 Authors

- John Wright
- Ka Ho Chan

## 📝 Key Findings

The project demonstrates:
- Effective use of dimensionality reduction (PCA) for high-dimensional medical data
- Comparison of different classification algorithms for cancer diagnosis
- Importance of hyperparameter tuning in model performance
- Application of cross-validation to ensure model robustness

## 🔬 Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib & seaborn**: Data visualization
- **NumPy**: Numerical computing
- **Jupyter Notebook**: Interactive development environment

## 📊 Visualizations

The project includes various visualizations:
- Feature distributions
- Correlation matrices
- PCA component analysis
- Confusion matrices
- Model performance comparisons

## 🎓 Educational Purpose

This project serves as an educational resource for:
- Understanding machine learning workflows
- Applying classification algorithms to medical data
- Learning hyperparameter optimization techniques
- Interpreting model performance metrics
- Working with real-world healthcare datasets

## 📄 License

This project uses publicly available data from the UCI Machine Learning Repository.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the WDBC dataset
- scikit-learn developers for the comprehensive ML library
- The open-source community for tools and resources