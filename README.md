# 🧪 Glass Type Prediction – End-to-End Machine Learning Pipeline
This project focuses on building an end-to-end machine learning pipeline to classify different types of glass based on their chemical composition and refractive index. The goal is to accurately predict the glass type using supervised learning techniques, while handling class imbalance and dimensionality reduction using PCA.

## 📌 Problem Statement
Glass classification is a classic multi-class classification problem where each sample represents a glass specimen with measured chemical properties. The task is to predict the **type of glass** (e.g., building windows, containers, tableware, etc.) based on features such as:
- Refractive Index (RI)
- Sodium (Na)
- Magnesium (Mg)
- Aluminum (Al)
- Silicon (Si)
- Potassium (K)
- Calcium (Ca)
- Barium (Ba)
- Iron (Fe)

This problem is challenging due to:
- Multi-class nature  
- Imbalanced class distribution  
- Non-linear relationships between features  

## 📂 Dataset
- Source: Kaggle (Glass Identification Dataset – UCI Repository)
- Number of samples: ~200+
- Number of features: 9 numeric features + 1 target class
- Target variable: `Type` (Glass category)

## ⚙️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn (for EDA & visualization)  
- Jupyter Notebook  

## 🔍 Project Workflow
1. **Data Loading & Cleaning**
   - Checked for missing values
   - Verified data types
   - Handled duplicates (if any)
2. **Exploratory Data Analysis (EDA)**
   - Distribution of features
   - Class imbalance analysis
   - Correlation analysis
3. **Feature Scaling**
   - Standardized numeric features using `StandardScaler`
4. **Dimensionality Reduction**
   - Applied **PCA (Principal Component Analysis)** to reduce dimensionality while preserving variance
5. **Model Building**
   - Trained supervised classification models
   - Applied class balancing techniques to improve minority class prediction
6. **Model Evaluation**
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - Macro and Weighted Averages

## 📊 Results

- **Final Accuracy:** ~97–98%  
- **Macro F1-Score:** ~0.97  
- **Weighted F1-Score:** ~0.98  

The final model achieved strong and balanced performance across all glass types, including minority classes that were initially misclassified in the baseline model.

## 🧠 Key Learnings
- Accuracy alone can be misleading in imbalanced multi-class problems  
- Macro F1-score provides a fairer evaluation across all classes  
- PCA helps reduce dimensionality but should be combined with proper feature scaling  
- Tree-based and non-linear models can outperform linear models for complex feature interactions  
- Iterative model improvement is critical in real-world ML workflows  

## 🚀 Future Improvements
- Try advanced models like XGBoost or LightGBM  
- Perform hyperparameter tuning using GridSearchCV  
- Use cross-validation for more robust evaluation  
- Deploy the model using Streamlit or FastAPI  
- Convert into a real-time prediction web app  

## 👨‍💻 Author

**Nishant A. Bilagi**  
Aspiring AI/ML Engineer | Data Science Enthusiast  
GitHub: https://github.com/nishantab95  
LinkedIn: www.linkedin.com/in/nishant-bilagi-2833851bb
