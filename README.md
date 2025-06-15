# predicting-customer-churn
Leveraging ML models XGBoost, Adaboost, Gradient Boost and Random Forest to predict customer churn

## project objective
Find a classification model that identifies customers at risk of closing their accounts 

## tools and techniques used 

### data manipulation and exploratory data analysis 

Pandas: Used for reading, manipulating, and summarizing data in tabular form.

NumPy: Provides support for large multi-dimensional arrays and matrices, along with mathematical functions.

Matplotlib: A foundational plotting library for creating static, animated, and interactive visualizations.

Seaborn: Built on top of Matplotlib, it provides a high-level interface for drawing attractive statistical graphics.

### feature engineering 

Scikit-learn preprocessing: Includes StandardScaler and MinMaxScaler for feature scaling, and OneHotEncoder for categorical variable encoding.

Imputation: SimpleImputer from scikit-learn is used to handle missing values.

Resampling: SMOTE (oversampling) and RandomUnderSampler (undersampling) from the imbalanced-learn library address class imbalance.

### model evaluation and selection 

Metrics:

Classification metrics: f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, and ConfusionMatrixDisplay from scikit-learn.

Data splitting: train_test_split for splitting data into training and test sets.

Cross-validation: StratifiedKFold and cross_val_score for robust model evaluation.

### model building and tuning 

Model libraries:

Scikit-learn: Includes DecisionTreeClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, and LogisticRegression.

XGBoost: XGBClassifier for gradient boosting.

Hyperparameter tuning: RandomizedSearchCV for efficient hyperparameter optimization.






