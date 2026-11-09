# Data Centre Server Hotspot Prediction
Machine learning binary classification pipeline to predict server rack overheating (ASHRAE A2 thermal threshold) from 26,981 sensor observations.
- **Tech Stack:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Key Steps:** Data cleaning, domain-specific thermal/power feature engineering, stratified 80/20 train-test split, and model comparison between **Random Forest** (93% Accuracy, 95% Recall) and **Decision Tree**.
