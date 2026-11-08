import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})
sns.set_style("whitegrid")

print("=" * 70)
print("  DATA CENTRE SERVER HOTSPOT PREDICTION SYSTEM")
print("  Machine Learning Classification Pipeline")
print("=" * 70)

print("\n" + "─" * 70)
print("STEP 1: DATA LOADING")
print("─" * 70)

df = pd.read_csv('final_dataset_std.csv', sep=';')

print(f"Dataset shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn groups:")
print(f"  P_ac   (AC Power)           : P_ac-0  to P_ac-7   (8 features)")
print(f"  P_cu   (CU Power)           : P_cu-0  to P_cu-7   (8 features)")
print(f"  T_out (Outdoor Temperature) : T_out-0 to T_out-7  (8 features)")
print(f"  T_MEAS (Measured Rack Temp) : T_MEAS-0 to T_MEAS-7 (8 features)")
print(f"  T_celCC (Cell CC Temp)      : T_celCC-0 to T_celCC-7 (8 features)")
print(f"  TLHC  (Temperature LHC)     : 1 feature")
print(f"  DoW   (Day of Week 0–6)     : 1 feature")
print(f"  WeH   (Weekend/Holiday 0/1) : 1 feature")

print("\n" + "─" * 70)
print("STEP 2: DATA CLEANING & EXPLORATION")
print("─" * 70)

null_counts = df.isnull().sum()
print(f"\nMissing values : {null_counts.sum()} (total)")

dup_count = df.duplicated().sum()
print(f"Duplicate rows : {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"  → Removed {dup_count} duplicates. New shape: {df.shape}")

inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"Infinite values: {inf_count}")

print(f"\nDescriptive statistics (first 5 features):")
print(df.iloc[:, :5].describe().round(4).to_string())

print(f"\n✓ Data is already standardised (z-score normalised)")
print(f"  Mean ≈ 0, Std ≈ 1 for continuous features")

print("\n" + "─" * 70)
print("STEP 3: FEATURE ENGINEERING & LABEL CREATION")
print("─" * 70)

t_out_cols = [c for c in df.columns if c.startswith('T_out')]
df['T_out_mean'] = df[t_out_cols].mean(axis=1)

t_meas_cols = [c for c in df.columns if c.startswith('T_MEAS')]
df['T_MEAS_max'] = df[t_meas_cols].max(axis=1)

df['T_MEAS_spread'] = df[t_meas_cols].max(axis=1) - df[t_meas_cols].min(axis=1)

p_ac_cols = [c for c in df.columns if c.startswith('P_ac')]
df['P_ac_mean'] = df[p_ac_cols].mean(axis=1)

p_cu_cols = [c for c in df.columns if c.startswith('P_cu')]
df['P_cu_mean'] = df[p_cu_cols].mean(axis=1)

print(f"Engineered features added:")
print(f"  T_out_mean     – Average outdoor temperature across 8 sensors")
print(f"  T_MEAS_max     – Maximum rack temperature across 8 racks")
print(f"  T_MEAS_spread  – Temperature spread (max − min) across racks")
print(f"  P_ac_mean      – Average AC power consumption")
print(f"  P_cu_mean      – Average CU power consumption")

HOTSPOT_THRESHOLD = 1.0

df['Hotspot'] = (df['TLHC'] > HOTSPOT_THRESHOLD).astype(int)

hotspot_counts = df['Hotspot'].value_counts()
print(f"\n── Label Creation ──")
print(f"Threshold : TLHC > {HOTSPOT_THRESHOLD} (standardised)")
print(f"  Normal  (0): {hotspot_counts[0]:>6d}  ({hotspot_counts[0]/len(df)*100:.1f}%)")
print(f"  Hotspot (1): {hotspot_counts[1]:>6d}  ({hotspot_counts[1]/len(df)*100:.1f}%)")

print("\n" + "─" * 70)
print("STEP 4: FEATURE SELECTION & TRAIN/TEST SPLIT")
print("─" * 70)

exclude_cols = ['TLHC', 'Hotspot']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['Hotspot']

print(f"Total features used: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain/Test Split (80/20, stratified):")
print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Testing set  : {X_test.shape[0]} samples")
print(f"  Train hotspot ratio: {y_train.mean():.3f}")
print(f"  Test  hotspot ratio: {y_test.mean():.3f}")

print(f"\n✓ No additional scaling required – dataset is pre-standardised")

print("\n" + "─" * 70)
print("STEP 5: MODEL DEVELOPMENT")
print("─" * 70)

print("\n── 5.1 PRIMARY MODEL: Random Forest Classifier ──")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Parameters:")
print(f"  n_estimators     = 100")
print(f"  max_depth         = 15")
print(f"  min_samples_split= 5")
print(f"  min_samples_leaf = 2")
print(f"  random_state     = 42")
print(f"  n_jobs           = -1")

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("✓ Random Forest training complete.")

print("\n── 5.2 COMPARISON MODEL: Decision Tree Classifier ──")

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("Parameters:")
print(f"  max_depth         = 10")
print(f"  min_samples_split= 5")
print(f"  min_samples_leaf = 2")
print(f"  random_state     = 42")

print("\nTraining Decision Tree...")
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("✓ Decision Tree training complete.")

print("\n" + "─" * 70)
print("STEP 6: EVALUATION & PERFORMANCE ANALYSIS")
print("─" * 70)

print("\n══════════════════════════════════════════════════════════")
print("  RANDOM FOREST – EVALUATION RESULTS")
print("══════════════════════════════════════════════════════════")

cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Normal  Hotspot")
print(f"  Actual Normal  {cm_rf[0][0]:>6d}  {cm_rf[0][1]:>6d}")
print(f"  Actual Hotspot {cm_rf[1][0]:>6d}  {cm_rf[1][1]:>6d}")
print(f"\nClassification Metrics:")
print(f"  Accuracy  = {acc_rf:.4f}  ({acc_rf*100:.2f}%)")
print(f"  Precision = {prec_rf:.4f}")
print(f"  Recall    = {rec_rf:.4f}")
print(f"  F1-Score  = {f1_rf:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf,
                            target_names=['Normal', 'Hotspot']))

print("══════════════════════════════════════════════════════════")
print("  DECISION TREE – EVALUATION RESULTS")
print("══════════════════════════════════════════════════════════")

cm_dt = confusion_matrix(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Normal  Hotspot")
print(f"  Actual Normal  {cm_dt[0][0]:>6d}  {cm_dt[0][1]:>6d}")
print(f"  Actual Hotspot {cm_dt[1][0]:>6d}  {cm_dt[1][1]:>6d}")
print(f"\nClassification Metrics:")
print(f"  Accuracy  = {acc_dt:.4f}  ({acc_dt*100:.2f}%)")
print(f"  Precision = {prec_dt:.4f}")
print(f"  Recall    = {rec_dt:.4f}")
print(f"  F1-Score  = {f1_dt:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_dt,
                            target_names=['Normal', 'Hotspot']))

print("══════════════════════════════════════════════════════════")
print("  MODEL COMPARISON SUMMARY")
print("══════════════════════════════════════════════════════════")
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf],
    'Decision Tree': [acc_dt, prec_dt, rec_dt, f1_dt]
})
comparison_df['Difference'] = comparison_df['Random Forest'] - comparison_df['Decision Tree']
print(comparison_df.to_string(index=False, float_format='%.4f'))
print()

print("\n" + "─" * 70)
print("STEP 7: VISUALISATIONS")
print("─" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Hotspot'],
            yticklabels=['Normal', 'Hotspot'],
            ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title('Random Forest – Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('Actual Label')

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal', 'Hotspot'],
            yticklabels=['Normal', 'Hotspot'],
            ax=axes[1], cbar_kws={'shrink': 0.8})
axes[1].set_title('Decision Tree – Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('Actual Label')

plt.tight_layout()
plt.savefig('viz1_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: viz1_confusion_matrices.png")

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
               color=sns.color_palette('viridis', n_colors=15))
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Random Forest – Top 15 Feature Importances', fontsize=14, fontweight='bold')

for bar, val in zip(bars, feat_imp_df['Importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('viz2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: viz2_feature_importance.png")

fig, ax = plt.subplots(figsize=(9, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]
dt_scores = [acc_dt, prec_dt, rec_dt, f1_dt]

x = np.arange(len(metrics))
width = 0.30

bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest',
               color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, dt_scores, width, label='Decision Tree',
               color='#FF9800', edgecolor='white')

ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Model Comparison – Random Forest vs Decision Tree',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('viz3_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: viz3_model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ['#4CAF50', '#F44336']
axes[0].pie(hotspot_counts.values, labels=['Normal', 'Hotspot'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0, 0.05), textprops={'fontsize': 12})
axes[0].set_title('Hotspot Label Distribution', fontsize=14, fontweight='bold')

axes[1].hist(df['TLHC'], bins=60, color='#42A5F5', edgecolor='white', alpha=0.8)
axes[1].axvline(x=HOTSPOT_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {HOTSPOT_THRESHOLD}')
axes[1].set_xlabel('TLHC (Standardised)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('TLHC Distribution with Hotspot Threshold', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('viz4_label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: viz4_label_distribution.png")

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"""
  Dataset          : {df.shape[0]} samples, {len(feature_cols)} features
  Task             : Binary Classification (Hotspot vs Normal)
  Hotspot Threshold: TLHC > {HOTSPOT_THRESHOLD} (standardised)
  Train/Test Split : 80% / 20% (stratified)
  
  ┌──────────────────────────────────────────────────┐
  │  PRIMARY MODEL  : Random Forest Classifier       │
  │    Accuracy  = {acc_rf:.4f}                              │
  │    Precision = {prec_rf:.4f}                              │
  │    Recall    = {rec_rf:.4f}                              │
  │    F1-Score  = {f1_rf:.4f}                              │
  ├──────────────────────────────────────────────────┤
  │  COMPARISON     : Decision Tree Classifier       │
  │    Accuracy  = {acc_dt:.4f}                              │
  │    Precision = {prec_dt:.4f}                              │
  │    Recall    = {rec_dt:.4f}                              │
  │    F1-Score  = {f1_dt:.4f}                              │
  └──────────────────────────────────────────────────┘
  
  Visualisations saved:
    1. viz1_confusion_matrices.png
    2. viz2_feature_importance.png
    3. viz3_model_comparison.png
    4. viz4_label_distribution.png
""")
print("=" * 70)
print("  Pipeline complete.")
print("=" * 70)