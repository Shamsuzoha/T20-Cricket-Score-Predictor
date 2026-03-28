# 2. Dataset Description
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')

# Basic dataset information
num_rows, num_cols = df.shape
print(f"Total data points: {num_rows}")
print(f"Total features (including target and ID): {num_cols}")

# Peek at the first few entries
print("\nSample data:")
print(df.head(3))

# Identify feature types
features = list(df.columns)
features.remove('Predicted Score')  # exclude target
print(f"\nFeature columns: {features}")
# Count categorical vs quantitative features
categorical_features = [col for col in features if df[col].dtype == 'object']
numeric_features = [col for col in features if df[col].dtype != 'object' and col != 'Match ID']
print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")

# Target variable
target = 'Predicted Score'
print(f"\nTarget variable: '{target}', Data type: {df[target].dtype}")
if pd.api.types.is_numeric_dtype(df[target]):
    print("-> This is a regression problem (target is continuous numeric).")

# Number of data points
print(f"\nNumber of data points: {len(df)}")

# Check if categorical features need encoding
print(f"\nNeed encoding for categorical features? {'Yes' if categorical_features else 'No'} (We have categorical data that machine learning models cannot directly use.)")

# Calculate correlation matrix for numeric features (including target)
numeric_cols = ['Overs Played', 'Wickets Lost', 'Run Rate', 'Opponent Strength', 'Predicted Score']
corr_matrix = df[numeric_cols].corr()

# Save correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('corr_heatmap.png'); plt.close()

# Analyze target distribution: how many instances for various score ranges
plt.figure(figsize=(6,4))
plt.hist(df[target], bins=15, color='orange', edgecolor='black')
plt.title('Distribution of Predicted Score')
plt.xlabel('Predicted Score (runs)')
plt.ylabel('Number of matches')
plt.tight_layout()
plt.savefig('score_distribution.png'); plt.close()

# Exploratory data analysis: relationship between categorical features and target
# Example: Average score by pitch condition
avg_score_by_pitch = df.groupby('Pitch Condition')[target].mean()
print(f"\nAverage score by pitch type:\n{avg_score_by_pitch}")
# Bar chart for average score by pitch condition
order = ['Bowling', 'Balanced', 'Batting']  # logical ordering
avg_score_by_pitch = avg_score_by_pitch.reindex(order)
plt.figure(figsize=(5,4))
sns.barplot(x=avg_score_by_pitch.index, y=avg_score_by_pitch.values, palette="viridis")
plt.title('Average Predicted Score by Pitch Condition')
plt.xlabel('Pitch Condition')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('avg_score_pitch.png'); plt.close()

# Additional EDA: Home vs Away and Weather effects (textual insight)
avg_home = df[df['Home/Away']=='Home'][target].mean()
avg_away = df[df['Home/Away']=='Away'][target].mean()
avg_weather = df.groupby('Weather')[target].mean()
print(f"\nAverage score Home: {avg_home:.1f}, Away: {avg_away:.1f}")
print(f"Average score by Weather:\n{avg_weather}")
