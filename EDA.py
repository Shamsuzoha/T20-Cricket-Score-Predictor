import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv")

df.head()
df.shape
df.describe()
df.info()

plt.figure()
sns.histplot(df['final_score'], bins=30, kde=True)
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.title("Distribution of Final Scores")

plt.savefig("EDA_final_score_distribution.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.countplot(x='wickets', data=df)
plt.xlabel("Wickets Fallen")
plt.ylabel("Count")
plt.title("Distribution of Wickets Fallen")

plt.savefig("EDA_wickets_distribution.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.scatterplot(x='overs', y='final_score', data=df)
plt.xlabel("Overs Completed")
plt.ylabel("Final Score")
plt.title("Final Score vs Overs Completed")

plt.savefig("EDA_score_vs_overs.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.scatterplot(x='wickets', y='final_score', data=df)
plt.xlabel("Wickets Fallen")
plt.ylabel("Final Score")
plt.title("Final Score vs Wickets Fallen")

plt.savefig("EDA_score_vs_wickets.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.scatterplot(x='run_rate', y='final_score', data=df)
plt.xlabel("Run Rate")
plt.ylabel("Final Score")
plt.title("Final Score vs Run Rate")

plt.savefig("EDA_score_vs_runrate.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.boxplot(x='wickets', y='final_score', data=df)
plt.xlabel("Wickets Fallen")
plt.ylabel("Final Score")
plt.title("Final Score Distribution by Wickets Fallen")

plt.savefig("EDA_boxplot_score_vs_wickets.png", bbox_inches="tight")
plt.close()

numeric_features = df.select_dtypes(include=[np.number])

sns.pairplot(numeric_features)
plt.savefig("EDA_pairplot.png")
plt.close()
