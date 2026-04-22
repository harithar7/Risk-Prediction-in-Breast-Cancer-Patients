import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


#EDA
# Load dataset
df = pd.read_csv("Breast_Cancer.csv")


print("Columns in dataset:", df.columns.tolist())

# If 'Regional Node Positive' does NOT exist, use 'Regional Node Examined'
if 'Regional Node Positive' not in df.columns:
    print("Using 'Regional Node Examined' instead of 'Regional Node Positive'")
    df['Regional Node Positive'] = df['Regional Node Examined']


print("Original Columns:", df.columns.tolist())

# Remove extra spaces
df.columns = df.columns.str.strip()

# Auto-fix common typo: Reginol → Regional
df.columns = df.columns.str.replace('Regional', 'Regional')

# Also fix if lowercase/other variation exists
df.columns = df.columns.str.replace('regional', 'Regional')

print("Fixed Columns:", df.columns.tolist())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# 1. Dataset structure
print("Shape of dataset:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# 2. Summary statistics
print("\nSummary Statistics (Numerical):\n", df.describe())
print("\nSummary Statistics (Categorical):\n", df.describe(include=['object', 'string']))

# 3. Missing values & duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# 4. Distribution & skewness of key clinical features
print(df.columns.tolist())
key_features = ['Age', 'Tumor Size', 'Regional Node Positive', 'Survival Months']
print("\nSkewness of Key Features:")
for col in key_features:
    print(f"{col}: {df[col].skew():.3f}")


#Comparative Analysis of Clinical Features

key_features = ['Age', 'Tumor Size', 'Regional Node Positive']
for col in key_features:
    print(f"\n{col} Statistics by Status:")
    print(df.groupby('Status')[col].describe())


#Correlation 

# Pearson correlation matrix
num_df = df.select_dtypes(include=np.number)
corr = num_df.corr(method='pearson')
print("Correlation Matrix:\n", corr)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Heatmap")
plt.xlabel("Clinical Features")
plt.ylabel("Clinical Features")
plt.show()






#1.To detect and interpret anomalous patterns in clinical data and evaluate
#   their effect on statistical reliability and survival analysis.
# Outlier detection using IQR
print("\n=== Objective 1: Regression Analysis ===")
print("\nOutlier Count (IQR method):")
for col in key_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")


 

#Visualization

#2.Using appropriate visualization techniques,
#   analyze how key clinical features vary across patient survival status.
print("\n=== Objective 2: Visualization ===")
sns.set_theme(style="whitegrid")


# Plot 1: Histogram - Age distribution by Status

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Status', kde=True, bins=25, palette='Set1')
plt.title("Age Distribution by Survival Status")
plt.xlabel("Patient Age (years)")
plt.ylabel("Number of Patients")
plt.show()

# Plot 2: Boxplot - Tumor Size by Status
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Status', y='Tumor Size', hue='Status', palette='Set2', legend=False)
plt.title("Tumor Size by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Tumor Size")
plt.show()

# Plot 3: Bar Chart - Mortality by Cancer Stage
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='6th Stage', hue='Status', palette='coolwarm')
plt.title("Cancer Stage vs Survival Status")
plt.xticks(rotation=30)
plt.xlabel("Cancer Stage")
plt.ylabel("Number of Patients")
plt.show()


# Pie Chart 4: Survival Status Distribution
plt.figure(figsize=(6,6))
status_counts = df['Status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',colors=colors, startangle=90, explode=(0.02, 0.02),wedgeprops={'edgecolor':'white', 'linewidth':2})
plt.title("Overall Survival Status Distribution", fontsize=13, fontweight='bold')
plt.axis('equal')
plt.show()



# Plot 5: Stacked Bar - Estrogen Status vs Survival
ct = pd.crosstab(df['Estrogen Status'], df['Status'], normalize='index') * 100
ct = ct[['Alive', 'Dead']]
ct.plot(kind='bar', stacked=True, color=['#2ecc71','#e74c3c'])
plt.title("Estrogen Status vs Survival (%)")
plt.ylabel("Percentage")
plt.show()


# Pair plot 6: - relationships between key numeric features colored by survival
pair_features = ['Age', 'Tumor Size', 'Regional Node Positive', 'Survival Months', 'Status']
sns.pairplot(df[pair_features], hue='Status', palette='Set1', diag_kind='kde', height=2)
plt.suptitle("Pair Plot of Key Clinical Features by Survival Status", y=1.02)
plt.show()





#Statistical Hypothesis Testing
#3.Formulate and evaluate appropriate statistical hypotheses to investigate differences in tumor characteristics between survival groups and the relationship between clinical variables and survival status.
#Clearly state the hypotheses, apply suitable statistical tests, and interpret the results.
# ---- Independent t-test (Tumor Size: Alive vs Dead) ----
print("\n=== Objective 3: Hypothesis Testing ===")
alive = df[df['Status']=='Alive']['Tumor Size']
dead  = df[df['Status']=='Dead']['Tumor Size']

t_stat, p_val = stats.ttest_ind(alive, dead, equal_var=False)
print("=== Independent t-test: Tumor Size vs Status ===")
print("H0: Mean Tumor Size is equal for Alive and Dead patients")
print("H1: Mean Tumor Size differs between Alive and Dead")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.5f}")
alpha=0.05
if p_val<alpha:
    print("\nResult: Reject the null Hypothesis")
    print("There IS a statistically significant difference in Tumor Size between Alive and Dead patients.")
else:
    print("\nResult: Fail to Reject the null Hypothesis")
    print("There is NO statistically significant difference in Tumor Size between Alive and Dead patients.")




#Simple Linear Regression
#4.Develop a simple linear regression model to analyze the relationship between a selected clinical feature and patient survival duration.
print("\n=== Objective 4: Regression Analysis ===")
X = df[['Regional Node Positive']]   
y = df['Survival Months']           

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

m = model.coef_[0]
c = model.intercept_
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Regression Equation: Survival Months = {m:.3f} * (Regional Node Positive) + {c:.3f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.3f}")

# Regression line plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=X['Regional Node Positive'], y=y, alpha=0.4)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title("Regression: Regional Node Positive vs Survival Months")
plt.xlabel("Regional Node Positive")
plt.ylabel("Survival Months")
plt.show()





#Case Study / Scenario-Based Analysis


#5.Define high-risk profile based on aggressiveness indicators
print("\n=== Objective 5: Risk Stratification ===")
high_risk = df[(df['Tumor Size'] > 35) &
    (df['Regional Node Positive'] >= 4) &
    (df['Estrogen Status'] == 'Negative')]

low_risk = df[(df['Tumor Size'] <= 35) &
    (df['Regional Node Positive'] < 4) &
    (df['Estrogen Status'] == 'Positive')]

print("High-Risk Patients:", len(high_risk))
print("High-Risk Mortality Rate: {:.2f}%".format((high_risk['Status']=='Dead').mean()*100))
print("Avg Survival Months (High-Risk):", round(high_risk['Survival Months'].mean(),2))

print("\nLow-Risk Patients:", len(low_risk))
print("Low-Risk Mortality Rate: {:.2f}%".format((low_risk['Status']=='Dead').mean()*100))
print("Avg Survival Months (Low-Risk):", round(low_risk['Survival Months'].mean(),2))

# Stage-wise mortality
print("\nMortality % by Cancer Stage:")
print(df.groupby('6th Stage')['Status'].apply(lambda x:(x=='Dead').mean()*100).round(2))

 
