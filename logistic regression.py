
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import scipy.stats as stats


np.random.seed(42)
data = {
    'Car': [f'Car{i}' for i in range(100)],
    'Model': [f'Model{i}' for i in range(100)],
    'Volume': np.random.rand(100),
    'Weight': np.random.rand(100),
    'CO2': (np.random.rand(100) + 0.5 * np.random.rand(100) > 1).astype(int)
}

df = pd.DataFrame(data)


print(df.head())

# Statistical summary
print(df.describe())





X = df[['Volume', 'Weight']]
y = df['CO2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


columns_to_plot = ["Volume", "Weight"]

for col in X_train.columns:
    if col in columns_to_plot:
        plt.figure(figsize=(14,4))
        
      
        plt.subplot(121)
        sns.histplot(X_train[col])
        plt.title(f"{col} - Histogram")

       
        plt.subplot(122)
        stats.probplot(X_train[col], dist="norm", plot=plt)
        plt.title(f"{col} - Probability Plot")

        plt.show()


model = LogisticRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

