# Here we visualize our data using matplotlib 

import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('C:\\Users\\pc\\OneDrive\\Desktop\\ai\\college\\Sem.2\\Machine Learning\\PRj.1\\BreastCancer_DS.csv')

# Display the first few rows of the dataset
print(data.head())

#visualize the data
plt.figure(figsize=(6, 6))
plt.scatter(data['smoothness_mean'], data['compactness_mean'], alpha=0.8, s=10, marker='x')
plt.title('Scatter plot of smoothness mean and compactness mean')
plt.xlabel('smoothness Mean')
plt.ylabel('compactness Mean')
plt.colorbar(label='Diagnosis (0: Benign, 1: Malignant)')
plt.show()


