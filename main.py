import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
# This CSV file contains features extracted from breast cancer cell images
# Each row is a patient sample, with many numerical features plus the diagnosis label (M = malignant, B = benign)
data = pd.read_csv("breast-cancer.csv")

# --- Data Exploration (Optional, kept commented out) ---
# View the first few rows of the dataset to inspect structure and values
# head = data.head()
# print(head)
"""
         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  smoothness_mean  compactness_mean  ...  area_worst  smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99         10.38          122.80     1001.0          0.11840           0.27760  ...      2019.0            0.1622             0.6656           0.7119                0.2654          0.4601                  0.11890          NaN
1    842517         M        20.57         17.77          132.90     1326.0          0.08474           0.07864  ...      1956.0            0.1238             0.1866           0.2416                0.1860          0.2750                  0.08902          NaN
2  84300903         M        19.69         21.25          130.00     1203.0          0.10960           0.15990  ...      1709.0            0.1444             0.4245           0.4504                0.2430          0.3613                  0.08758          NaN
3  84348301         M        11.42         20.38           77.58      386.1          0.14250           0.28390  ...       567.7            0.2098             0.8663           0.6869                0.2575          0.6638                  0.17300          NaN
4  84358402         M        20.29         14.34          135.10     1297.0          0.10030           0.13280  ...      1575.0            0.1374             0.2050           0.4000                0.1625          0.2364                  0.07678          NaN

[5 rows x 33 columns]
"""

# Check dataset structure: number of rows, columns, data types, and null counts
# data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 33 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   id                       569 non-null    int64
 1   diagnosis                569 non-null    object
 2   radius_mean              569 non-null    float64
 3   texture_mean             569 non-null    float64
 4   perimeter_mean           569 non-null    float64
 5   area_mean                569 non-null    float64
 6   smoothness_mean          569 non-null    float64
 7   compactness_mean         569 non-null    float64
 8   concavity_mean           569 non-null    float64
 9   concave points_mean      569 non-null    float64
 10  symmetry_mean            569 non-null    float64
 11  fractal_dimension_mean   569 non-null    float64
 12  radius_se                569 non-null    float64
 13  texture_se               569 non-null    float64
 14  perimeter_se             569 non-null    float64
 15  area_se                  569 non-null    float64
 16  smoothness_se            569 non-null    float64
 17  compactness_se           569 non-null    float64
 18  concavity_se             569 non-null    float64
 19  concave points_se        569 non-null    float64
 20  symmetry_se              569 non-null    float64
 21  fractal_dimension_se     569 non-null    float64
 22  radius_worst             569 non-null    float64
 23  texture_worst            569 non-null    float64
 24  perimeter_worst          569 non-null    float64
 25  area_worst               569 non-null    float64
 26  smoothness_worst         569 non-null    float64
 27  compactness_worst        569 non-null    float64
 28  concavity_worst          569 non-null    float64
 29  concave points_worst     569 non-null    float64
 30  symmetry_worst           569 non-null    float64
 31  fractal_dimension_worst  569 non-null    float64
 32  Unnamed: 32              0 non-null      float64
dtypes: float64(31), int64(1), object(1)
memory usage: 146.8+ KB
"""

# Compute descriptive statistics (mean, std, min, max, quartiles) for all numeric columns
# describe = data.describe()
# print(describe)
"""
                 id  radius_mean  texture_mean  perimeter_mean    area_mean  smoothness_mean  compactness_mean  concavity_mean  ...   area_worst  smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst  Unnamed: 32
count  5.690000e+02   569.000000    569.000000      569.000000   569.000000       569.000000        569.000000      569.000000  ...   569.000000        569.000000         569.000000       569.000000            569.000000      569.000000               569.000000          0.0
mean   3.037183e+07    14.127292     19.289649       91.969033   654.889104         0.096360          0.104341        0.088799  ...   880.583128          0.132369           0.254265         0.272188              0.114606        0.290076                 0.083946          NaN
std    1.250206e+08     3.524049      4.301036       24.298981   351.914129         0.014064          0.052813        0.079720  ...   569.356993          0.022832           0.157336         0.208624              0.065732        0.061867                 0.018061          NaN
min    8.670000e+03     6.981000      9.710000       43.790000   143.500000         0.052630          0.019380        0.000000  ...   185.200000          0.071170           0.027290         0.000000              0.000000        0.156500                 0.055040          NaN
25%    8.692180e+05    11.700000     16.170000       75.170000   420.300000         0.086370          0.064920        0.029560  ...   515.300000          0.116600           0.147200         0.114500              0.064930        0.250400                 0.071460          NaN
50%    9.060240e+05    13.370000     18.840000       86.240000   551.100000         0.095870          0.092630        0.061540  ...   686.500000          0.131300           0.211900         0.226700              0.099930        0.282200                 0.080040          NaN
75%    8.813129e+06    15.780000     21.800000      104.100000   782.700000         0.105300          0.130400        0.130700  ...  1084.000000          0.146000           0.339100         0.382900              0.161400        0.317900                 0.092080          NaN
max    9.113205e+08    28.110000     39.280000      188.500000  2501.000000         0.163400          0.345400        0.426800  ...  4254.000000          0.222600           1.058000         1.252000              0.291000        0.663800                 0.207500          NaN

[8 rows x 32 columns]
"""

# Visualise missing values: heatmap shows which cells are empty (NaN) in the dataset
# sns.heatmap(data.isnull())
# plt.tight_layout()
# plt.show()

# --- Data Cleaning ---
# Remove unnecessary columns:
# "Unnamed: 32" is entirely empty (all NaN), and "id" is just an identifier, not useful for prediction
data.drop(["Unnamed: 32", "id"], inplace=True, axis=1)

# Re-check dataset after dropping redundant columns
# head = data.head()
# print(head)
"""
  diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  smoothness_mean  compactness_mean  concavity_mean  ...  perimeter_worst  area_worst  smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst
0         M        17.99         10.38          122.80     1001.0          0.11840           0.27760          0.3001  ...           184.60      2019.0            0.1622             0.6656           0.7119                0.2654          0.4601                  0.11890
1         M        20.57         17.77          132.90     1326.0          0.08474           0.07864          0.0869  ...           158.80      1956.0            0.1238             0.1866           0.2416                0.1860          0.2750                  0.08902
2         M        19.69         21.25          130.00     1203.0          0.10960           0.15990          0.1974  ...           152.50      1709.0            0.1444             0.4245           0.4504                0.2430          0.3613                  0.08758
3         M        11.42         20.38           77.58      386.1          0.14250           0.28390          0.2414  ...            98.87       567.7            0.2098             0.8663           0.6869                0.2575          0.6638                  0.17300
4         M        20.29         14.34          135.10     1297.0          0.10030           0.13280          0.1980  ...           152.20      1575.0            0.1374             0.2050           0.4000                0.1625          0.2364                  0.07678

[5 rows x 31 columns]
"""

# Convert diagnosis labels into binary numeric format:
# "M" (malignant) → 1, "B" (benign) → 0
data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]

# Confirm transformation worked
# head = data.head()
# print(head)

# Cast diagnosis column to categorical type for efficiency and clarity
data["diagnosis"] = data["diagnosis"].astype("category", copy=False)

"""
   diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  smoothness_mean  compactness_mean  concavity_mean  ...  perimeter_worst  area_worst  smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst
0          1        17.99         10.38          122.80     1001.0          0.11840           0.27760          0.3001  ...           184.60      2019.0            0.1622             0.6656           0.7119                0.2654          0.4601                  0.11890
1          1        20.57         17.77          132.90     1326.0          0.08474           0.07864          0.0869  ...           158.80      1956.0            0.1238             0.1866           0.2416                0.1860          0.2750                  0.08902
2          1        19.69         21.25          130.00     1203.0          0.10960           0.15990          0.1974  ...           152.50      1709.0            0.1444             0.4245           0.4504                0.2430          0.3613                  0.08758
3          1        11.42         20.38           77.58      386.1          0.14250           0.28390          0.2414  ...            98.87       567.7            0.2098             0.8663           0.6869                0.2575          0.6638                  0.17300
4          1        20.29         14.34          135.10     1297.0          0.10030           0.13280          0.1980  ...           152.20      1575.0            0.1374             0.2050           0.4000                0.1625          0.2364                  0.07678

[5 rows x 31 columns]
"""

# Plot bar chart of class distribution (malignant vs benign)
# data["diagnosis"].value_counts().plot(kind="bar")
# plt.show()

# Check memory usage and data types again
# data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   diagnosis                569 non-null    category
 1   radius_mean              569 non-null    float64
 2   texture_mean             569 non-null    float64
 3   perimeter_mean           569 non-null    float64
 4   area_mean                569 non-null    float64
 5   smoothness_mean          569 non-null    float64
 6   compactness_mean         569 non-null    float64
 7   concavity_mean           569 non-null    float64
 8   concave points_mean      569 non-null    float64
 9   symmetry_mean            569 non-null    float64
 10  fractal_dimension_mean   569 non-null    float64
 11  radius_se                569 non-null    float64
 12  texture_se               569 non-null    float64
 13  perimeter_se             569 non-null    float64
 14  area_se                  569 non-null    float64
 15  smoothness_se            569 non-null    float64
 16  compactness_se           569 non-null    float64
 17  concavity_se             569 non-null    float64
 18  concave points_se        569 non-null    float64
 19  symmetry_se              569 non-null    float64
 20  fractal_dimension_se     569 non-null    float64
 21  radius_worst             569 non-null    float64
 22  texture_worst            569 non-null    float64
 23  perimeter_worst          569 non-null    float64
 24  area_worst               569 non-null    float64
 25  smoothness_worst         569 non-null    float64
 26  compactness_worst        569 non-null    float64
 27  concavity_worst          569 non-null    float64
 28  concave points_worst     569 non-null    float64
 29  symmetry_worst           569 non-null    float64
 30  fractal_dimension_worst  569 non-null    float64
dtypes: category(1), float64(30)
memory usage: 134.2 KB
"""

# --- Prepare Features and Target Variable ---
# y = target variable (diagnosis: 0 = benign, 1 = malignant)
y = data["diagnosis"]

# X = all predictor variables (all numeric feature columns except diagnosis)
X = data.drop(["diagnosis"], axis=1)

# --- Feature Scaling ---
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
# StandardScaler standardises features by removing the mean and scaling to unit variance
# This ensures that all features contribute equally to the model (important for logistic regression)
scalar = StandardScaler()

# Fit the scaler to the dataset and transform the data
X_scaled = scalar.fit_transform(X)

# --- Train/Test Split ---
from sklearn.model_selection import train_test_split

# Split dataset into training and testing subsets
# 70% of data for training, 30% for testing
# random_state=42 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --- Logistic Regression Model ---
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model instance
# Logistic regression is a classification algorithm that predicts probability of class membership
model = LogisticRegression()

# Fit (train) the model on the training data
model.fit(X_train, y_train)

# Use the trained model to predict labels on the test set
y_pred = model.predict(X_test)

# --- Model Evaluation ---
from sklearn.metrics import accuracy_score

# Calculate the overall accuracy (percentage of correct predictions)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy*100:.2f}%")
"""Accuracy: 98.25%"""

from sklearn.metrics import classification_report

# Generate a classification report: precision, recall, f1-score, and support per class
# precision = true positives / (true positives + false positives)
# recall = true positives / (true positives + false negatives)
# f1-score = harmonic mean of precision and recall
report = classification_report(y_test, y_pred)
# print(report)
"""
              precision    recall  f1-score   support

           0       0.99      0.98      0.99       108
           1       0.97      0.98      0.98        63

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171
"""
