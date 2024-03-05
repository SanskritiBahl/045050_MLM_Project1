#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd, numpy as np # For Data Manipulation
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder # For Encoding Categorical Data [Nominal | Ordinal]
from sklearn.preprocessing import OneHotEncoder # For Creating Dummy Variables of Categorical Data [Nominal]
from sklearn.impute import SimpleImputer, KNNImputer # For Imputation of Missing Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # For Rescaling Data
from sklearn.model_selection import train_test_split # For Splitting Data into Training & Testing Sets 
from sklearn.cluster import KMeans, Birch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load CSV file
import pandas as pd

# Specify the data types for columns
dtype_dict = {
    'column_name_0': str,  # Replace 'column_name_0' with the actual column name in your CSV file
    'column_name_8': str   # Replace 'column_name_8' with the actual column name in your CSV file
}

# Reading the CSV file with specified data types and disabling low_memory warning
df = pd.read_csv('Updated_stock_details_5_years_Companies.csv', dtype=dtype_dict, low_memory=False)
df.head()


# In[19]:


pip install memory-profiler


# In[20]:


from memory_profiler import memory_usage


# This dataset provide details of the various stocks. The variables are opening price, highest valye, lowest value, closing value, volume of the stocks traded, dividends and stock splits of the corresponding companies.
# 
# Open, High, Low, Close: These columns represent the opening, highest, lowest, and closing prices of the stocks, respectively. The statistics include count (number of data points), mean (average value), standard deviation (measure of data dispersion), minimum and maximum values, as well as percentiles (25th, 50th, and 75th percentiles). For example, the mean closing price is approximately 94.78, with a standard deviation of approximately 171.01, indicating variability in the closing prices across the dataset.
# 
# Volume: This column represents the trading volume of the stocks. The statistics show similar information as described above for the other numeric variables. The mean volume is approximately 5.85 million, with a wide range of values indicated by the standard deviation and variation across percentiles.
# 
# Dividends, Stock Splits: These columns represent dividends and stock splits, respectively. The statistics reveal that the majority of data points have zero values for dividends and stock splits, as indicated by the 25th, 50th, and 75th percentiles being zero. The maximum values show outliers in the data, such as a maximum dividend value of 4.23 and a maximum stock split value of 2.0.

# In[21]:


# Display summary statistics of numeric columns
print(df.describe())


# Categorising data into categroical and non categorical data based on their data types. We have date and company as categorical variables and open,high,close,low,dividends and stock splits as non-categorical variables.

# In[22]:


summary = df.dtypes

categorical_vars = summary[summary == 'object'].index.tolist()
non_categorical_vars = summary[summary != 'object'].index.tolist()

print("Categorical Variables:")
print(categorical_vars)

print("\nNon-Categorical Variables:")
print(non_categorical_vars)


# In[23]:


# Divide the data into categorical and non-categorical variables
df_cat = df[['Date','Company']]
df_noncat = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]


# In[24]:


# Correlation heatmap of numeric variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Above is the correlation heatmap for all the numeric variables. This shows a better presentation of the dataset. 
# Each cell in the heatmap represents the correlation coefficient between two of these variables. Red shows postive correlation, blue shows negative and white shows weak correlation.
# 
# From the map we can infer that stong postive correlations are shown by two pairs :  low and high as 0.87 and open and close as 0.99.
# Negative correlation is significantly shown between dividends and stock splits stating that companies that pay more dividends have lower stock splits.

# In[25]:


# Data Pre-Processing
#  Missing Data Treatment
# ---------------------------

# Dataset Used : df_cat

si_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # Strategy = median [When Odd Number of Categories Exists]
si_cat_fit = si_cat.fit_transform(df_cat)
df_cat_mdi = pd.DataFrame(si_cat_fit, columns=df_cat.columns); df_cat_mdi # Missing Categorical Data Imputed Subset
df_cat_mdi.info()


# The output indicates the result of missing 
# data treatment using the SimpleImputer from the scikit-learn library on 
# a pandas DataFrame named df_cat. 
# The DataFrame contains 602,519 entries across 2 columns: "Date" and "Company". 
# Both columns have non-null counts equal to the total number of entries, indicating that there are no missing values remaining after the treatment. 
# The Dtype for both columns is 'object', suggesting that they contain string or categorical data. 
# The memory usage for the DataFrame is reported as 9.2+ MB, reflecting the memory consumed by the DataFrame in the Python environment. 
# This result signifies that the missing data in the DataFrame has been successfully 
# imputed using the 'most frequent' strategy, ensuring that all entries are filled with the most frequent value observed in each respective column.

# In[27]:


# Impute Missing Non-Categorical Data using Descriptive Statistics : Central Tendency
# Dataset Used : df_noncat

si_noncat = SimpleImputer(missing_values=np.nan, strategy='mean')
si_noncat_fit = si_noncat.fit_transform(df_noncat)
df_noncat_mdi_si = pd.DataFrame(si_noncat_fit, columns=df_noncat.columns); df_noncat_mdi_si # Missing Non-Categorical Data Imputed Subset using Simple Imputer
df_noncat_mdi_si.info()


# The output represents the result of imputing missing values in a DataFrame named df_noncat using the SimpleImputer from the scikit-learn library. The DataFrame contains 602,519 entries and 7 columns: "Open", "High", "Low", "Close", "Volume", "Dividends", and "Stock Splits". All columns have non-null counts equal to the total number of entries, indicating that there are no missing values remaining after the imputation process. The Dtype for all columns is 'float64', suggesting that they contain numerical data. The memory usage for the DataFrame is reported as 32.2 MB, reflecting the memory consumed by the DataFrame in the Python environment. This result signifies that the missing non-categorical data in the DataFrame has been successfully imputed using the 'mean' strategy, where missing values are replaced with the mean value of each respective column's non-missing entries, thereby preserving the central tendency of the data.

# In[28]:


#Numeric Encoding of Categorical Data [Nominal & Ordinal] 
# -------------------------------------------------------------

# Dataset Used : df_cat
df_cat_copy = df_cat.copy() 

# Using Pandas (Inferior)
df_cat_copy_pd = df_cat_copy.astype('category')
df_cat_copy_pd['Company'] = df_cat_copy_pd['Company'].cat.codes
# Drop the Date variable
df_cat_copy_pd.dropna(subset=['Date'], inplace=True)
df_cat_copy_pd # (Missing Data Treated) Numeric Coded Categorical Dataset using Pandas


# In[29]:


# Exclude non-numeric columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
data_numeric = df[numeric_cols]

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the numeric data
data_scaled = scaler.fit_transform(data_numeric)
data_scaled

#Pre-Processed Dataset
df_ppd = df_cat_copy_pd .join(df_noncat_mdi_si); df_ppd # Pre-Processed Dataset
df_ppd = pd.merge(df_cat_copy_pd, df_noncat_mdi_si, left_index=True, right_index=True); 
df_ppd


# In[30]:


# Split the preprocessed dataset into training and testing datasets
train_df, test_df = train_test_split(df_ppd, test_size=0.25, random_state=1234)
train_df # Training Dataset
test_df # Testing Dataset


# # Clustering
# 
# Performing the K-Means clustering. Ploting the clusters for training and testing datasets. The datasets contain numeric variables.
# 

# In[31]:


# Exclude non-numeric columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
data_numeric_train = train_df[numeric_cols]
data_numeric_test = test_df[numeric_cols]



# In[32]:


import pandas as pd, numpy as np # For Data Manipulation
import matplotlib.pyplot as plt, seaborn as sns # For Data Visualization
import scipy.cluster.hierarchy as sch # For Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering as agclus, KMeans as kmclus # For Agglomerative & K-Means Clustering
from sklearn.metrics import silhouette_score as sscore, davies_bouldin_score as dbscore # For Clustering Model Evaluation


# In[33]:


#  K-Means Clustering
# -----------------------

#  Determine Value of 'K' in K-Means using Elbow Curve & KMeans-Inertia

# ---------------------------------------------------------------------------

## For training dataset

wcssd = [] # Within-Cluster-Sum-Squared-Distance
nr_clus = range(1,11) # Number of Clusters
for k in nr_clus:
    kmeans = kmclus(n_clusters=k, n_init='auto', random_state=111) 
    kmeans.fit(data_numeric_train)
    wcssd.append(kmeans.inertia_) 
plt.plot(nr_clus, wcssd, marker='x')
plt.xlabel('Values of K') 
plt.ylabel('Within Cluster Sum Squared Distance') 
plt.title('Elbow Curve for Optimal K')
plt.show()



# The x-axis represents the number of clusters (k) tried by the K-Means algorithm (from 1 to 10 in this case).
# The y-axis represents the "Within-Cluster Sum-Squared Distance" (WCSS), which measures how tightly data points within a cluster are grouped together. Lower WCSS values generally indicate better clustering, as data points are closer to their respective cluster centers.
# Interpreting the Curve:
# 
# As the number of clusters increases (moving to the right on the x-axis), the WCSS typically decreases (goes down) at first. This is because splitting data points into more clusters will naturally create smaller and tighter clusters, lowering the WCSS.
# However, at a certain point, adding more clusters won't significantly improve the WCSS, and it might even start to increase slightly. This "elbow" point is often considered the optimal number of clusters (k) to choose, as it balances the trade-off between creating tighter clusters and overfitting the data (creating too many unnecessary clusters).

# In[34]:


# 2.1.1. Determine Value of 'K' in K-Means using Elbow Curve & KMeans-Inertia
# ---------------------------------------------------------------------------
##For testing dataset

''' 
KMeans-Inertia : Sum of Squared Distances of Samples to their closest Cluster Center (Centroid), Weighted by the Sample Weights (if provided)
'''
wcssd = [] # Within-Cluster-Sum-Squared-Distance
nr_clus = range(1,11) # Number of Clusters
for k in nr_clus:
    kmeans = kmclus(n_clusters=k, n_init='auto', random_state=111) 
    kmeans.fit(data_numeric_test)
    wcssd.append(kmeans.inertia_) 
plt.plot(nr_clus, wcssd, marker='x')
plt.xlabel('Values of K') 
plt.ylabel('Within Cluster Sum Squared Distance') 
plt.title('Elbow Curve for Optimal K')
plt.show()


# The x-axis shows the number of clusters (k) tried by the K-Means algorithm (in this case, from 1 to 10).
# The y-axis shows the "Within-Cluster Sum-Squared Distance" (WCSS), which measures how tightly data points within a cluster are grouped together. Lower WCSS values generally indicate better clustering, as data points are closer to their respective cluster centers.
# Interpreting the curve:
# 
# As the number of clusters increases (moving to the right on the x-axis), the WCSS typically decreases (goes down) at first. This is because splitting data points into more clusters will create smaller and tighter clusters, lowering the WCSS.
# However, at a certain point, adding more clusters won't significantly improve the WCSS, and it might even start to increase slightly. This "elbow" point is often considered the optimal number of clusters (k) to choose, as it balances the trade-off between creating tighter clusters and overfitting the data (creating too many unnecessary clusters).

# In[84]:


# 2.1.2. Create K-Means Clusters [K=2]
# ------------------------------------------
#For training dataset

km_2cluster = kmclus(n_clusters=2, n_init='auto', random_state=222)
km_2cluster_model = km_2cluster.fit_predict(data_numeric_train); km_2cluster_model

# 2.1.3. K-Means Clustering Model Evaluation [K=2 | K=3]
# ------------------------------------------------------

sscore_km_2cluster = sscore(data_numeric_train, km_2cluster_model); 
sscore_km_2cluster



# In[36]:


km_2cluster = kmclus(n_clusters=2, n_init='auto', random_state=222)
km_2cluster_model = km_2cluster.fit_predict(data_numeric_train); km_2cluster_model
dbscore_km_2cluster = dbscore(data_numeric_train, km_2cluster_model); 
dbscore_km_2cluster


# In[37]:


# Create K-Means Clusters [K=2]
# ------------------------------------------
#For testing dataset

km_2cluster = kmclus(n_clusters=2, n_init='auto', random_state=222)
km_2cluster_model = km_2cluster.fit_predict(data_numeric_test); km_2cluster_model

# 2.1.3. K-Means Clustering Model Evaluation [K=2 | K=3]
# ------------------------------------------------------

sscore_km_2cluster = sscore(data_numeric_test, km_2cluster_model); sscore_km_2cluster


# In[38]:


km_2cluster = kmclus(n_clusters=2, n_init='auto', random_state=222)
km_2cluster_model = km_2cluster.fit_predict(data_numeric_test); km_2cluster_model

dbscore_km_2cluster = dbscore(data_numeric_test, km_2cluster_model); 
dbscore_km_2cluster


# In[40]:


#  Create K-Means Clusters [K=3]
# ------------------------------------------
#For training dataset

km_3cluster = kmclus(n_clusters=3, n_init='auto', random_state=333)
km_3cluster_model = km_3cluster.fit_predict(data_numeric_train); km_3cluster_model
sscore_km_3cluster = sscore(data_numeric_train, km_3cluster_model); 
sscore_km_3cluster



# In[41]:


km_3cluster = kmclus(n_clusters=3, n_init='auto', random_state=333)
km_3cluster_model = km_3cluster.fit_predict(data_numeric_train); km_3cluster_model
dbscore_km_3cluster = dbscore(data_numeric_train, km_3cluster_model); dbscore_km_3cluster


# In[42]:


#  Create K-Means Clusters [K=3]
# ------------------------------------------
#For testing dataset

km_3cluster = kmclus(n_clusters=3, n_init='auto', random_state=333)
km_3cluster_model = km_3cluster.fit_predict(data_numeric_test); km_3cluster_model
sscore_km_3cluster = sscore(data_numeric_test, km_3cluster_model); sscore_km_3cluster


# In[43]:


km_3cluster = kmclus(n_clusters=3, n_init='auto', random_state=333)
km_3cluster_model = km_3cluster.fit_predict(data_numeric_test); km_3cluster_model
dbscore_km_3cluster = dbscore(data_numeric_test, km_3cluster_model); dbscore_km_3cluster


# For the training data we can observe that for Silhouette Score  as clusters ranges from 2 to 5,the silhouette score decreases slightly, indicating that the separation between clusters decreases as more clusters are added.
# The Davies-Bouldin score also increases slightly as the number of clusters increases, suggesting that the separation between clusters deteriorates with more clusters.Hence, the results suggest that while adding more clusters might capture finer details in the data, it may not necessarily lead to better clustering performance according to these metrics.
# 
# 

# In[45]:


from sklearn.metrics import silhouette_score, davies_bouldin_score
#For training dataset
cluster_range = range(2, 6)  # from 2 to 5 clusters

# Initialize lists to store silhouette and Davies-Bouldin scores for each cluster number
sscore_list = []
dbscore_list = []

# Iterate over the cluster range
for n_clusters in cluster_range:
    # Create K-Means cluster with the current number of clusters
    km_cluster = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    km_cluster_model = km_cluster.fit_predict(data_numeric_train)
    
    # Calculate silhouette score
    sscore_km_cluster = silhouette_score(data_numeric_train, km_cluster_model)
    sscore_list.append(sscore_km_cluster)
    
    # Calculate Davies-Bouldin score
    dbscore_km_cluster = davies_bouldin_score(data_numeric_train, km_cluster_model)
    dbscore_list.append(dbscore_km_cluster)

# Print silhouette and Davies-Bouldin scores for each cluster number
for n_clusters, sscore_km, dbscore_km in zip(cluster_range, sscore_list, dbscore_list):
    print(f"Number of clusters: {n_clusters}, Silhouette score: {sscore_km:.4f}, Davies-Bouldin score: {dbscore_km:.4f}")


# In case of testing dataset,we can observe that for Silhouette Score as clusters ranges from 2 to 5,the silhouette score decreases gradually. This suggests that while the clusters are still reasonably well-defined, they become less distinct as more clusters are added.
# In case of Davies-Bouldin score, as the number of clusters increases, the Davies-Bouldin score also increases. This indicates that the separation between clusters deteriorates as more clusters are added.
# These results suggest that while adding more clusters may provide some additional granularity in clustering the data, it leads to clusters that are less distinct and less well-separated from each other. 

# In[46]:


# Define the range of cluster numbers
#For testing dataset
cluster_range = range(2, 6)  # from 2 to 5 clusters

# Initialize lists to store silhouette and Davies-Bouldin scores for each cluster number
sscore_list = []
dbscore_list = []

# Iterate over the cluster range
for n_clusters in cluster_range:
    # Create K-Means cluster with the current number of clusters
    km_cluster = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    km_cluster_model = km_cluster.fit_predict(data_numeric_test)
    
    # Calculate silhouette score
    sscore_km_cluster = silhouette_score(data_numeric_test, km_cluster_model)
    sscore_list.append(sscore_km_cluster)
    
    # Calculate Davies-Bouldin score
    dbscore_km_cluster = davies_bouldin_score(data_numeric_test, km_cluster_model)
    dbscore_list.append(dbscore_km_cluster)

# Print silhouette and Davies-Bouldin scores for each cluster number
for n_clusters, sscore_km, dbscore_km in zip(cluster_range, sscore_list, dbscore_list):
    print(f"Number of clusters: {n_clusters}, Silhouette score: {sscore_km:.4f}, Davies-Bouldin score: {dbscore_km:.4f}")


# In[47]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Define function to plot clustered data
def plot_clustered_data(dataset, km_cluster_model):
    plt.figure(figsize=(10, 6))
    for cluster_label in np.unique(km_cluster_model):
        plt.scatter(dataset[km_cluster_model == cluster_label]['Volume'], dataset[km_cluster_model == cluster_label]['Close'], label=f'Cluster {cluster_label}')
    centroids = km_cluster.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')
    plt.title('K-Means Clustered Data')
    plt.xlabel('Volume')
    plt.ylabel('Close')
    plt.legend()

# Define the range of cluster numbers
cluster_range = range(2, 6)  # from 2 to 5 clusters

# Iterate over the cluster range
for n_clusters in cluster_range:
    # Create K-Means cluster with the current number of clusters for training data
    km_cluster_train = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    km_cluster_model_train = km_cluster_train.fit_predict(data_numeric_train)
    
    # Create K-Means cluster with the current number of clusters for testing data
    km_cluster_test = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    km_cluster_model_test = km_cluster_test.fit_predict(data_numeric_test)
    
    # Plot clustered data for training and testing datasets
    plt.subplot(2, 2, n_clusters - 1)
    plot_clustered_data(data_numeric_train, km_cluster_model_train)
    plt.title(f'Training Data - {n_clusters} Clusters')
    
    plt.subplot(2, 2, n_clusters + 1)
    plot_clustered_data(data_numeric_test, km_cluster_model_test)
    plt.title(f'Testing Data - {n_clusters} Clusters')

plt.tight_layout()
plt.show()


# Here the Training dataset has been taken into account for clustering. Clusters are color coded in thr graphs and has been intrepreted beloow in two sections. Section 1 : Interpretation based on Volume 
# 
# 
# Section 1 : Interpretation based on Volume 
# Cluster 0 (Blue): This cluster includes stocks that tend to have low trading volume on the x-axis.
# Cluster 1 (Orange): This cluster includes stocks with a medium range of trading volume.
# Cluster 2 (Green): This cluster includes stocks with a higher range of trading volume compared to clusters 0 and 1.
# Cluster 3 (Red): This cluster includes stocks with the highest trading volume on the x-axis
# 
# Section 2 : Interpretation based on Closing Prices 
# Low Volume Stocks (Cluster 0): These stocks might be less liquid (difficult to buy or sell quickly) due to their low trading volume. They could be smaller companies or those that are not actively traded.
# Medium Volume Stocks (Cluster 1): These stocks might have a balance between liquidity and potential for price movements due to their moderate trading volume.
# High Volume Stocks (Clusters 2 & 3): These stocks are likely more liquid (easier to buy or sell quickly) due to their high trading volume. They could be larger companies or those that are more actively traded.

# In[49]:


# 2.2. Create a KMeans Cluster Member Dataframe
# ---------------------------------------------

# Cluster Model Used : km_3cluster_model

data_numeric_test_kmcluster = data_numeric_test.copy()
data_numeric_test_kmcluster.reset_index(level=0, inplace=True, names='stock_index')
data_numeric_test_kmcluster['cluster_number'] = km_3cluster_model
data_numeric_test_kmcluster.sort_values('cluster_number', inplace=True); 
data_numeric_test_kmcluster

#mtcars_subset_kmcluster = pd.DataFrame()
#mtcars_subset_kmcluster['Car_Index'] = mtcars_subset.index.values
#mtcars_subset_kmcluster['Cluster_Number'] = km_3cluster_model
#mtcars_subset_kmcluster.sort_values('Cluster_Number', inplace=True); mtcars_subset_kmcluster

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2.3. Plot the K-Means Clustered Data 
# ------------------------------------

# Cluster Model Used : km_3cluster_model

cluster_labels = list(data_numeric_test_kmcluster['cluster_number'].unique())
scatter_plot = plt.scatter(x='Volume', y='Close', data=data_numeric_test_kmcluster, c='cluster_number', cmap='brg')
plt.legend(handles=scatter_plot.legend_elements()[0], labels=cluster_labels)
plt.xlabel('Volume')
plt.ylabel('Close')
plt.title('Volume vs Close Scatter Plot : K-Means Clusters for Testing Data')
plt.grid()
plt.show()

#km_clusterplot_sb = sns.scatterplot(x='hp', y='mpg', hue='cluster_number', palette='brg', data=data_numeric_test_kmcluster)
#km_clusterplot_sb.set(title='Volume vs Close Scatter Plot : K-Means Clusters')


# Here the Testing dataset has been taken into account for clustering. Clusters are color coded in thr graphs and has been intrepreted beloow in two sections. Section 1 : Interpretation based on Volume 
# 
# 
# Section 1 : Interpretation based on Volume 
# Cluster 0 (Blue): This cluster includes stocks that tend to have low trading volume on the x-axis.
# Cluster 1 (Orange): This cluster includes stocks with a medium range of trading volume.
# Cluster 2 (Green): This cluster includes stocks with a higher range of trading volume compared to clusters 0 and 1.
# Cluster 3 (Red): This cluster includes stocks with the highest trading volume on the x-axis
# 
# Section 2 : Interpretation based on Closing Prices 
# Low Volume Stocks (Cluster 0): These stocks might be less liquid (difficult to buy or sell quickly) due to their low trading volume. They could be smaller companies or those that are not actively traded.
# Medium Volume Stocks (Cluster 1): These stocks might have a balance between liquidity and potential for price movements due to their moderate trading volume.
# High Volume Stocks (Clusters 2 & 3): These stocks are likely more liquid (easier to buy or sell quickly) due to their high trading volume. They could be larger companies or those that are more actively traded.

# In[53]:


from sklearn.cluster import KMeans
import numpy as np

# Assuming 'X' is your data matrix

# Step 1: Fit the k-means model
kmeans = KMeans(n_clusters=5)
kmeans.fit(data_numeric_test)

# Step 2: Retrieve cluster centroids
centroids = kmeans.cluster_centers_

# Step 3: Assign each data point to its corresponding cluster
cluster_labels = kmeans.labels_

# Step 4: Calculate cluster statistics
for i in range(2,5):
    cluster_data = data_numeric_test[cluster_labels == i]
    cluster_size = len(cluster_data)
    centroid = centroids[i]
    within_cluster_sos = np.sum((cluster_data - centroid) ** 2)
    print(f"Cluster {i+1} - Size: {cluster_size}, Centroid: {centroid}, Within-cluster Sum of Squares: {within_cluster_sos}")


# Cluster 3: This cluster contains 28 data points. The centroid values for the features indicate relatively high values across all numerical variables. The within-cluster sum of squares (WCSS) is considerable, indicating that the data points within this cluster are relatively spread out from the centroid. The volume feature has an extremely high value, which contributes significantly to the overall WCSS.
# 
# Cluster 4: This cluster contains 297 data points. The centroid values for the features are slightly lower compared to Cluster 3, but still relatively high. The within-cluster sum of squares (WCSS) is also notable, indicating a spread-out distribution of data points around the centroid. The volume feature has a high value, contributing to the overall WCSS.
# 
# Cluster 5: This cluster contains 1618 data points, making it the largest cluster among the three. The centroid values for the features are comparatively lower than Clusters 3 and 4, but still significant. The within-cluster sum of squares (WCSS) is substantial, suggesting a spread-out distribution of data points around the centroid. The volume feature has a relatively lower value compared to Clusters 3 and 4, but it still contributes significantly to the overall WCSS.
# 
# 
# Cluster 3 represents a group of data points with high values across all features, especially in the volume feature. Cluster 4 exhibits slightly lower values compared to Cluster 3 but still maintains significant values across all features. Cluster 5, the largest cluster, has comparatively lower centroid values but still exhibits considerable variability in its data points, with the volume feature also playing a significant role in the cluster's composition.

# In[55]:


from sklearn.cluster import KMeans
import numpy as np

# Assuming 'X' is your data matrix

# Step 1: Fit the k-means model
kmeans = KMeans(n_clusters=5)
kmeans.fit(data_numeric_train)

# Step 2: Retrieve cluster centroids
centroids = kmeans.cluster_centers_

# Step 3: Assign each data point to its corresponding cluster
cluster_labels = kmeans.labels_

# Step 4: Calculate cluster statistics
for i in range(2,5):
    cluster_data = data_numeric_train[cluster_labels == i]
    cluster_size = len(cluster_data)
    centroid = centroids[i]
    within_cluster_sos = np.sum((cluster_data - centroid) ** 2)
    print(f"Cluster {i+1} - Size: {cluster_size}, Centroid: {centroid}, Within-cluster Sum of Squares: {within_cluster_sos}")


# Cluster 3: This cluster contains 24 data points. The centroid values for the features indicate relatively high values across all numerical variables. The within-cluster sum of squares (WCSS) is considerable, indicating that the data points within this cluster are relatively spread out from the centroid. The volume feature has an extremely high value, which contributes significantly to the overall WCSS.
# 
# Cluster 4: This cluster contains 4089 data points, making it the largest cluster among the three. The centroid values for the features are moderate, with slightly lower values compared to Cluster 3. However, the within-cluster sum of squares (WCSS) is substantial, indicating a spread-out distribution of data points around the centroid. The volume feature has a relatively lower value compared to Cluster 3, but it still contributes significantly to the overall WCSS.
# 
# Cluster 5: This cluster contains 219 data points. The centroid values for the features are lower compared to Clusters 3 and 4 but still significant. The within-cluster sum of squares (WCSS) is notable, suggesting a spread-out distribution of data points around the centroid. The volume feature has a lower value compared to Clusters 3 and 4, but it still contributes significantly to the overall WCSS.
# 
# Cluster 3 represents a group of data points with high values across all features, especially in the volume feature. Cluster 4 exhibits moderate values compared to Cluster 3 but still maintains significant variability in its data points, with the volume feature playing a substantial role. Cluster 5 has comparatively lower centroid values but still demonstrates notable variability in its data points, with the volume feature also contributing significantly to the cluster's composition.

# In[56]:


# Exclude non-numeric columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
data_numeric_train = train_df[numeric_cols]
data_numeric_test = test_df[numeric_cols]

# Measure memory usage before running the code
memory_before = memory_usage()

# Start time measurement
start_time = time.time()

# The provided Python code goes here

# End time measurement
end_time = time.time()

# Measure memory usage after running the code
memory_after = memory_usage()

# Calculate memory usage difference
memory_diff = memory_after[0] - memory_before[0]

# Calculate time taken
time_taken = end_time - start_time

print(f"Memory usage difference: {memory_diff} MB")
print(f"Time taken: {time_taken} seconds")


# The memory usage difference of 0.03125 MB indicates the change in memory consumption before and after running the provided Python code. This difference represents the additional memory allocated during the execution of the code. In this case, the increase in memory usage is relatively small, suggesting that the code does not require a significant amount of memory to execute.
# 
# Regarding the time taken, the result of 6.985664367675781e-05 seconds indicates the duration of time elapsed during the execution of the code. This time measurement is very small, indicating that the code executed quickly. The execution time is essential for assessing the efficiency of the code, especially when dealing with large datasets or complex operations. In this scenario, the code executed swiftly, which is favorable for optimizing performance and responsiveness in real-time applications.
