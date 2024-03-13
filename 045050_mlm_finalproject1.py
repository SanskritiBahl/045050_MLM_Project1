#!/usr/bin/env python
# coding: utf-8

# # Project Title: Exploring Car Diversity 
# 
# 1. The project explores various diversities in terms of price, taxes mileage and engine size.
# 
# Project Contents : 
# 1. Project Objectives | Problem Statements
# 2. Description of Data
# 3. Analysis of Data
# 4. Results | Observations
# 5. Managerial Insights
# 
# 1. Project Objectives | Problem Statements
# 1.1. PO1 | PS1: Segmentation of Cars Data using Unsupervised Machine Learning Clustering Algorithms
# 1.2. PO2 | PS2: Identification of Appropriate Number of Segments or Clusters
# 1.3. PO3 | PS3: Determination of Segment or Cluster Characteristics
# 
# 2. Description of Data
# The dataset used for analysis was sourced from Kaggle and comprises information on approximately 97,712 records pertaining to 10 variables related to cars. The data size is approximately 28.08 MB, indicating a substantial volume of information to analyze. Each record in the dataset provides details on various attributes such as car manufacturer, model, transmission type, fuel type, price, mileage, tax, and engine size. This dataset offers a comprehensive overview of car characteristics and serves as a valuable resource for conducting exploratory data analysis and deriving insights into the automotive industry (The results are presented in cell 84).
# 
# 2.2. Description of Variables
# 
# The dataset comprises both categorical and non-categorical variables. Among the categorical variables, there are four features with nominal categories: model, fuelType, and Manufacturer. These variables represent distinct categories without any inherent order. Additionally, the transmission variable falls under categorical features with ordinal categories, implying a specific order or hierarchy among its values. On the other hand, the non-categorical variables include year, price, mileage, tax, mpg (miles per gallon), and engineSize, which are continuous numerical variables representing various attributes related to the cars in the dataset. Overall, the dataset encompasses a diverse range of variables capturing both qualitative and quantitative aspects of car characteristics, facilitating comprehensive analysis and insights into the automotive domain.(The results are shown in cell 85)
# 
# 2.3. Descriptive Statistics 
# The categorical variables in the dataset provide valuable insights into the distribution of car characteristics. In terms of model distribution, Fiesta emerges as the most frequent model, accounting for approximately 6.66% of the total, followed by Golf and Focus. Manual transmission dominates the dataset, constituting approximately 56.80% of the total, while Semi-Auto and Automatic transmissions follow with 22.82% and 20.37%, respectively. Petrol-fueled cars are the most prevalent, comprising approximately 55.25% of the dataset, followed by Diesel at 41.37%. Among manufacturers, Ford has the highest representation at 18.23%, followed by Volkswagen and Vauxhall. 
# 
# The non-categorical variables exhibit various measures of central tendency and dispersion. The average year of the cars in the dataset is around 2017, with a standard deviation of approximately 2.12 years, indicating a relatively narrow spread of model years. The average price of cars is approximately 16,773, with a substantial standard deviation of 9,869, revealing considerable variability in car prices. Similarly, the average mileage stands at approximately 23,219 miles, with a standard deviation of around 21,061 miles, indicating a wide range of mileage among cars in the dataset. The average tax amount is approximately 120, with a standard deviation of approximately 63. The average miles per gallon (mpg) is approximately 55.21, with a standard deviation of approximately 16.18. Lastly, the average engine size is around 1.66, with a standard deviation of approximately 0.56. 
# 
# Additionally, correlation statistics reveal relationships between non-categorical variables. There is a moderate positive correlation between year and price (r = 0.49), indicating that newer cars tend to have higher prices. Conversely, there is a strong negative correlation between year and mileage (r = -0.74), implying that older cars generally have higher mileage. A weak positive correlation exists between price and tax (r = 0.31), indicating that higher-priced cars tend to have higher taxes. Furthermore, there is a moderate negative correlation between price and mpg (r = -0.30), suggesting that cars with higher prices tend to have lower miles per gallon. Additionally, there is a weak positive correlation between price and engine size (r = 0.64), indicating that higher-priced cars tend to have larger engine sizes. (the results are shown in cell 86)
# 
# 3. Analysis of Data
# 
# 3.1. Data Pre-Processing
# 
# 3.1.1. Missing Data Statistics and Treatment
# 
# The DataFrame consists of a mix of data types: four object columns (representing categorical variables), four int64 columns (representing integer variables), and two float64 columns (representing floating-point or decimal variables). There are no missing values (non-null counts) in any of the variables. Additionally,the dataframe occupies approximately 7.5+ MB of memory.Additionally none of the rescords have missing values.(the results are shown in cell 87 and 88).
# 
# 3.1.2. Numerical Encoding of Categorical Variables or Features 
# 
# Three categorical variables, namely 'transmission', 'fuelType', and 'Manufacturer', are encoded into numerical representations using mapping dictionaries. For example, the 'transmission' variable, which originally had categories like 'Manual', 'Semi-Auto', 'Automatic', and 'Other', is encoded into numerical labels 0, 1, 2, and 3, respectively. The variable 'price' has 3825 outliers, 'mileage' has 3836 outliers, 'tax' has 28594 outliers, 'mpg' has 930 outliers, and 'engineSize' has 648 outliers. These counts highlight the presence of data points that are significantly different from the majority of observations within each respective variable. (The results are shown in cell 92). For the treatment of the outliers min-max scalling have been used. Here the non categorical variables have been put within the range of [0,1]. (the results are shown in cell 93).
# 
# Further the non categorical variable dataset and categorical dataset are merged and hence named as pre-processed data set. Subset of the pre-processed dataset have been taken which contains these variable : 'Manufacturer_enc','model_enc','transmission_enc','fuelType_enc','price','mileage','tax','engineSize'. 
# 
# 3.2 PO1 and PO2| PS1 and PS2: Identification of Appropriate Number of Segments or ClustersUnsupervised Machine Learning Clustering Algorithm: K-Means (Base Model) 
# 
# Based on the Davies-Bouldin (DB) score, which is a metric used for evaluating clustering algorithms where lower values indicate better clustering, the optimal number of clusters appears to be 4. This is because the DB score is lowest (0.403) when K=4 clusters, indicating that the clusters are more separable and distinct compared to other values of K. Additionally, considering the silhouette (SS) score, which measures the compactness and separation of the clusters (higher values indicate better-defined clusters), the SS score is relatively high (0.677) for K=4 clusters, further supporting the choice of 4 clusters. Therefore, based on both the DB score and SS score, 4 clusters seem to be the most appropriate choice for partitioning the data in this scenario.
# (Results from the cells 79,97,98,99,102,103 and 104)
# (ss score and db score are for K = 2 clusters 0.6267625599600847 and 0.5574726168752606
# ss score and db score are for K = 3 clusters 0.5689062348222101 and 0.5378836668469141
# ss score and db score are for K = 4 clusters 0.6769623940921728 and 0.40324035586816387
# ss score and db score are for K = 5 clusters 0.6271411200820675 and 0.47295457154613113)
# 
# 3.3 KMeans Cluster Member Dataframe have been perpared since now approporiate number of clusters are 4. For the dataset where there are 0,1,2,3 clusters which means K = 2, K = 3 and K = 4. This dataset have been divided onto various data frames. Features of the clusters are analysed on the basis of categorical and non-categorical variables.
# The average price among various clusters are different.There are statistically significant differences in the mean price values between all pairs of clusters. Within various clusters, in atleast one of the clusters average mileage and tax is not same. There are statistically significant differences in the mean mileage and taxes values between all pairs of clusters.
# 
# In Cluster 0, the chi-square tests for all categorical variables (model_enc, transmission_enc, fuelType_enc, Manufacturer_enc) yielded a chi-square statistic of 0.0 and a p-value of 1.0, indicating no significant association between these variables and cluster number. This suggests that the distribution of car models, transmission types, fuel types, and manufacturers is similar across Cluster 0, with no discernible pattern or differentiation based on these categorical variables.
# 
# Similarly, in Cluster 1, Cluster 2, and Cluster 3, the chi-square tests for all categorical variables produced a chi-square statistic of 0.0 and a p-value of 1.0, indicating no significant relationship between these variables and cluster number within each respective cluster. This implies that the distribution of car models, transmission types, fuel types, and manufacturers is consistent across all clusters, with no significant differences in the composition of these categorical variables among the clusters.
# 
# Overall, the results suggest that there are no distinct patterns or associations observed between the categorical variables (car models, transmission types, fuel types, and manufacturers) and cluster numbers across different clusters. Therefore, these categorical variables do not contribute significantly to the differentiation or segmentation of clusters based on car features within the dataset.
# 
# 4. Results | Observations
# 
# The analysis of the dataset using unsupervised machine learning clustering algorithms, particularly K-Means, resulted in the identification of 4 distinct clusters as the optimal segmentation for the data. These clusters exhibit differences in various car attributes such as price, mileage, tax, and engine size. Additionally, the chi-square tests conducted on categorical variables within each cluster revealed no significant association between these variables (car model, transmission type, fuel type, manufacturer) and cluster number.
# 
# Observations from the analysis include:
# 
# a. Cluster Segmentation: The dataset was successfully segmented into 4 clusters based on car attributes, with each cluster exhibiting distinct characteristics in terms of price, mileage, tax, and engine size.
# b. Categorical Variables: Chi-square tests indicated no significant association between categorical variables (car model, transmission type, fuel type, manufacturer) and cluster number within each cluster, suggesting a consistent distribution of these variables across all clusters.
# c. Statistical Differences: Significant differences were observed in the mean price, mileage, and tax values between clusters, indicating varying car characteristics among the clusters.
# d. Cluster Homogeneity: Despite statistical differences in numerical variables, categorical variables did not contribute significantly to the differentiation or segmentation of clusters, implying homogeneity in the distribution of car models, transmission types, fuel types, and manufacturers across clusters.
# 
# 5. Managerial Implications
# 
# a. Market Segmentation: The identification of distinct clusters based on numerical variables such as price, mileage, and tax can inform marketing strategies. Companies can tailor their marketing efforts based on the characteristics of each cluster, targeting specific customer segments with relevant pricing, mileage, and tax considerations.
# 
# b. Product Development: Understanding the diversity in car features across clusters can guide product development initiatives. Manufacturers can focus on enhancing or introducing features that appeal to different clusters, thereby meeting the varied preferences and needs of consumers within each segment.
# 
# c. Inventory Management: Dealerships and manufacturers can optimize their inventory management processes by stocking vehicles that align with the preferences of each cluster. By understanding the distribution of car models, transmission types, fuel types, and manufacturers within clusters, businesses can ensure efficient allocation of resources and inventory.
# 
# d. Competitive Analysis: Analyzing the distribution of car manufacturers across clusters  provide insights into market competitiveness. Companies can assess their market share within each segment and identify areas for growth or improvement compared to competitors.
# 
# e. Strategic Pricing: Variations in average price across clusters highlight opportunities for strategic pricing. Businesses can adjust pricing strategies based on the preferences and purchasing behaviors of consumers within each cluster, optimizing revenue and profitability.
# 
# f. Customer Targeting: Understanding the characteristics of each cluster based on mainly non-categorical variables such as price, tax, mileage and engine size enables targeted customer engagement and communication strategies. Companies can personalize their messaging and offerings to resonate with the preferences and priorities of consumers within specific segments, enhancing customer satisfaction and loyalty.
# 

# In[134]:


import pandas as pd

# Load CSV file
df = pd.read_csv('CarsData.csv')
df.head()


# In[135]:


# Data Source
data_source = "https://www.kaggle.com/datasets/meruvulikith/90000-cars-data-from-1970-to-2024/data"  # Fill in the actual data source if available

# Data Size
data_size = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert to MB

# Data Shape
num_variables = df.shape[1]
num_records = df.shape[0]

# Print Description of Data
print("2. Description of Data")
print("\t2.1. Data Source, Size, Shape")
print(f"\t\t2.1.1. Data Source: {data_source}")
print(f"\t\t2.1.2. Data Size: {data_size:.2f} MB")
print(f"\t\t2.1.3. Data Shape: Dimension: {num_variables} variables | {num_records} records")


# In[136]:


# 2.2. Description of Variables
print("2.2. Description of Variables")

# 2.2.1. Index Variable(s)
index_variables = df.index.name
print(f"\t2.2.1. Index Variable(s): {index_variables}")

# 2.2.2. Variables or Features having Categories | Categorical Variables or Features (CV)
categorical_variables = df.select_dtypes(include=['object']).columns.tolist()
print("\t2.2.2. Variables or Features having Categories | Categorical Variables or Features (CV):")
for var in categorical_variables:
    print(f"\t\t{var}")

# 2.2.2.1. Variables or Features having Nominal Categories | Categorical Variables or Features - Nominal Type
nominal_variables = ['model', 'fuelType', 'Manufacturer']
print("\t2.2.2.1. Variables or Features having Nominal Categories | Categorical Variables or Features - Nominal Type:")
for var in nominal_variables:
    print(f"\t\t{var}")

# 2.2.2.2. Variables or Features having Ordinal Categories | Categorical Variables or Features - Ordinal Type
ordinal_variables = ['transmission']
print("\t2.2.2.2. Variables or Features having Ordinal Categories | Categorical Variables or Features - Ordinal Type:")
for var in ordinal_variables:
    print(f"\t\t{var}")

# 2.2.3. Non-Categorical Variables or Features
non_categorical_variables = df.select_dtypes(exclude=['object']).columns.tolist()
print("\t2.2.3. Non-Categorical Variables or Features:")
for var in non_categorical_variables:
    print(f"\t\t{var}")


# In[137]:


# 2.3. Descriptive Statistics

# 2.3.1. Descriptive Statistics: Categorical Variables or Features
print("2.3.1. Descriptive Statistics: Categorical Variables or Features")

# 2.3.1.1. Count | Frequency Statistics
print("\t2.3.1.1. Count | Frequency Statistics:")
categorical_variables = ['model', 'transmission', 'fuelType', 'Manufacturer']
for var in categorical_variables:
    print(f"\t\t{var}:")
    print(df[var].value_counts())

# 2.3.1.2. Proportion (Relative Frequency) Statistics
print("\t2.3.1.2. Proportion (Relative Frequency) Statistics:")
for var in categorical_variables:
    print(f"\t\t{var}:")
    print(df[var].value_counts(normalize=True))

# 2.3.2. Descriptive Statistics: Non-Categorical Variables or Features
print("2.3.2. Descriptive Statistics: Non-Categorical Variables or Features")

# 2.3.2.1. Measures of Central Tendency
print("\t2.3.2.1. Measures of Central Tendency:")
non_categorical_variables = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
print(df[non_categorical_variables].mean())
print(df[non_categorical_variables].median())

# 2.3.2.2. Measures of Dispersion
print("\t2.3.2.2. Measures of Dispersion:")
print(df[non_categorical_variables].std())
print(df[non_categorical_variables].var())

# 2.3.2.3. Correlation Statistics (with Test of Correlation)
print("\t2.3.2.3. Correlation Statistics (with Test of Correlation):")
print(df[non_categorical_variables].corr())


# In[138]:


#Analysis of Data

#Data Pre-Processing Data
#Missing data in each of the variables
df.info() # Dataframe Information (Provide Information on Missing Data)
variable_missing_data = df.isna().sum(); variable_missing_data # Variable-wise Missing Data Information

##None of the variables contain any missing data, hence there isn't any need of missing treatmet.


# In[139]:


record_missing_data = df.isna().sum(axis=1).sort_values(ascending=False).head(); record_missing_data # Record-wise Missing Data Information (Top 5)


# In[140]:


# Check for NaN values in the entire DataFrame
nan_values = df.isna().any()

# Print the columns with NaN values
print("Columns with NaN values:")
print(nan_values[nan_values].index)

##So it is evident from the result that all columns in the DataFrame are free from missing values, and there are no NaN values present.


# In[141]:


df_cat = df[['model','transmission','fuelType','Manufacturer']] # Categorical Data [Nominal | Ordinal]
df_noncat = df[['year','price', 'mileage','tax','mpg','engineSize']] # Non-Categorical Data


# In[142]:


from sklearn.preprocessing import LabelEncoder

# Define mapping dictionaries
transmission_mapping = {'Manual': 0, 'Semi-Auto': 1, 'Automatic': 2, 'Other': 3}
fuelType_mapping = {'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3, 'Other': 4}
Manufacturer_mapping = {'ford': 0, 'volkswagen': 1, 'vauxhall': 2, 'merc': 3, 'BMW': 4, 'Audi': 5, 'toyota': 6, 'skoda': 7, 'hyundi': 8}

# Perform numerical encoding for 'transmission'
df_cat['transmission_enc'] = df_cat['transmission'].map(transmission_mapping)

# Perform numerical encoding for 'fuelType'
df_cat['fuelType_enc'] = df_cat['fuelType'].map(fuelType_mapping)

# Perform numerical encoding for 'Manufacturer'
df_cat['Manufacturer_enc'] = df_cat['Manufacturer'].map(Manufacturer_mapping)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'model'
df_cat['model_enc'] = label_encoder.fit_transform(df_cat['model'])

# Display the updated DataFrame
print(df_cat)


# In[143]:


#Outlier treatment

def outlier_statistics(df):
    # Define non-categorical variables
    non_cat_vars = ['price', 'mileage', 'tax', 'mpg', 'engineSize']
    
    # Compute the first and third quartiles
    Q1 = df[non_cat_vars].quantile(0.25)
    Q3 = df[non_cat_vars].quantile(0.75)
    
    # Compute the interquartile range
    IQR = Q3 - Q1
    
    # Compute lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers for each variable
    outliers_count = ((df[non_cat_vars] < lower_bound) | (df[non_cat_vars] > upper_bound)).sum()
    
    return outliers_count

# Compute outlier statistics
outliers_stats = outlier_statistics(df_noncat)
print("Outlier Statistics for Non-Categorical Variables:")
print(outliers_stats)

##For the variable 'price', there are 3825 outliers.
##For the variable 'mileage', there are 3836 outliers.
##For the variable 'tax', there are 28594 outliers.
##For the variable 'mpg', there are 930 outliers.
##For the variable 'engineSize', there are 648 outliers.


# In[144]:


from sklearn.preprocessing import MinMaxScaler

# Define the non-categorical variables
non_cat_vars = ['price', 'mileage', 'tax', 'mpg', 'engineSize']

# Step 1: Outlier Statistics
outlier_stats = df_noncat[non_cat_vars].describe(percentiles=[0.25, 0.75]).T
outlier_stats['IQR'] = outlier_stats['75%'] - outlier_stats['25%']
outlier_stats['Lower Bound'] = outlier_stats['25%'] - 1.5 * outlier_stats['IQR']
outlier_stats['Upper Bound'] = outlier_stats['75%'] + 1.5 * outlier_stats['IQR']

# Step 2: Outlier Treatment (Winsorization)
for var in non_cat_vars:
    lower_bound = outlier_stats.loc[var, 'Lower Bound']
    upper_bound = outlier_stats.loc[var, 'Upper Bound']
    df_noncat[var] = df_noncat[var].clip(lower=lower_bound, upper=upper_bound)

# Step 3: Min-Max Scaling
scaler = MinMaxScaler()
df_noncat_scaled = df_noncat.copy()
df_noncat_scaled[non_cat_vars] = scaler.fit_transform(df_noncat_scaled[non_cat_vars])

# Display outlier statistics
print("Outlier Statistics:")
print(outlier_stats)

# Display the first few rows of the scaled DataFrame
print("\nScaled DataFrame:")
print(df_noncat_scaled.head())


# In[145]:


# Pre-Processed Dataset
df_ppd = df_cat.join(df_noncat_scaled); df_ppd # Pre-Processed Dataset
#df_ppd = pd.merge(df_cat_ppd, df_noncat_ppd, left_index=True, right_index=True); df_ppd


# In[146]:


# Subset of df_ppd 
# Perform numerical encoding for 'model'
df_ppd['model_enc'] = label_encoder.fit_transform(df_ppd['model'])

# Subset of df_ppd 
df_ppd_subset = df_ppd[['Manufacturer_enc','model_enc','transmission_enc','fuelType_enc','price','mileage','tax','engineSize']]
df_ppd_subset


# In[147]:


from pydataset import data # For Datasets
import pandas as pd, numpy as np # For Data Manipulation
import matplotlib.pyplot as plt, seaborn as sns # For Data Visualization
import scipy.cluster.hierarchy as sch # For Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering as agclus, KMeans as kmclus # For Agglomerative & K-Means Clustering
from sklearn.metrics import silhouette_score as sscore, davies_bouldin_score as dbscore # For Clustering Model Evaluation

# K-Means Clustering
# -----------------------

# 2.1.1. Determine Value of 'K' in K-Means using Elbow Curve & KMeans-Inertia
# ---------------------------------------------------------------------------
''' 
KMeans-Inertia : Sum of Squared Distances of Samples to their closest Cluster Center (Centroid), Weighted by the Sample Weights (if provided)
'''
wcssd = [] # Within-Cluster-Sum-Squared-Distance
nr_clus = range(1,11) # Number of Clusters
for k in nr_clus:
    kmeans = kmclus(n_clusters=k, init='random', random_state=111) 
    kmeans.fit(df_ppd_subset)
    wcssd.append(kmeans.inertia_) 
plt.plot(nr_clus, wcssd, marker='x')
plt.xlabel('Values of K') 
plt.ylabel('Within Cluster Sum Squared Distance') 
plt.title('Elbow Curve for Optimal K')
plt.show()


# In[148]:


# Create K-Means Clusters [K=2]
# ------------------------------------------

km_2cluster = kmclus(n_clusters=2, n_init='auto', random_state=222)
km_2cluster_model = km_2cluster.fit_predict(df_ppd_subset); km_2cluster_model

# 2.1.3. K-Means Clustering Model Evaluation [K=2 | K=3]
# ------------------------------------------------------

sscore_km_2cluster = sscore(df_ppd_subset, km_2cluster_model); sscore_km_2cluster


# In[149]:


dbscore_km_2cluster = dbscore(df_ppd_subset, km_2cluster_model); dbscore_km_2cluster


# In[150]:


#K-Means Clusters [K=3]

km_3cluster = kmclus(n_clusters=3, n_init='auto', random_state=333)
km_3cluster_model = km_3cluster.fit_predict(df_ppd_subset); km_3cluster_model

sscore_km_3cluster = sscore(df_ppd_subset, km_3cluster_model); sscore_km_3cluster


# In[151]:


# dbscore
dbscore_km_3cluster = dbscore(df_ppd_subset, km_3cluster_model); dbscore_km_3cluster


# In[152]:


#K-Means Clusters [K=4]

km_4cluster = kmclus(n_clusters=4, n_init='auto', random_state=444)
km_4cluster_model = km_4cluster.fit_predict(df_ppd_subset); km_4cluster_model

sscore_km_4cluster = sscore(df_ppd_subset, km_4cluster_model); sscore_km_4cluster


# In[104]:


# dbscore
dbscore_km_4cluster = dbscore(df_ppd_subset, km_4cluster_model); dbscore_km_4cluster


# In[153]:


#K-Means Clusters [K=5]

km_5cluster = kmclus(n_clusters=5, n_init='auto', random_state=555)
km_5cluster_model = km_5cluster.fit_predict(df_ppd_subset); km_5cluster_model

sscore_km_5cluster = sscore(df_ppd_subset, km_5cluster_model); sscore_km_5cluster


# In[154]:


# dbscore
dbscore_km_5cluster = dbscore(df_ppd_subset, km_5cluster_model); dbscore_km_5cluster


# In[155]:


# Create a KMeans Cluster Member Dataframe
# ---------------------------------------------
# In context of 4 clusters
# Cluster Model Used : km_4cluster_model

df_ppd_subset_kmcluster = df_ppd_subset.copy()
df_ppd_subset_kmcluster.reset_index(level=0, inplace=True, names='car_index')
df_ppd_subset_kmcluster['cluster_number'] = km_4cluster_model
df_ppd_subset_kmcluster.sort_values('cluster_number', inplace=True); df_ppd_subset_kmcluster

#mtcars_subset_kmcluster = pd.DataFrame()
#mtcars_subset_kmcluster['Car_Index'] = mtcars_subset.index.values
#mtcars_subset_kmcluster['Cluster_Number'] = km_4cluster_model
#mtcars_subset_kmcluster.sort_values('Cluster_Number', inplace=True); df_ppd_subset_kmcluster


# In[156]:


# Segregate DataFrame based on cluster number
df_cluster_0 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 0]
df_cluster_1 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 1]
df_cluster_2 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 2]
df_cluster_3 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 3]

# Now df_cluster_0 contains rows with cluster number 0, df_cluster_1 contains rows with cluster number 1, and so on.


# In[157]:


df_cluster_0


# In[158]:


# Null Hypothesis : The average price across all clusters are same
# Alternate Hypithesis : In atleast one cluster the average price is different
import scipy.stats as sps # For Probability & Inferential Statistics
# ANOVA Using Scipy
mean_test_anova_scipy = sps.f_oneway(df_cluster_0.price,df_cluster_1.price,df_cluster_2.price,df_cluster_3.price) # ANOVA
mean_test_anova_scipy

## Give the p-value as 0.0 which is less than 0.05, hence null hypothesis is rejected. So in atleast one of the clusters average price is not same. 


# In[159]:


# For each pairwise comparison, the p-value is very low (0.000), indicating that the observed differences 
# in means are statistically significant.

#The confidence intervals (Lower CI and Upper CI) do not include zero for any pairwise comparison, which further supports the conclusion that the differences in means are statistically significant.

#Therefore, it can be inferred that there are statistically significant differences in the mean price values between all pairs of clusters 
# Pairwise Comparision Using Scipy (Tukey HSD)
mean_pairwise_compare_scipy = sps.tukey_hsd(df_cluster_0.price,df_cluster_1.price,df_cluster_2.price,df_cluster_3.price)
print(mean_pairwise_compare_scipy)


# In[160]:


# Null Hypothesis : The average tax across all clusters are same
# Alternate Hypothesis : In atleast one cluster the average price is different
import scipy.stats as sps # For Probability & Inferential Statistics
# ANOVA Using Scipy
mean_test_anova_scipy = sps.f_oneway(df_cluster_0.tax,df_cluster_1.tax,df_cluster_2.tax,df_cluster_3.tax) # ANOVA
mean_test_anova_scipy

## Give the p-value as 0.0 which is less than 0.05, hence null hypothesis is rejected. So in atleast one of the clusters average price is not same. 


# In[161]:


# For each pairwise comparison, the p-value is very low (0.000), indicating that the observed differences 
# in means are statistically significant.

#The confidence intervals (Lower CI and Upper CI) do not include zero for any pairwise comparison, which further supports the conclusion that the differences in means are statistically significant.

#Therefore, it can be inferred that there are statistically significant differences in the mean tax values between all pairs of clusters 
# Pairwise Comparision Using Scipy (Tukey HSD)
mean_pairwise_compare_scipy = sps.tukey_hsd(df_cluster_0.tax,df_cluster_1.tax,df_cluster_2.tax,df_cluster_3.tax)
print(mean_pairwise_compare_scipy)


# In[162]:


# Null Hypothesis : The average mileage across all clusters are same
# Alternate Hypothesis : In atleast one cluster the average mileage is different
import scipy.stats as sps # For Probability & Inferential Statistics
# ANOVA Using Scipy
mean_test_anova_scipy = sps.f_oneway(df_cluster_0.mileage,df_cluster_1.mileage,df_cluster_2.mileage,df_cluster_3.mileage) # ANOVA
mean_test_anova_scipy

## Give the p-value as 3.952333929559486e-121 which converges to 0 which is less than 0.05, hence null hypothesis is rejected. So in atleast one of the clusters average mileage is not same. 


# In[163]:


# For each pairwise comparison, the p-value is very low (0.000), indicating that the observed differences 
# in means are statistically significant.

#The confidence intervals (Lower CI and Upper CI) do not include zero for any pairwise comparison, which further supports the conclusion that the differences in means are statistically significant.

#Therefore, it can be inferred that there are statistically significant differences in the mean price values between all pairs of clusters 
# Pairwise Comparision Using Scipy (Tukey HSD)
mean_pairwise_compare_scipy = sps.tukey_hsd(df_cluster_0.mileage,df_cluster_1.mileage,df_cluster_2.mileage,df_cluster_3.mileage)
print(mean_pairwise_compare_scipy)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting for Cluster 0
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='mileage', hue='fuelType_enc', data=df_cluster_0, palette='Set1')
plt.title('Cluster 0')
plt.xlabel('Price')
plt.ylabel('Mileage')
plt.legend(title='fuelType_enc', loc='upper right')
plt.show()


# In[165]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting for Cluster 0
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='mileage', hue='cluster_number', data=df_ppd_subset_kmcluster, palette='Set1')
plt.title('Various Clusters')
plt.xlabel('Price')
plt.ylabel('Mileage')
plt.legend(title='cluster_number', loc='upper right')
plt.show()


# In[166]:


df_cluster_1 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 1]
# Plotting for Cluster 1
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='tax', hue='cluster_number', data=df_cluster_1, palette='Set1')
plt.title('Cluster 1')
plt.xlabel('Price')
plt.ylabel('Mileage')
plt.legend(title='Manufacturer_enc', loc='upper right')
plt.show()


# In[167]:


df_cluster_2 = df_ppd_subset_kmcluster[df_ppd_subset_kmcluster['cluster_number'] == 2]
# Plotting for Cluster 2
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='mileage', hue='Manufacturer_enc', data=df_cluster_2, palette='Set1')
plt.title('Cluster 2')
plt.xlabel('Price')
plt.ylabel('Mileage')
plt.legend(title='Manufacturer_enc', loc='upper right')
plt.show()


# In[168]:


from scipy.stats import chi2_contingency

# Define categorical variables
categorical_variables = ['model_enc', 'transmission_enc', 'fuelType_enc', 'Manufacturer_enc']

# Segregate DataFrame based on cluster number
data_frames = [df_cluster_0, df_cluster_1, df_cluster_2, df_cluster_3]

# Perform chi-square test of independence for each cluster
for idx, df_cluster in enumerate(data_frames):
    print(f"Cluster {idx}:")
    for cat_var in categorical_variables:
        contingency_table = pd.crosstab(df_cluster[cat_var], df_cluster['cluster_number'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"\nChi-square test for {cat_var}:")
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p_value}")
        print(f"Degrees of freedom: {dof}")
        print("Expected frequencies:")
        print(expected)


    


# In[169]:


import pandas as pd
from scipy.stats import f_oneway

# Assuming df_cluster_0, df_cluster_1, df_cluster_2, and df_cluster_3 are the DataFrames for each cluster

# Step 1: Calculate Mean Price for Each Cluster
mean_price_cluster_0 = df_cluster_0['price'].mean()
mean_price_cluster_1 = df_cluster_1['price'].mean()
mean_price_cluster_2 = df_cluster_2['price'].mean()
mean_price_cluster_3 = df_cluster_3['price'].mean()

# Step 2: Perform Statistical Tests (ANOVA)
price_cluster_0 = df_cluster_0['price']
price_cluster_1 = df_cluster_1['price']
price_cluster_2 = df_cluster_2['price']
price_cluster_3 = df_cluster_3['price']

f_statistic, p_value = f_oneway(price_cluster_0, price_cluster_1, price_cluster_2, price_cluster_3)

# Step 3: Visualize Price Distribution (Optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(price_cluster_0, alpha=0.5, label='Cluster 0')
plt.hist(price_cluster_1, alpha=0.5, label='Cluster 1')
plt.hist(price_cluster_2, alpha=0.5, label='Cluster 2')
plt.hist(price_cluster_3, alpha=0.5, label='Cluster 3')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution Across Clusters')
plt.legend()
plt.show()

# Step 4: Interpret Results
if p_value < 0.05:
    print("There are significant differences in mean prices between clusters.")
    # Further analysis to identify clusters with distinct pricing characteristics
    # For example, pairwise t-tests to compare mean prices between clusters
else:
    print("There are no significant differences in mean prices between clusters.")


# In[176]:


import pandas as pd
from scipy.stats import f_oneway

# Assuming df_cluster_0, df_cluster_1, df_cluster_2, and df_cluster_3 are the DataFrames for each cluster

# Step 1: Calculate Mean Price for Each Cluster
mean_price_cluster_0 = df_cluster_0['mileage'].mean()
mean_price_cluster_1 = df_cluster_1['mileage'].mean()
mean_price_cluster_2 = df_cluster_2['mileage'].mean()
mean_price_cluster_3 = df_cluster_3['mileage'].mean()

# Step 2: Perform Statistical Tests (ANOVA)
price_cluster_0 = df_cluster_0['mileage']
price_cluster_1 = df_cluster_1['mileage']
price_cluster_2 = df_cluster_2['mileage']
price_cluster_3 = df_cluster_3['mileage']

f_statistic, p_value = f_oneway(price_cluster_0, price_cluster_1, price_cluster_2, price_cluster_3)

# Step 3: Visualize Price Distribution (Optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(price_cluster_0, alpha=0.5, label='Cluster 0')
plt.hist(price_cluster_1, alpha=0.5, label='Cluster 1')
plt.hist(price_cluster_2, alpha=0.5, label='Cluster 2')
plt.hist(price_cluster_3, alpha=0.5, label='Cluster 3')
plt.xlabel('mileage')
plt.ylabel('Frequency')
plt.title('Mileage Distribution Across Clusters')
plt.legend()
plt.show()

# Step 4: Interpret Results
if p_value < 0.05:
    print("There are significant differences in mean prices between clusters.")
    # Further analysis to identify clusters with distinct pricing characteristics
    # For example, pairwise t-tests to compare mean prices between clusters
else:
    print("There are no significant differences in mean prices between clusters.")


# In[175]:


import pandas as pd
from scipy.stats import f_oneway

# Assuming df_cluster_0, df_cluster_1, df_cluster_2, and df_cluster_3 are the DataFrames for each cluster

# Step 1: Calculate Mean Price for Each Cluster
mean_price_cluster_0 = df_cluster_0['tax'].mean()
mean_price_cluster_1 = df_cluster_1['tax'].mean()
mean_price_cluster_2 = df_cluster_2['tax'].mean()
mean_price_cluster_3 = df_cluster_3['tax'].mean()

# Step 2: Perform Statistical Tests (ANOVA)
price_cluster_0 = df_cluster_0['tax']
price_cluster_1 = df_cluster_1['tax']
price_cluster_2 = df_cluster_2['tax']
price_cluster_3 = df_cluster_3['tax']

f_statistic, p_value = f_oneway(price_cluster_0, price_cluster_1, price_cluster_2, price_cluster_3)

# Step 3: Visualize Price Distribution (Optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(price_cluster_0, alpha=0.5, label='Cluster 0')
plt.hist(price_cluster_1, alpha=0.5, label='Cluster 1')
plt.hist(price_cluster_2, alpha=0.5, label='Cluster 2')
plt.hist(price_cluster_3, alpha=0.5, label='Cluster 3')
plt.xlabel('Tax')
plt.ylabel('Frequency')
plt.title('Tax Distribution Across Clusters')
plt.legend()
plt.show()

# Step 4: Interpret Results
if p_value < 0.05:
    print("There are significant differences in mean prices between clusters.")
    # Further analysis to identify clusters with distinct pricing characteristics
    # For example, pairwise t-tests to compare mean prices between clusters
else:
    print("There are no significant differences in mean prices between clusters.")


# In[174]:


import pandas as pd
from scipy.stats import f_oneway

# Assuming df_cluster_0, df_cluster_1, df_cluster_2, and df_cluster_3 are the DataFrames for each cluster

# Step 1: Calculate Mean Price for Each Cluster
mean_price_cluster_0 = df_cluster_0['engineSize'].mean()
mean_price_cluster_1 = df_cluster_1['engineSize'].mean()
mean_price_cluster_2 = df_cluster_2['engineSize'].mean()
mean_price_cluster_3 = df_cluster_3['engineSize'].mean()

# Step 2: Perform Statistical Tests (ANOVA)
price_cluster_0 = df_cluster_0['engineSize']
price_cluster_1 = df_cluster_1['engineSize']
price_cluster_2 = df_cluster_2['engineSize']
price_cluster_3 = df_cluster_3['engineSize']

f_statistic, p_value = f_oneway(price_cluster_0, price_cluster_1, price_cluster_2, price_cluster_3)

# Step 3: Visualize Price Distribution (Optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(price_cluster_0, alpha=0.5, label='Cluster 0')
plt.hist(price_cluster_1, alpha=0.5, label='Cluster 1')
plt.hist(price_cluster_2, alpha=0.5, label='Cluster 2')
plt.hist(price_cluster_3, alpha=0.5, label='Cluster 3')
plt.xlabel('EngineSize')
plt.ylabel('Frequency')
plt.title('EngineSize Distribution Across Clusters')
plt.legend()
plt.show()

# Step 4: Interpret Results
if p_value < 0.05:
    print("There are significant differences in mean prices between clusters.")
    # Further analysis to identify clusters with distinct pricing characteristics
    # For example, pairwise t-tests to compare mean prices between clusters
else:
    print("There are no significant differences in mean prices between clusters.")


# In[ ]:




