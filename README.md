# Creadit Card Fraud Detection
ML Project that predict credit card fraud - Supervised Leanring

## Data Wraggling
The dataset has the fraud/non-fraud classification, thus we use supervised learning model

### Pre-processing 
1. Initial Variable Selection
I go through all the features in the data to see which one should be included and which one should be dropped to simplify dataset and avoid irrelevant features
~~~
print("Initial data shape:", data.shape)
print(data.info())
~~~
I dropped all the irrelevant or giving no values as order, name, street, city, state, population or transaction numbers because this give no information because for location and distance we already have the long_titude and latitude: 'Unnamed: 0.1','Unnamed: 0','first','last','street','city','zip','trans_num','unix_time','state','city_pop'
~~~
data_drop_features = data.drop(columns=['Unnamed: 0.1','Unnamed: 0','first','last','street','city','zip','trans_num','unix_time','state','city_pop'])
print("Data shape after dropping columns:", data_drop_features.shape)
~~~
Then now I have 14 features to work with and next I will work with cleaning data from this new dataset

2. Data Cleaning
First I check the data characteristic, and check if there is missing values, outliers, and see the statistics of numerical variables and the unique value of categorical variables

~~~
#Check data info
print(data_drop_features.info())
print(data_drop_features.describe())
print("Missing values:\n", data_drop_features.isnull().sum())
print("Unique values per column:\n", data_drop_features.nunique())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 97748 entries, 0 to 97747
Data columns (total 13 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   trans_date_trans_time  97748 non-null  object 
 1   cc_num                 97748 non-null  int64  
 2   merchant               97748 non-null  object 
 3   category               97748 non-null  object 
 4   amt                    97748 non-null  float64
 5   gender                 97748 non-null  object 
 6   lat                    97748 non-null  float64
 7   long                   97748 non-null  float64
 8   job                    97748 non-null  object 
 9   dob                    97748 non-null  object 
 10  merch_lat              97748 non-null  float64
 11  merch_long             97748 non-null  float64
 12  is_fraud               97748 non-null  int64  
dtypes: float64(5), int64(2), object(6)
memory usage: 9.7+ MB
None
             cc_num           amt           lat          long     merch_lat  \
count  9.774800e+04  97748.000000  97748.000000  97748.000000  97748.000000   
mean   4.103967e+17    102.832444     38.525509    -90.210550     38.524928   
std    1.298122e+18    217.047969      5.079548     13.799118      5.114803   
min    6.041621e+10      1.000000     20.027100   -165.672300     19.031242   
25%    1.800429e+14     10.567500     34.668900    -96.790900     34.749667   
50%    3.521815e+15     50.160000     39.342600    -87.458100     39.349543   
75%    4.642255e+15     91.782500     41.894800    -80.128400     41.944618   
max    4.992346e+18  15047.030000     66.693300    -67.950300     67.510267   

         merch_long      is_fraud  
count  97748.000000  97748.000000  
mean     -90.210050      0.076789  
std       13.813079      0.266258  
min     -166.654993      0.000000  
25%      -96.864098      0.000000  
50%      -87.391641      0.000000  
75%      -80.207857      0.000000  
max      -66.980744      1.000000  
Missing values:
 trans_date_trans_time    0
cc_num                   0
merchant                 0
category                 0
amt                      0
gender                   0
lat                      0
long                     0
job                      0
dob                      0
merch_lat                0
merch_long               0
is_fraud                 0
dtype: int64
Unique values per column:
 trans_date_trans_time    97609
cc_num                     983
merchant                   693
category                    14
amt                      25048
gender                       2
lat                        968
long                       969
job                        494
dob                        968
merch_lat                97438
merch_long               97627
is_fraud                     2
dtype: int64
~~~
This is ideal dataset as there is no missing value.
The data type is not correct for time variables as trans_date_trans_time and dob need to be transformed into datetime64[ns].
~~~
#Convert datetime transaction and dob is not right format
data_drop_features['trans_date_trans_time'] = pd.to_datetime(data_drop_features['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
data_drop_features['dob'] = pd.to_datetime(data_drop_features['dob'], format='%Y-%m-%d')
print("Data types after datetime conversion:\n", data_drop_features.dtypes)
~~~
The amount of money have sound distribution with no negative amount.
The category variable merchant and job are numerous which need to be transformed and grouped because there are more than 400 unique values.


2. Data Exploration & Features Engineering
I then check the class distrubution and the characteristics of data to see potential relationships

#### Class Distribution
Check the the class imbalance, fraud is ~8% of the total database. Fraud is uncommom which makes it hard to spot. Therefore, we will need to have more fraud in the data, but instead of oversampling by duplicating the fraud transactions to make it prominent I will use SMOTE (Synthetic Minority Oversampling Technique) to create synthetic scenerio from the neighest neighbors, this help avoid overfitting.
~~~
print(data_drop_features['is_fraud'].value_counts(normalize=True))
is_fraud
0    0.923211
1    0.076789
Name: count, dtype: float64
~~~
#### Fearures Engineering
1. Time feautures
Time features are important in fraud detection because it shows specific patterns in making transactions such as odds hours, and timing paterns.
However, since time is cylincal so we need to transform time into meaningfull features, in this dataset, we have the transaction date and hours which we can derive features from. I start by seeing the pattern of fraud rate depending on hours and day of weeks of the transactions.
~~~
# Derive time into hour and weeks from transaction time
data_drop_features['hour'] = data_drop_features['trans_date_trans_time'].dt.hour
data_drop_features['day_of_week'] = data_drop_features['trans_date_trans_time'].dt.dayofweek
~~~
Hour and Day of the Week
~~~
#Visualize fraud by hour
plt.figure(figsize=(12, 6))
hour_fraud = data_drop_features.groupby('hour')['is_fraud'].mean() * 100
sns.lineplot(x=hour_fraud.index, y=hour_fraud.values, marker='o',color='#FB8500')
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate (%)')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
~~~
![download](https://github.com/user-attachments/assets/619c84af-9335-4921-bc0f-209f82a24487)
> It is clear that night transactions are from 22PM to 3AM have high fraud rates. Therefore, I will create a night transaction features.
~~~
data_drop_features['night_trans'] = data_drop_features['hour'].apply(lambda x: 1 if 0 <= x <= 3 else 0) #night transactions have high fraud rate
~~~
Then I visualize the day of the week
~~~
#Visualize fraud by day of week
plt.figure(figsize=(12, 6))
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_fraud = data_drop_features.groupby('day_of_week')['is_fraud'].mean() * 100
sns.barplot(x=days, y=day_fraud.values,  color='#FFB703')
plt.title('Fraud Rate by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Fraud Rate (%)')
~~~
![download](https://github.com/user-attachments/assets/9ad76cd6-cde1-45a7-b350-6a27c56ddb75)
> There is not a acute difference between days of week, but high fraud rate are usually between Wed and Fri
Time difference between transactions
Because the first transaction have nothing to compare to so it will be filled == 0
~~~
data_drop_features = data_drop_features.sort_values(by=['cc_num', 'trans_date_trans_time'])
data_drop_features['time_diff'] = data_drop_features.groupby('cc_num')['trans_date_trans_time'].diff()
data_drop_features['time_diff'] = data_drop_features['time_diff'].fillna(pd.Timedelta(seconds=0))
print(data_drop_features['time_diff'].describe())
count                        97748
mean     4 days 19:52:36.978690101
std      7 days 00:17:08.051386260
min                0 days 00:00:00
25%                0 days 19:24:10
50%                2 days 13:14:47
75%                6 days 01:38:30
max              156 days 10:01:28
Name: time_diff, dtype: object
~~~
Based on the distributtion I categorized them into group, fraud can happened closed to each other, and we can see that 75% of the transactions are around 7 days, which make sense to break more groups for transactions under 1 days into granularity as minutes and hours.
~~~
time_bins = [
    0,                  # 0 minutes
    5,                  # 0-5 minutes
    15,                 # 5-15 minutes
    30,                 # 15-30 minutes
    60,                 # 30-60 minutes (1 hour)
    360,                # 1-6 hours
    720,                # 6-12 hours
    1440,               # 12-24 hours (1 day)
    4320,               # 1-3 days
    10080,              # 3-7 days
    float('inf')        # > 7 days
]
~~~
Then I make the visualization to see the fraud rate accordingly
~~~
time_labels = [
    '0-5 min',
    '5-15 min',
    '15-30 min',
    '30-60 min',
    '1-6 hrs',
    '6-12 hrs',
    '12-24 hrs',
    '1-3 days',
    '3-7 days',
    '> 7 days'
]

# Create the bins
data_drop_features['time_diff_group'] = pd.cut(
    data_drop_features['time_diff_minutes'],
    bins=time_bins,
    labels=time_labels,
    right=False
)

time_fraud = data_drop_features.groupby('time_diff_group')['is_fraud'].agg(['mean', 'count']).reset_index()
time_fraud['fraud_rate'] = time_fraud['mean'] * 100  # Convert to percentage
time_fraud['count_pct'] = time_fraud['count'] / time_fraud['count'].sum() * 100  # Percentage of transactions

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot fraud rate as bars
bars = ax1.bar(time_fraud['time_diff_group'], time_fraud['fraud_rate'], color='#FFB703')
ax1.set_xlabel('Time Since Previous Transaction')
ax1.set_ylabel('Fraud Rate (%)')
ax1.tick_params(axis='y')
ax1.set_xticklabels(time_fraud['time_diff_group'], rotation=45, ha='right')

# Add a second y-axis for transaction counts
ax2 = ax1.twinx()
line = ax2.plot(time_fraud['time_diff_group'], time_fraud['count_pct'], 'o-', color='#FB8500', linewidth=2)
ax2.set_ylabel('Percentage of Transactions (%)')
ax2.tick_params(axis='y')
~~~
![download](https://github.com/user-attachments/assets/766e1a50-b76d-450e-a308-c68c5ce42b04)


