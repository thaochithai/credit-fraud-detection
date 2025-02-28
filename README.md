# Creadit Card Fraud Detection
ML Project that predict credit card fraud - Supervised Leanring

### Results and Insights
Key highlights:
Data insights:
- There are more fraudulent transations from 10PM to 3AM
- Fraudulent transactions are usually high value and from shopping and grocery category
- The time between two transactions can be a good indicator
- 60+ age group is more likely to have fraud transactions than younger age group

Models results:
- Use SMOTE techniques to handle imbalanced class 
- Gradient Boosting is the best models out of three choosen, with AUC > 95%, high overall accuracy, and high precision
  
![download](https://github.com/user-attachments/assets/0fc5abc3-c67f-40af-a057-2436bd151b4d)
All the model have ROC and accuracy of more than 90%. Which mean overall, the models have high probability to identify postive class (non-fraud) over the negative class (fraud) with high overall model prediction. However, with the imbalanced data, we should look closer into Precision, Gradient Boosting has highest percision which mean out of total prediction, 92% are correctly classified as true postive (non-fraud). Therefore, Gradient Boosting is the best model.

---
#Project Details

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
> It can be seen from the graph that those transactions that are close to each other only account for small percentage of transactions but has high fraud rate especially 5-15 min, 15-30 min and 30-60 mins has fraud rate more than 60%

2. Monetary features
The amount of the transactions is also a good indicator of the fraud because usually the amount is high. I categorize the amount into different bin to visualize.
~~~
print(data_drop_features['amt'].describe())
count    97748.000000
mean       102.832444
std        217.047969
min          1.000000
25%         10.567500
50%         50.160000
75%         91.782500
max      15047.030000

plt.figure(figsize=(12, 6))
data_drop_features['amount_bin'] = pd.cut(
    data_drop_features['amt'],
    bins=[0, 25, 100, 500, 1000, float('inf')],
    labels=['$0-$25', '$25-$100', '$100-$500', '$500-$1000', '$1000+']
)
amount_fraud = data_drop_features.groupby('amount_bin')['is_fraud'].mean() * 100
sns.barplot(x=amount_fraud.index, y=amount_fraud.values,color='#FFB703')
plt.title('Fraud Rate by Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Fraud Rate (%)')
~~~
![download](https://github.com/user-attachments/assets/641483fe-73f1-4d3d-8557-99073dcbc4ab)
> High value transactions are more prone to be fraudulent transactions
3. Age Features
~~~
data_drop_features['dob'] = pd.to_datetime(data_drop_features['dob'])
defined_date = pd.to_datetime('2019-12-31')
data_drop_features['age'] = (defined_date - data_drop_features['dob']).dt.days // 365

bins = [0, 18, 30, 45, 60, 100]
labels = ['Under 18', '18-29', '30-44', '45-59', '60+']
data_drop_features['age_group'] = pd.cut(data_drop_features['age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(12, 6))
age_fraud = data_drop_features.groupby('age_group')['is_fraud'].mean() * 100
sns.barplot(x=age_fraud.index, y=age_fraud.values,color='#FFB703')
plt.title('Fraud Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Fraud Rate (%)')
~~~
![download](https://github.com/user-attachments/assets/177287c3-51ac-4024-861c-2179721e1223)
> Older age group are more prone to be fraud than the younger age groups

3. Category Features
I also visualize the fraud rate in relation to the catgory of the transactions
~~~
# Calculate fraudulent transaction counts per category
category_fraud_count = data_drop_features[data_drop_features['is_fraud'] == 1]['category'].value_counts()
category_total_count = data_drop_features['category'].value_counts()
categories = sorted(set(category_fraud_count.index) | set(category_total_count.index))

fraud_counts = [category_fraud_count.get(cat, 0) for cat in categories]
total_counts = [category_total_count.get(cat, 0) for cat in categories]
bar_width = 0.35
x = np.arange(len(categories))

# Create the bar chart
plt.figure(figsize=(14, 7))
bar1 = plt.bar(x - bar_width/2, fraud_counts, bar_width, label='Fraudulent Transactions', color='#FFB703')
bar2 = plt.bar(x + bar_width/2, total_counts, bar_width, label='Total Transactions', color='#FB8500')

# Set labels and title
plt.xlabel('Category')
plt.ylabel('Number of Transactions')
plt.title('Fraudulent vs Total Transactions by Category')
plt.xticks(x, categories, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
~~~
![download](https://github.com/user-attachments/assets/8a2fa0db-5f10-474e-adc1-a8f72f5d9fc0)
> Grocery_pos and shopping_net and shopping_pos are likely to be fraudulent than other category

4. Distance Feature
I used the longtitude and latitude of the card holder and longtitude and latitude of the merchance to calculate the distance and visualize it.
To calculate the distance, I use geopy library to define a function
~~~
def calculate_geopy_distance(row):
    card_location = (row['lat'], row['long'])
    merchant_location = (row['merch_lat'], row['merch_long'])
    return geodesic(card_location, merchant_location).kilometers
data_drop_features['distance_km'] = data_drop_features.apply(calculate_geopy_distance, axis=1)
print(data_drop_features['distance_km'].describe())
count    97748.000000
mean        76.146272
std         29.014056
min          0.469721
25%         55.388372
50%         78.351432
75%         98.297578
max        146.541139
Name: distance_km, dtype: float64
~~~
Then I visualize it
~~~
plt.figure(figsize=(12, 6))
data_drop_features['distance_bin'] = pd.cut(
    data_drop_features['distance_km'],
    bins=[0, 5, 20, 50, 100, float('inf')],
    labels=['0-5 km', '5-20 km', '20-50 km', '50-100 km', '100+ km']
)
distance_fraud = data_drop_features.groupby('distance_bin')['is_fraud'].mean() * 100
sns.barplot(x=distance_fraud.index, y=distance_fraud.values,color='#FFB703')
plt.title('Fraud Rate by Distance')
plt.xlabel('Distance Between Customer and Merchant')
plt.ylabel('Fraud Rate (%)')
~~~
![download](https://github.com/user-attachments/assets/d02c2dba-c441-4f11-9c4b-a73320bc1e40)
> The longer the distance between the card holder and the merchant the more fraud but also it is not so much different and we have to consider the type of transaction it is online or offline. For training the model i will not group it but use the distance only.

5. Merchant and Job Features
Since there are so many merchant and job so I group them based on the fraud rate into 'low-risk', 'medium-risk', 'high-risk'
~~~
#Grouping Merchant
merchant_group = data_drop_features.groupby('merchant')['is_fraud'].mean().reset_index()
merchant_group['merchant_risk'] = pd.qcut(merchant_group['is_fraud'], q=3, labels=['low-risk', 'medium-risk', 'high-risk'])
data_drop_features = data_drop_features.merge(merchant_group[['merchant', 'merchant_risk']], on='merchant', how='left')

#Grouping Job
job_fraud = data_drop_features.groupby('job')['is_fraud'].mean().reset_index()
job_fraud['job_risk'] = pd.qcut(job_fraud['is_fraud'], q=3, labels=['low-risk', 'medium-risk', 'high-risk'])
data_drop_features = data_drop_features.merge(job_fraud[['job', 'job_risk']], on='job', how='left')
~~~
Finally I create a correlation map of all numeric features and also the catergory one that is not group by fraud rate
~~~
# Correlation Heatmap
plt.figure(figsize=(14, 10))
numeric_cols = ['amt','time_diff_minutes','age', 'is_fraud','distance_km']

category_encoded = pd.get_dummies(data_drop_features[['category','night_trans']], drop_first=True) # Pass a list of column names
correlation_data = pd.concat([data_drop_features[numeric_cols], category_encoded], axis=1)
correlation = correlation_data.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='Oranges')
plt.title('Correlation Heatmap')
plt.tight_layout()
~~~
![download](https://github.com/user-attachments/assets/ff5b182c-b5b2-402f-9e20-7908ab316423)

> is_fraud is highly correlated with amount, moderate correlation with night_trans, category_shopping_net and category_shopping_net which is also spotted out before. Night transactions are usually made with these category and gas_transport. High transaction are highly correlated with shopping category. Distance have no correlation with other variable, and from the descriptive graph above, distance should not be included to train model. Time_different_minutes negatively correlated with fraud, which also suggest that the less time in between transaction, the more likely it is fraud.

#### Final Features Set
- Categorical: time_diff_group, merchant_risk, job_risk, age_group, night_trans
- Numercial: amt
~~~
data_final = data_drop_features[['category','amt','time_diff_group','merchant_risk','is_fraud','job_risk',
                                 'age_group','night_trans']]
~~~
## Building Models
I build the pipeline of spliting data into training and testing set (70%-30%), and transforming the numerical and categorical
- Numerical > use StandardScaler() to rescale data to have mean of 0 and standard deviatopm of 1 unit
- Categorical > use OneHotEncoder to create dummies

As I mentioned below because fraud only amount for small percentage, I use SMOTE method to create synthetic sample to make fraud more prominent

I chose 3 models, Linear Regression as a baseline model, tree-model is Random Forest and boosting model Gradient Boosting

~~~
def build_and_evaluate_models(data_final):
    """Build and evaluate multiple models for fraud detection"""
    # Separate features and target
    X = data_final.drop('is_fraud', axis=1)
    y = data_final['is_fraud']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Define categorical and numerical features
    categorical_features = ['category', 'job_risk', 'age_group','time_diff_group','merchant_risk']
    numerical_features = X.columns.drop(categorical_features).tolist()

    # Initialize preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
         remainder='passthrough'
    )

    # Initialize preprocessing and models
    smote = SMOTE(random_state=42, sampling_strategy=0.3)

    # Adjusted parameters for imbalanced data
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            min_samples_leaf=5,
            max_depth=10
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            subsample=0.8,
            max_depth=5,
            min_samples_leaf=5,
            learning_rate=0.1
        )
    }

    # Store predictions and probabilities
    predictions = {}
    probabilities = {}

    # Train models and make predictions
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor), # Apply column transformer first
            ('smote', smote), # Then apply SMOTE
            ('classifier', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Store predictions and probabilities
        predictions[name] = pipeline.predict(X_test)
        probabilities[name] = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics for each model
    metrics = []
    for model_name, y_pred in predictions.items():
        metrics.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, probabilities[model_name])
        })

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Performance Metrics\n")
    print(metrics_df)

    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(metrics_df.set_index('Model'),
                annot=True,
                fmt=".3f",
                cmap="Oranges",
                cbar=True,
                linewidths=0.5)
    plt.title('Model Performance Metrics')
    plt.show()

    # Find and print best model
    best_model = metrics_df.loc[metrics_df['ROC-AUC'].idxmax(), 'Model']
    print(f"\nBest performing model based on ROC-AUC: {best_model}")

    return metrics_df
~~~
### Data Source
Provided by UniGap
