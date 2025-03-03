# Credit Card Fraud Detection

![Credit Card Fraud Detection](https://img.shields.io/badge/Finance-Credit_Card_Fraud_Detection-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Overview
A machine learning project that predicts credit card fraud using supervised learning techniques. This project analyzes transaction patterns and customer demographics to identify potentially fraudulent credit card activities with high accuracy.

## 🔍 Key Findings

### Data Insights
- 🕙 **Temporal Patterns**: Higher rates of fraudulent transactions occur between 10 PM and 3 AM
- 💰 **Transaction Characteristics**: Fraudulent transactions typically involve higher monetary values
- 🛒 **Category Analysis**: Shopping and grocery categories show elevated fraud rates
- ⏱️ **Transaction Timing**: Short intervals between consecutive transactions serve as strong fraud indicators
- 👴 **Demographic Trends**: The 60+ age group demonstrates increased vulnerability to fraud compared to younger demographics

### Model Performance
- Implemented SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance
- 🏆 Gradient Boosting emerged as the top-performing model with:
  - AUC > 95%
  - High overall accuracy
  - Superior precision in fraud detection

![download](https://github.com/user-attachments/assets/fbbdac07-a0f5-4260-9463-81cfcd02cf20)

All evaluated models achieved ROC curves and accuracy ratings exceeding 90%. This indicates strong overall performance in differentiating between legitimate and fraudulent transactions. However, given the inherent class imbalance in fraud detection, precision metrics provide crucial additional insight. Gradient Boosting demonstrated the highest precision (92%), meaning that of all transactions flagged as fraudulent, 92% were correctly classified. This makes Gradient Boosting the preferred model for deployment.

## 🛠️ Project Steps

### Data Preprocessing
I started by exploring and cleaning the initial dataset:

```python
# Check initial data shape and information
print("Initial data shape:", data.shape)
print(data.info())

# Drop irrelevant features
data_drop_features = data.drop(columns=['Unnamed: 0.1','Unnamed: 0','first','last','street','city','zip','trans_num','unix_time','state','city_pop'])
print("Data shape after dropping columns:", data_drop_features.shape)
```

The dataset had no missing values, which was ideal. I converted datetime fields to the proper format:

```python
# Convert datetime transaction and dob to correct format
data_drop_features['trans_date_trans_time'] = pd.to_datetime(data_drop_features['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
data_drop_features['dob'] = pd.to_datetime(data_drop_features['dob'], format='%Y-%m-%d')
```

I checked class distribution and found that fraud transactions were only about 8% of the data, which then i need to work with the sampling later in the model. 
Fraud is uncommom which makes it hard to spot. Therefore, we will need to have more fraud in the data, but instead of oversampling by duplicating the fraud transactions to make it prominent I will use SMOTE (Synthetic Minority Oversampling Technique) to create synthetic scenerio from the neighest neighbors, this help avoid overfitting.

```python
print(data_drop_features['is_fraud'].value_counts(normalize=True))
# is_fraud
# 0    0.923211
# 1    0.076789
```

### Feature Engineering

#### ⏰ Time Features
I extracted hour and day of week from transaction timestamps:

```python
data_drop_features['hour'] = data_drop_features['trans_date_trans_time'].dt.hour
data_drop_features['day_of_week'] = data_drop_features['trans_date_trans_time'].dt.dayofweek
```

Visualization showed higher fraud rates during night hours:

![Hour Fraud Rate](https://github.com/user-attachments/assets/619c84af-9335-4921-bc0f-209f82a24487)

I created a night transaction flag based on this insight:

```python
data_drop_features['night_trans'] = data_drop_features['hour'].apply(lambda x: 1 if 0 <= x <= 3 else 0)
```

I also analyzed the time difference between consecutive transactions, which proved to be a strong indicator:

```python
data_drop_features = data_drop_features.sort_values(by=['cc_num', 'trans_date_trans_time'])
data_drop_features['time_diff'] = data_drop_features.groupby('cc_num')['trans_date_trans_time'].diff()
data_drop_features['time_diff'] = data_drop_features['time_diff'].fillna(pd.Timedelta(seconds=0))
```

I categorized these time differences into meaningful groups:

![download](https://github.com/user-attachments/assets/c70f6aa4-d7ea-4d50-b966-9d9f777ad7a3)


Transactions occurring 5-60 minutes apart showed significantly higher fraud rates.

#### 💵 Monetary Features
I analyzed transaction amounts and found that higher value transactions had higher fraud rates:

![download](https://github.com/user-attachments/assets/a3932b80-2d87-4413-b043-023426aea5bb)


#### 👤 Age Features
I calculated customer age and grouped them to analyze fraud patterns:

```python
data_drop_features['age'] = (defined_date - data_drop_features['dob']).dt.days // 365
bins = [0, 18, 30, 45, 60, 100]
labels = ['Under 18', '18-29', '30-44', '45-59', '60+']
data_drop_features['age_group'] = pd.cut(data_drop_features['age'], bins=bins, labels=labels, right=False)
```

![download](https://github.com/user-attachments/assets/b35fc799-36bf-4e59-93f5-47fd16a63d48)


The 60+ age group showed higher vulnerability to fraud.

#### 🛍️ Category Features
Analysis of transaction categories revealed shopping and grocery transactions had higher fraud rates:

![download](https://github.com/user-attachments/assets/53daf378-78c0-40cf-95f7-e7822c38e0e6)


#### 📍 Distance Feature
I calculated the distance between customer and merchant locations:

```python
def calculate_geopy_distance(row):
    card_location = (row['lat'], row['long'])
    merchant_location = (row['merch_lat'], row['merch_long'])
    return geodesic(card_location, merchant_location).kilometers

data_drop_features['distance_km'] = data_drop_features.apply(calculate_geopy_distance, axis=1)
```
![Distance Fraud Rate](https://github.com/user-attachments/assets/d02c2dba-c441-4f11-9c4b-a73320bc1e40)

#### 🏬 Merchant and Job Risk Features
I grouped merchants and jobs based on their historical fraud rates:

```python
# Grouping Merchant
merchant_group = data_drop_features.groupby('merchant')['is_fraud'].mean().reset_index()
merchant_group['merchant_risk'] = pd.qcut(merchant_group['is_fraud'], q=3, labels=['low-risk', 'medium-risk', 'high-risk'])

# Grouping Job
job_fraud = data_drop_features.groupby('job')['is_fraud'].mean().reset_index()
job_fraud['job_risk'] = pd.qcut(job_fraud['is_fraud'], q=3, labels=['low-risk', 'medium-risk', 'high-risk'])
```

I created a correlation heatmap to visualize relationships between features:

![download](https://github.com/user-attachments/assets/3397a607-1f58-4c9e-95a4-91d30d954ee2)


### Model Building

I developed a comprehensive modeling pipeline:

```python
def build_and_evaluate_models(data_final):
    """Build and evaluate multiple models for fraud detection"""
    # Separate features and target
    X = data_final.drop('is_fraud', axis=1)
    y = data_final['is_fraud']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Define categorical and numerical features
    categorical_features = [col for col in data_final.columns if data_final[col].dtype in ['object','category']] #transforming
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
```

## 🧠 Key Technical Implementations

- **Feature Engineering Pipeline**: Created domain-specific features from raw transaction data
- **Class Imbalance Handling**: Applied SMOTE to create synthetic fraud examples
- **Model Optimization**: Fine-tuned parameters for imbalanced classification
- **Evaluation Strategy**: Emphasized precision to minimize false positives
- **Preprocessing Pipeline**: Streamlined data transformation for production readiness

## 📈 Conclusions

The project successfully demonstrates that machine learning can effectively detect credit card fraud by identifying subtle patterns in transaction data. Key insights include:

1. 🌙 Night transactions (10PM-3AM) require elevated scrutiny
2. 💸 High-value transactions should trigger additional verification
3. ⏲️ Transactions occurring within short time intervals (5-60 minutes) show the highest fraud risk
4. 👵 Special protection measures may benefit the 60+ age demographic
5. 🛍️ Shopping and grocery categories demonstrate higher fraud vulnerability

The Gradient Boosting model provides the optimal balance between overall accuracy and precision in fraud identification, making it suitable for real-world deployment.

## 🔮 Future Work
- Real-time fraud detection implementation
- Additional feature engineering based on transaction sequences
- Exploration of deep learning approaches for improved detection

## 📚 Data Source
The dataset was provided by UniGap.
