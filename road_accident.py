import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")
df = pd.read_csv(r'D:\road_accident_project\US_Accidents_March23.csv\US_Accidents_March23.csv')

print("Dataset loaded!")
print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())
print(df.dtypes)
print(df.isnull().sum())
print("\n--- Severity Count ---")
print(df['Severity'].value_counts())

print("\n--- Top 10 States with Most Accidents ---")
print(df['State'].value_counts().head(10))

print("\n--- Weather Conditions ---")
print(df['Weather_Condition'].value_counts().head(10))
print("\n--- Before Cleaning ---")
print("Shape:", df.shape)

# Drop columns we don't need
df = df.drop(columns=['ID', 'Source', 'End_Lat', 'End_Lng', 
                       'Description', 'Zipcode', 'Country',
                       'Airport_Code', 'Weather_Timestamp',
                       'Turning_Loop'])

# Drop rows with missing values in important columns
df = df.dropna(subset=['City', 'State', 'Weather_Condition', 
                        'Sunrise_Sunset', 'Temperature(F)'])

print("\n--- After Cleaning ---")
print("Shape:", df.shape)
print("Missing values remaining:", df.isnull().sum().sum())
# Fill numeric columns with median
df['Temperature(F)'] = df['Temperature(F)'].fillna(df['Temperature(F)'].median())
df['Humidity(%)'] = df['Humidity(%)'].fillna(df['Humidity(%)'].median())
df['Visibility(mi)'] = df['Visibility(mi)'].fillna(df['Visibility(mi)'].median())
df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].median())
df['Pressure(in)'] = df['Pressure(in)'].fillna(df['Pressure(in)'].median())

# Fill categorical columns with mode
df['Wind_Direction'] = df['Wind_Direction'].fillna(df['Wind_Direction'].mode()[0])
df['Weather_Condition'] = df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0])

# Drop Wind_Chill and Precipitation (too many missing)
df = df.drop(columns=['Wind_Chill(F)', 'Precipitation(in)'])

print("Missing values remaining:", df.isnull().sum().sum())
print("Final Shape:", df.shape)
# Convert Start_Time to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
# Extract useful features
df['Hour'] = df['Start_Time'].dt.hour
df['Day_of_Week'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year
df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x < 6 or x >= 20 else 0)
df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

print("New columns added:", ['Hour', 'Day_of_Week', 'Month', 'Year', 'Is_Night', 'Is_Weekend'])
print("Final Shape:", df.shape)
print("\nSample:")
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
# ---- PHASE 2: EDA ----
plt.figure(figsize=(8,5))
df['Severity'].value_counts().sort_index().plot(kind='bar', color=['green','orange','red','darkred'])
plt.title('Accident Count by Severity')
plt.xlabel('Severity (1=Minor, 4=Fatal)')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('severity_chart.png')
plt.show()
print("Chart saved!")
plt.figure(figsize=(10,5))
df['Hour'].value_counts().sort_index().plot(kind='line', color='blue', linewidth=2, marker='o', markersize=4)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour (0 = Midnight, 12 = Noon)')
plt.ylabel('Number of Accidents')
plt.xticks(range(0,24))
plt.grid(True)
plt.tight_layout()
plt.savefig('hour_chart.png')
plt.show()
plt.figure(figsize=(10,5))
df['State'].value_counts().head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 States with Most Accidents')
plt.xlabel('State')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('states_chart.png')
plt.show()
plt.close()
plt.figure(figsize=(10,5))
df['State'].value_counts().head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 States with Most Accidents')
plt.xlabel('State')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('states_chart.png')
plt.show()
plt.figure(figsize=(10,5))
df['Weather_Condition'].value_counts().head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Weather Conditions During Accidents')
plt.xlabel('Number of Accidents')
plt.tight_layout()
plt.savefig('weather_chart.png')
plt.show()
plt.close()
plt.figure(figsize=(10,5))
days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['Day_of_Week'].value_counts().reindex(days_order).plot(kind='bar', color='purple')
plt.title('Accidents by Day of Week')
plt.xlabel('Day')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dayofweek_chart.png')
plt.show()
plt.close()
plt.figure(figsize=(10,5))
df['Year'].value_counts().sort_index().plot(kind='bar', color='darkorange')
plt.title('Accidents by Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('year_chart.png')
plt.show()
plt.close() 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Libraries imported for ML!")
# Select features and target
features = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
            'Wind_Speed(mph)', 'Pressure(in)', 'Hour', 
            'Month', 'Is_Night', 'Is_Weekend']

target = 'Severity'

df_ml = df[features + [target]].dropna()

print("ML Dataset Shape:", df_ml.shape)
print("\nFeatures:", features)
print("\nTarget:", target)
print("\nSeverity Distribution:")
print(df_ml[target].value_counts())
X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape) 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import pandas as pd

feature_importance = pd.Series(model.feature_importances_, index=features)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,5))
feature_importance.plot(kind='bar', color='steelblue')
plt.title('Feature Importance - What Affects Accident Severity?')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
plt.close()

print("\nFeature Importance:")
print(feature_importance)
import joblib

joblib.dump(model, 'accident_model.pkl')
print("Model saved as accident_model.pkl!")