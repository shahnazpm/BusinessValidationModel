# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# %%
#funtction to find null values and their percentage
def null_values(df):
    pd.set_option('display.max_rows', None )
    null_values=df.isnull().sum()/len(df)*100
    nan_values=df.isna().sum()/len(df)*100
    result = pd.DataFrame({'Null Values': df.isnull().sum(),'Null Values %': null_values,'NaN Values': df.isna().sum(),'NaN Values %': nan_values})
    return result

# %%
# Function to find Basic Stats like mode,median,mode,min_val,Max_val of the column in Dataframe and return results in set
def basic_stats(df,column):
    mean=df[column].mean()
    median=df[column].median()
    mode=df[column].mode()[0]
    min_val=df[column].min()
    max_val=df[column].max()
    stats = {'mean': mean,'median': median,'mode': mode,'min': min_val,'max': max_val}
    return stats

# %%
def find_outliers(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Calculate the first quartile (Q1) and third quartile (Q3) for each numeric column
    Q1 = numeric_df.quantile(0.25, axis=0)
    Q3 = numeric_df.quantile(0.75, axis=0)

    # Calculate the interquartile range (IQR) for each numeric column
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers for each numeric column
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers for each numeric column
    #outliers_df = numeric_df[(numeric_df < lower_bound) | (numeric_df > upper_bound)]
    outliers_df = numeric_df[(numeric_df < lower_bound) | (numeric_df > upper_bound)].dropna(how='all')

    return outliers_df


# %%
#Function to draw histogram
def histo_graph(df, column, ax=None):
    # Calculate statistics for the specified column
    mean_val = df[column].mean()
    median_val = df[column].median()
    mode_val = df[column].mode()[0]

    # If an axis is provided, plot on that axis; otherwise, create a new figure
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()  # Get the current axis

    # Plot histogram for the specified column
    ax.hist(df[column], bins=10, edgecolor='black', alpha=0.7)

    # Add mean, median, and mode lines
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1)
    ax.axvline(mode_val, color='black', linestyle='dashed', linewidth=1)

    # Add annotations for mean, median, and mode
    max_ylim = ax.get_ylim()[1]
    ax.text(mean_val, max_ylim*0.9, f'Mean: {mean_val:.2f}', color='red')
    ax.text(median_val, max_ylim*0.8, f'Median: {median_val:.2f}', color='green')
    ax.text(mode_val, max_ylim*0.7, f'Mode: {mode_val:.2f}', color='black')

    # Set title and labels
    ax.set_title(f'Distribution Plot of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # If no axis was provided, show the plot
    if ax is None:
        plt.show()

# %%
# Function to plot side by side
def plot_histo(df, columns):
    num_columns = len(columns)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(5 * num_columns, 6))

    for i, column in enumerate(columns):
        # Call the existing histo_graph function, passing the appropriate axis
        histo_graph(df, column, ax=axes[i])  # Pass the specific axis for each subplot

    plt.tight_layout()
    plt.show()

# %%
business=pd.read_json(r'C:\Users\marad.BEN10\Downloads\mlproject\yelp_dataset\yelp_academic_dataset_business.json',lines=True)

# %%
checkin=pd.read_json(r'C:\Users\marad.BEN10\Downloads\mlproject\yelp_dataset\yelp_academic_dataset_checkin.json',lines=True)

# %%
# Create an empty list to store chunks
chunks = []

# Set the chunk size (number of lines to read at a time)
chunk_size = 10000  # Adjust the chunk size based on your system's memory

# Read the JSON file in chunks
for chunk in pd.read_json(r'C:\Users\marad.BEN10\Downloads\mlproject\yelp_dataset\yelp_academic_dataset_user.json', lines=True, chunksize=chunk_size):
    chunks.append(chunk)  # Append each chunk to the list

# Concatenate all chunks into a single DataFrame
users = pd.concat(chunks,ignore_index=True)

# %%
# Create an empty list to store chunks
chunks = []

# Set the chunk size (number of lines to read at a time)
chunk_size = 10000  # Adjust the chunk size based on your system's memory

# Read the JSON file in chunks
for chunk in pd.read_json(r'C:\Users\marad.BEN10\Downloads\mlproject\yelp_dataset\yelp_academic_dataset_review.json', lines=True, chunksize=chunk_size):
    chunks.append(chunk)  # Append each chunk to the list

# Concatenate all chunks into a single DataFrame
review = pd.concat(chunks,ignore_index=True)

# %%
# Create an empty list to store chunks
chunks = []

# Set the chunk size (number of lines to read at a time)
chunk_size = 10000  # Adjust the chunk size based on your system's memory

# Read the JSON file in chunks
for chunk in pd.read_json(r'C:\Users\marad.BEN10\Downloads\mlproject\yelp_dataset\yelp_academic_dataset_tip.json', lines=True, chunksize=chunk_size):
    chunks.append(chunk)  # Append each chunk to the list

# Concatenate all chunks into a single DataFrame
tip = pd.concat(chunks,ignore_index=True)

# %%
pd.set_option('display.max_columns', None)

# %% [markdown]
# ## business.json (Businesses)
# This file contains information about all the businesses listed on Yelp.
# 
# ### name: Name of the business.
# ### address, city, state, postal_code: Location details of the business.
# ### latitude, longitude: Geographic coordinates of the business.
# ### stars: Average rating (out of 5) based on user reviews.
# ### review_count: Number of reviews for the business.
# ### categories: List of categories (e.g., restaurant, coffee shop, etc.).
# ### is_open: Whether the business is currently operating.
# ### attributes: Business-related attributes like Wi-Fi availability, outdoor seating, parking, etc.
# ### hours: Opening and closing hours for each day of the week.

# %%
business.head()

# %% [markdown]
# 
# ## tip.json (Tips)
# Contains short advice or tips left by users for businesses. This is different from reviews, as tips are shorter and meant to provide concise, practical information.
# 
# # Key Columns:
# ### user_id: The identifier for the user leaving the tip (links to user.json).
# ### business_id: Identifier for the business (links to business.json).
# ### text: The tip content.
# ### date: The date the tip was written.
# ### compliment_count: Number of compliments this tip has received.

# %%
tip.head()

# %%


# %% [markdown]
# # review.json (User Reviews)
# This file contains reviews written by users for businesses. Each review has a star rating, a timestamp, and text content. This is typically the largest file because of the volume of reviews.
# 
# ## review_id: Unique identifier for each review.
# ### user_id: Identifier for the user who wrote the review.
# ### business_id: Identifier for the business being reviewed (links to business.json).
# ### stars: Rating (out of 5) given by the user in this review.
# ### date: Date the review was written.
# ### text: The content of the review.
# ### useful, funny, cool: User feedback metrics on how useful, funny, or cool the review was, as voted by other users.
# 

# %%
review.head()

# %% [markdown]
# ## checkin.json (Check-ins)
# This file records the check-in data for businesses, showing the date and time that users checked in to a business.
# ### Key Columns:
# ### business_id: Identifier for the business (links to business.json).
# ### date: Timestamps of check-ins, separated by commas.

# %%
checkin.head()

# %% [markdown]
# ## user
# Contains information about the users who write reviews, including their social connections, review activity, and Yelp ratings.
# 
# ### Key Columns:
# ### user_id: Unique identifier for each user.
# ### name: The Yelp user's name.
# ### review_count: Number of reviews written by the user.
# ### yelping_since: Date when the user created their Yelp account.
# ### friends: A list of the user's friends (other user_ids).
# ### useful, funny, cool: Cumulative counts of how often the user’s reviews were rated as useful, funny, or cool by other users.
# ### elite: List of years in which the user achieved elite status on Yelp.
# ### fans: Number of fans the user has.
# ### average_stars: The user’s average rating (out of 5) across all their reviews.
# ### compliment_*: Counts of compliments the user has received in various categories (e.g., compliment_profile, compliment_cute, compliment_list, etc.).

# %%
users.head()

# %%
review_summary = review.groupby('business_id').agg({
    'stars': ['mean', 'count'],  # Get average rating and review count per business
    'useful': 'sum',  # Sum useful votes for all reviews of a business
})

# %%
review_summary.columns = ['avg_review_stars', 'review_count', 'total_useful_votes']

# %%
review_summary.head()

# %%
business_merged = pd.merge(business, review_summary, on='business_id', how='left')

# %%
business_merged.head()

# %%
user_summary = review.merge(users, on='user_id',how='left')

# %%
user_summary.head()

# %%
# Aggregate user data to create summary stats for users who reviewed the business
user_summary_agg= user_summary.groupby('business_id').agg({
    'elite': 'count',  # Number of reviews written by elite users grouped by bussiness
    'review_count': 'mean',  # Average number of reviews per reviewer grouped by bussiness
    'fans': 'mean',  # Average number of fans of reviewers grouped by bussiness
})

# %%
user_summary_agg.head()

# %%
user_summary_agg.columns = ['elite_user_count', 'review_count_per_bussiness', 'avg_fans_per_reviewer']

# %%
business_merged = pd.merge(business_merged, user_summary_agg, on='business_id', how='left')

# %%
business_merged.head()

# %%
checkin.head()

# %%
# Split the 'date' column by commas to get individual check-ins and count them
checkin['checkin_count'] = checkin['date'].apply(lambda x: len(str(x).split(',')))

# %%
checkin.head()

# %%
# Count total check-ins per business
checkin_summary = checkin.groupby('business_id').agg({'checkin_count': 'sum'})

# %%
checkin_summary.head()

# %%
business_merged = pd.merge(business_merged, checkin_summary, on='business_id', how='left')

# %%
tip.head()

# %%
# Aggregate tip data by business_id
tip_summary = tip.groupby('business_id').agg({'text': 'count', 'compliment_count': 'sum'}).rename(columns={'text': 'tip_count', 'compliment_count': 'total_compliments'})

# %%
# Merge with business data
business_merged = pd.merge(business_merged, tip_summary, on='business_id', how='left')

# %%
business_merged.head()

# %%
null_values(business_merged)

# %%
business_merged.head()

# %%
# List of columns to plot
val_parms = ["stars", "review_count_x", "is_open"]

# Create a figure with subplots
num_columns = len(val_parms)
fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(5 * num_columns, 6))

# Call the histo_graph function for each column, passing the corresponding axis
for i, column in enumerate(val_parms):
    histo_graph(business_merged, column, ax=axes[i]) 

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Creating a Target Variable
# 
# ### We are make our own metrics to what makes a bussiness a Validated bussiness. 
# 
# #### 1) is the bussiness open 
# #### 2) is bussiness has good Ratings (>2.5 stars)
# #### 3) is bussiness has sufficient review count (i.e. has more than 20 reviews or not)
# 
# 
# #### if all the above conditions are met then we are saying that bussiness is validated

# %%
#business_merged.describe(include='all')

# %%
business_merged.head(2)

# %%
# Define business validation: A business is validated if it's open, has > 2.5 stars, and has more than 20 reviews.
#business_merged['validated'] = (business_merged['is_open'] == 1) & (business_merged['stars'] >= 2.5) & (business_merged['review_count_x'] >= 20)

# %%
#business_merged['validated']=business_merged['validated'].astype(int)

# %%
business_merged.dtypes

# %%
#print(business_merged['validated'].describe())# basics stats to understand our target variable
#print(business_merged['validated'].value_counts(normalize=True)*100) # class disturbution of our taget variable

# %% [markdown]
# ### Here we can see that our class is imbalnced so we undersample our majority class (i.e."class 0")

# %%
#histo_graph(business_merged,'validated')

# %%
histo_graph(business_merged,'is_open')

# %%
business_merged.head()

# %% [markdown]
# #### If you observe attributes feature here the data is form dictionary key,value pairs. in this we have some important features like wheter the place is wheel chair accessble or not, wether there is parking available or not ?? etc..
# #### so , we extracting some important features from this and making them new features 

# %%
def extract_binary_features(feature_dict):
    # Ensure feature_dict is a dictionary, otherwise, return default values
    if not isinstance(feature_dict, dict):
        feature_dict = {}

    # Helper function to check if a feature is 'True'
    def is_true(value):
        return 1 if str(value).strip().lower() == 'true' else 0

    # Check if BusinessParking exists in any form
    business_parking_str = feature_dict.get('BusinessParking', 'None')
    business_parking_exists = 1 if isinstance(business_parking_str, str) and business_parking_str.lower() != 'none' else 0

    # Extract and return features
    return {
        'RestaurantsDelivery': is_true(feature_dict.get('RestaurantsDelivery', 'False')),
        'OutdoorSeating': is_true(feature_dict.get('OutdoorSeating', 'False')),
        'BusinessAcceptsCreditCards': is_true(feature_dict.get('BusinessAcceptsCreditCards', 'False')),
        'BikeParking': is_true(feature_dict.get('BikeParking', 'False')),
        'RestaurantsTakeOut': is_true(feature_dict.get('RestaurantsTakeOut', 'False')),
        'WiFi': 1 if feature_dict.get('WiFi') in ["u'free'", "True", "free"] else 0,
        'Caters': is_true(feature_dict.get('Caters', 'False')),
        'WheelchairAccessible': is_true(feature_dict.get('WheelchairAccessible', 'False')),
        'BusinessParking': business_parking_exists,
        'RestaurantsPriceRange2': feature_dict.get('RestaurantsPriceRange2', '0')  # Default price range '0'
    }


# %%
# Apply the feature extraction function to the 'attributes' column
business_merged['attributes'] = business_merged['attributes'].apply(lambda x: extract_binary_features(x if isinstance(x, dict) else {}))

# Convert the extracted dictionary back into individual columns
attributes_df = pd.json_normalize(business_merged['attributes'])

# Merge the extracted attributes back into the business_data DataFrame
business_merged = pd.concat([business_merged, attributes_df], axis=1)

# %%
business_merged.head()

# %%
import re

# Function to calculate hours per day
def calculate_daily_hours(time_range):
    if time_range == '0:0-0:0':  # Closed for the entire day
        return 0
    try:
        # Extract hours and minutes using regex
        match = re.match(r'(\d+):(\d+)-(\d+):(\d+)', time_range)
        if match:
            open_hour, open_min, close_hour, close_min = map(int, match.groups())
            
            # Handle cases like '9:0-0:0' (business closes at midnight)
            if close_hour == 0:
                close_hour = 24
            
            # Calculate the total hours
            open_time = open_hour + open_min / 60.0
            close_time = close_hour + close_min / 60.0
            
            return close_time - open_time if close_time > open_time else 0
        else:
            return 0  # Invalid time range
    except Exception as e:
        print(f"Error parsing time range {time_range}: {e}")
        return 0

# Function to calculate total hours for the entire week
def calculate_weekly_hours(hours_dict):
    total_hours = 0
    for day, time_range in hours_dict.items():
        total_hours += calculate_daily_hours(time_range)
    return total_hours

# %%
business_merged['hours'] = business_merged['hours'].apply(lambda x: x if isinstance(x, dict) else {})

# Calculate weekly hours for each business
business_merged['weekly_hours'] = business_merged['hours'].apply(calculate_weekly_hours)


# %%
histo_graph(business_merged,'weekly_hours')

# %%
business_merged.head()

# %%
# gives no of unique cities avaible in datset
print(len(business_merged['city'].value_counts().unique()))

# %%
# gives no of unique postal_code avaible in datset
print(len(business_merged['postal_code'].value_counts().unique()))

# %%
# gives no of unique states avaible in datset
print(len(business_merged['state'].value_counts().unique()))

# %%
numerical_df = business_merged.select_dtypes(include=['number'])
# Calculate the correlation matrix
corr_matrix = numerical_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Set the title
plt.title('Correlation Matrix Heatmap')
plt.show()

# %% [markdown]
# ### Droping Unneccessary features 
# #### as from the above you can see that we have way too many cities and postal code to perform one hot encoding, to save computing prower we are ignoring postal_code and city and considering state

# %%
business_merged=business_merged.drop(['business_id','name','address','hours','postal_code','latitude','longitude','attributes','categories', 'review_count_y','total_useful_votes','elite_user_count', 'stars', 'avg_fans_per_reviewer', 'RestaurantsTakeOut'],axis=1)


# %%
null_values(business_merged)

# %%
# gives no of unique cities avaible in datset
print(len(business_merged['city'].value_counts().unique()))

# %%
business_merged.dtypes

# %%
null_values(business_merged)

# %%
numerical_columns = business_merged.select_dtypes(include=['number']).columns

# Create subplots
num_columns = len(numerical_columns)
fig, axes = plt.subplots(nrows=(num_columns + 1) // 2, ncols=3, figsize=(12, 4 * ((num_columns + 1) // 2)))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Call the histo_graph function for each numerical column
for i, col in enumerate(numerical_columns):
    histo_graph( business_merged, col,axes[i])

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()

# %%
binary_columns = ['is_open',
        'RestaurantsDelivery','OutdoorSeating', 'BusinessAcceptsCreditCards', 'BikeParking',
       'RestaurantsTakeOut', 'WiFi', 'Caters', 'WheelchairAccessible',
       'BusinessParking', 'RestaurantsPriceRange2']


# %% [markdown]
# # Create subplots
# num_columns = len(binary_columns)
# nrows = (num_columns + 1) // 2  # Calculate number of rows needed
# fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows))
# axes = axes.flatten()  # Flatten the axes array for easy iteration
# 
# # Create grouped bar charts for each binary feature against the validation column
# for i, col in enumerate(binary_columns):
#     if col != 'validated':  # Skip the validation column itself
#         grouped_data = business_merged.groupby(['validated', col]).size().unstack(fill_value=0)
#         grouped_data.plot(kind='bar', stacked=True, ax=axes[i], alpha=0.7)
# 
#         axes[i].set_title(f'Grouped Bar Chart of {col} by Validation')
#         axes[i].set_xlabel('Validation')
#         axes[i].set_ylabel('Count')
#         axes[i].legend(title=col)
#         axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
# 
# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
# 
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()

# %%
#Split Data for preprocessing

# Define your features and target variable
X = business_merged.drop(columns=['is_open'])  # Features
y = business_merged['is_open']  # Target variable (validated)

# First, split into train and temp (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Split temp into test and validation sets
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Check the shapes of the resulting sets
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Validation set: {X_val.shape}, {y_val.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')

# %%


# %%
# Fill missing values
X_train['tip_count'] = X_train['tip_count'].fillna('0')
X_train['total_compliments'] = X_train['total_compliments'].fillna('0')
#X_train['attributes'] = X_train['attributes'].fillna('unknown')
#X_train['categories'] = X_train['categories'].fillna('unknown')
#X_train['hours'] = X_train['hours'].fillna('unknown')
X_train['checkin_count'] = X_train['checkin_count'].fillna('0')

# %%


# %%
print(X_train.columns)

# %%
numerical_df = X_train.select_dtypes(include=['number'])
# Calculate the correlation matrix
corr_matrix = numerical_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Set the title
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
# Check the shapes of the resulting sets
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Validation set: {X_val.shape}, {y_val.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')

# %%
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('Y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('Y_test.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('Y_val.csv', index=False)

# %% [markdown]
# #### If you observer Review_count, checkin_count and tip_count, they are strongly colreated with each other, this suggest that any one of these feature conveys same information as other, so we are just keeping Review_count feature and droping others
# 
# #### now if you observe tip_count and total_compliments, you can see that they are also correlated. this suugess that if one might take take time to complement they are most probably are tiping, we can drop this if we need while doing model tuning 
# 
# #### similary, if you observe RestaurantsDelivery,OutdoorSeating,RestaurantsTakeOut and Caters. they are mildly correlated with each others. this might be if the restarent is big enough they might be providing all these options. if our model overfits, we can drop some of these features
# 
# #### there is also a mild correlation between BikeParking,RestaurantsTakeOut and BusinessParking ,this suugest that if restaurants has BusinessParking, they might also offeing take out. coming to parking if they have BusinessParking, they most probably having bike parking as bikes wont take much space. we can drop BikeParking,RestaurantsTakeOut and keep BusinessParking if model overfits.
# 

# %%



