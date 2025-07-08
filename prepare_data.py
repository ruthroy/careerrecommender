import pandas as pd
import pickle

# Load the dataset (update the path if needed)
df = pd.read_csv('CareerRecommenderDataset.csv')

# Preprocessing: Convert 'Yes'/'No' to 1/0 for all columns except 'Courses' and 'Career_Options'
cols_to_convert = [col for col in df.columns if col not in ['Courses', 'Career_Options']]
for col in cols_to_convert:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode the 'Courses' column
df_encoded = pd.get_dummies(df, columns=['Courses'])

# Convert boolean columns to integers
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

# Handle 'Career_Options' column: create binary columns for each unique career option
df_encoded['Career_Options_List'] = df_encoded['Career_Options'].str.split(', ')
all_career_options = set(career for sublist in df_encoded['Career_Options_List'] for career in sublist)
for career in all_career_options:
    df_encoded[career] = df_encoded['Career_Options_List'].apply(lambda x: 1 if career in x else 0)
df_encoded = df_encoded.drop(columns=['Career_Options', 'Career_Options_List'])

# Identify the career option columns
career_option_columns = [col for col in df_encoded.columns if col in all_career_options]

# X: all columns except career options; y: only career options
X = df_encoded.drop(columns=career_option_columns)
y = df_encoded[career_option_columns]

# Save preprocessed data for use in the web app
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)
with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)

print('Preprocessing complete. Data saved as X.pkl, y.pkl, and df.pkl.')
