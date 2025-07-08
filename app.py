from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load your preprocessed data
with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)


# Helper: get interest columns (all columns in X except one-hot 'Courses' columns)
def get_interest_columns():
    # Exclude columns that start with 'Courses_' (one-hot encoded)
    return [col for col in X.columns if not col.startswith('Courses_')]

# Helper: get all course options
def get_course_options():
    return [col.replace('Courses_', '') for col in X.columns if col.startswith('Courses_')]

# Recommendation function (cosine similarity, simplified for web)
from sklearn.metrics.pairwise import cosine_similarity
def recommend_careers(user_interests, user_course, df, X, top_n=5):
    # Build user input vector
    user_data = {}
    for col in get_interest_columns():
        user_data[col] = int(user_interests.get(col, 0))
    # Set all course columns to 0, then set selected course to 1
    for course in get_course_options():
        colname = f'Courses_{course}'
        user_data[colname] = 1 if user_course == course else 0
    user_df = pd.DataFrame([user_data], columns=X.columns)
    # Compute similarity
    sim = cosine_similarity(user_df, X)[0]
    # Get top N most similar users
    top_indices = sim.argsort()[::-1][:top_n]
    # Collect all career options from those users
    recommended = set()
    for idx in top_indices:
        careers = df.iloc[idx]['Career_Options']
        for career in careers.split(','):
            recommended.add(career.strip())
    return list(recommended)[:top_n]

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    interest_columns = get_interest_columns()
    course_options = get_course_options()
    if request.method == 'POST':
        # Get user interests from form
        user_interests = {col: request.form.get(col, 0) for col in interest_columns}
        user_course = request.form.get('course', course_options[0] if course_options else None)
        recommendations = recommend_careers(user_interests, user_course, df, X, top_n=5)
    return render_template(
        'index.html',
        recommendations=recommendations,
        interest_columns=interest_columns,
        course_options=course_options
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
