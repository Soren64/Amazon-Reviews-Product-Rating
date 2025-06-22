# Amazon-Reviews-Product-Rating
Machine Learning project using Python to make predictions on product score ratings.

# Contributors
Nicholas Matsuda
Graham Laroche
Joy Patel
Justin Tarnowski

# Project Overview
This project is a machine learning application developed in Python that predicts product ratings based on customer review text using the Amazon Reviews 23 dataset. It demonstrates data preprocessing, feature extraction, and basic predictive modeling techniques.

# Features
Utilized XGBoost regression model for predicting average product ratings based on review and product features.

Engineered features including helpful votes, review text length, and product price to improve model input.

Merged and processed large JSON datasets containing both review texts and product metadata for comprehensive analysis.

Evaluated model performance using Mean Absolute Error (MAE) and Spearman’s Rank Correlation to measure prediction accuracy and rank agreement.

Implemented a custom Top-k Hit Rate metric to assess how well the model predicts top-rated products.

Visualized results using scatter plots for predicted vs actual ratings and bar charts for hit rate metrics, aiding interpretability.

# Technologies Used
Python 3.x

pandas

scikit-learn

matplotlib

NumPy

# Dataset
This project uses the Amazon Reviews 23 dataset, which is available at:

https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

# How to Use
Clone the repository

Install the required Python packages (preferably in a virtual environment):

bash
Copy
Edit
pip install -r requirements.txt  
Run the Jupyter notebooks or Python script 'product_ratings.py' to preprocess data and train the model

Modify or extend the model and preprocessing steps as desired

Make sure you download the respective jsonl files for the category you wish to use; both the raw data and meta data files
Change the pathway as needed

# Future Improvements
Implement more advanced natural language processing techniques (e.g., word embeddings, transformers)

Experiment with deeper or ensemble machine learning models

Enhance data cleaning to handle edge cases and missing data

Additional feature engineering

Potential data sanitation and/or balancing; train the model with a greater balance of positive and negative examples

Notes
This project was primarily developed and coded independently as part of a team assignment.

The implementation is functional but could be further optimized and improved with additional time and resources.
