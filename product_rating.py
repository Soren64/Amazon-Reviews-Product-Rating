from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


#Load the data (change path as necessary)
print("Reading files.")
df_reviews = pd.read_json('Appliances.jsonl', lines=True)
lines_read = sum(1 for _ in open('Appliances.jsonl', 'r'))
print(f"Total lines in the dataset: {lines_read}")
print("Reviews read.")
df_metadata = pd.read_json('meta_Appliances.jsonl', lines=True)
print("Metadata read.")

#Merge the datasets
print("Merging the data.")
df = pd.merge(df_reviews, df_metadata, left_on='parent_asin', right_on='parent_asin', how='inner')
print("Data merged.")


#Feature engineering
df['review_text_length'] = df['text'].apply(len)

#Define features and target
X = df[['helpful_vote', 'review_text_length', 'price']]
y = df['average_rating']

#Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Train the model using XGBoost
print("Training model.")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
print("Model Trained.")
xgb_model.fit(X_train, y_train)
print("Model fitted.")


#Make predictions
print("Predicting ratings on test set.")
y_pred = xgb_model.predict(X_test)



#Evaluation of model
print("Evaluation:")

#Calculate MAE w/ respect to regression task
mae = mean_absolute_error(y_test, y_pred)
#print(f"Mean Absolute Error (MAE): {mae:.4f}")

#Create a copy DataFrame for evaluation
df_test = X_test.copy()
df_test['actual_rating'] = y_test
df_test['predicted_rating'] = y_pred

#Sort predictions by predicted rating
df_test_sorted_pred = df_test[['actual_rating', 'predicted_rating']].sort_values(by='predicted_rating', ascending=False)


#Calculate Spearman's Rank Correlation
spearman_corr, _ = spearmanr(y_test, y_pred)

#Define Top-k Hit Rate function
def top_k_hit_rate(y_true, y_pred, k=10):
    top_k_true = np.argsort(y_true)[::-1][:k]
    top_k_pred = np.argsort(y_pred)[::-1][:k]
    hit_rate = len(set(top_k_true) & set(top_k_pred)) / k
    return hit_rate

#Calculate Top-10 Hit Rate
top_10_hit_rate = top_k_hit_rate(y_test, y_pred, k=10)

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
print(f"Top-10 Hit Rate: {top_10_hit_rate:.4f}")


#Visualization


#Spearman's correlation visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Ratings (Spearman's Rank Correlation)")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.show()

#Top-k hit rate visualization (Hit rate for k = 10)
hit_rate = top_k_hit_rate(y_test, y_pred, k=10)
plt.bar([1], [hit_rate], tick_label="Top-10 Hit Rate")
plt.ylim(0, 1)
plt.title("Top-10 Hit Rate")
plt.show()
