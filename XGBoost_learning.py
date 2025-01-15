import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

problems_info = pd.read_csv("codeforces_problems.csv")
problems_info = problems_info[~(problems_info.iloc[:, 2:].eq(False).all(axis=1))]
problems_info = problems_info.reset_index(drop=True)

all_tasks = []
id_of_tasks = {}

for i in range(0, len(problems_info)):
    all_tasks.append(problems_info.iloc[i]['task_id'])
    id_of_tasks[all_tasks[-1]] = i

themes_of_tasks = []
for i in range(2, len(problems_info.columns)):
    problems_info[problems_info.columns[i]] = problems_info[problems_info.columns[i]].astype(int)
    themes_of_tasks.append(problems_info.columns[i])

submissions_database = pd.read_csv("user_abilities_database.csv")
submissions_database = submissions_database.iloc[:int(len(submissions_database) * 0.1)]
submissions_database = submissions_database[~(submissions_database.iloc[:, 4:].eq(0).all(axis=1))]
submissions_database = submissions_database[submissions_database['task_id'].isin(all_tasks)]
submissions_database = submissions_database.drop(columns=['ability_37'])
submissions_database = submissions_database.reset_index(drop=True)

submissions_database['is_rising'] = 0
submissions_database['similarity'] = 0

previous_user = "?"
lastRating = -1
is_rising_parameter = 1
count_decrease = 0
for i in range(0, len(submissions_database)):
    print(f"Progress: {i}/{len(submissions_database)}")
    if submissions_database.iloc[i]['handle'] == previous_user:
        if submissions_database.iloc[i]['rating_at_submission'] < lastRating:
            count_decrease += 1
            if count_decrease >= 2:
                is_rising_parameter = 0
        if submissions_database.iloc[i]['rating_at_submission'] - lastRating >= 50:
            count_decrease = 0
            is_rising_parameter = 1
    else:
        is_rising_parameter = 1
        count_decrease = 0
    submissions_database.loc[i, 'is_rising'] = is_rising_parameter
    previous_user = submissions_database.iloc[i]['handle']
    lastRating = submissions_database.iloc[i]['rating_at_submission']
    vector_ability = []
    for g in range(0, 37):
        vector_ability.append(submissions_database.iloc[i][f'ability_{g}'])
    vector_themes_of_tasks = []
    for g in range(0, 37):
        vector_themes_of_tasks.append(problems_info.loc[id_of_tasks[submissions_database.iloc[i]['task_id']], themes_of_tasks[g]])
    vector_ability = np.array(vector_ability)
    vector_ability = vector_ability / np.linalg.norm(vector_ability)
    vector_themes_of_tasks = np.array(vector_themes_of_tasks)
    submissions_database.loc[i, 'similarity'] = np.dot(vector_ability, vector_themes_of_tasks)

submissions_database = submissions_database.dropna()
submissions_database = submissions_database[submissions_database['is_rising'] == 1]
submissions_database = submissions_database[submissions_database['similarity'] > 0]
submissions_database = submissions_database.drop(columns=['handle', 'task_id', 'verdict'])
submissions_database = submissions_database.reset_index(drop=True)

feature_columns = ['rating_at_submission'] + [f'ability_{i}' for i in range(37)]
X = submissions_database[feature_columns]
y = submissions_database['similarity']

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"XGBoost - RMSE: {rmse:.6f}, R²: {r2:.6f}")

final_user_abilities = pd.read_csv("final_user_abilities.csv")

print("\nInitial data final_user_abilities:")
print(final_user_abilities.head())

prediction_required_columns = ['rating_at_submission'] + [f'ability_{i}' for i in range(37)]
missing_prediction_columns = [col for col in prediction_required_columns if col not in final_user_abilities.columns]
if missing_prediction_columns:
    raise ValueError(f"Missing required columns for prediction: {missing_prediction_columns}")

X_pred = final_user_abilities[prediction_required_columns]

if X_pred.isnull().sum().sum() > 0:
    X_pred.fillna(0, inplace=True)

X_pred_scaled = scaler.transform(X_pred)

similarity_predictions = xgb_model.predict(X_pred_scaled)

final_user_abilities['predicted_similarity'] = similarity_predictions

print("\nPredicted similarity values for final_user_abilities:")
print(final_user_abilities[['handle', 'predicted_similarity']].head())

final_user_abilities.to_csv("final_user_abilities_with_similarity.csv", index=False)
print("\nPredicted similarity values saved to 'final_user_abilities_with_similarity.csv'")
