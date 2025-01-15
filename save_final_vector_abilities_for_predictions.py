import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

submissions = pd.read_csv("users_submissions.csv")
tasks_info = pd.read_csv("codeforces_problems.csv")

tasks_info.iloc[:, 1:] = tasks_info.iloc[:, 1:].astype(float)

task_types = tasks_info.columns[1:]

learning_rate = 0.1

def build_user_abilities(user_submissions, tasks_info, task_types, learning_rate):
    user_abilities = np.zeros(len(task_types), dtype=np.float64)
    for _, row in user_submissions.iterrows():
        task_id = row["id_of_submission_task"]
        verdict = row["verdict"]
        problem_rating = row["problem_rating"]
        user_rating = row["rating_at_submission"]
        task_row = tasks_info[tasks_info["task_id"] == task_id]
        if task_row.empty:
            continue
        task_features = task_row.iloc[:, 1:].values.flatten().astype(np.float64)
        if verdict == "OK":
            weight = max(0, (problem_rating - user_rating) / max(problem_rating, user_rating))
            user_abilities += learning_rate * task_features * weight
    return user_abilities

user_handles = ["denk"] # Write there handles of users to predict or modify this part of code

def fetch_user_ratings(handles):
    handles_str = ';'.join(handles)
    try:
        response = requests.get(f"https://codeforces.com/api/user.info?handles={handles_str}")
        response.raise_for_status()
        rating_data = response.json()
        if rating_data["status"] != "OK":
            return {handle: None for handle in handles}
        handle_to_rating = {}
        for user in rating_data["result"]:
            handle_to_rating[user["handle"]] = user.get("rating", None)
        for handle in handles:
            if handle not in handle_to_rating:
                handle_to_rating[handle] = None
        return handle_to_rating
    except Exception as e:
        return {handle: None for handle in handles}

handle_ratings = fetch_user_ratings(user_handles)

final_abilities = []

for handle in tqdm(user_handles, desc="Processing Users"):
    user_submissions = submissions[submissions["handle"] == handle].iloc[::-1]
    abilities = build_user_abilities(user_submissions, tasks_info, task_types, learning_rate)
    user_rating = handle_ratings.get(handle, None)
    ability_dict = {
        "handle": handle,
        "rating_at_submission": user_rating,
    }
    for i, ability in enumerate(abilities[1:]):
        ability_dict[f"ability_{i}"] = ability
    final_abilities.append(ability_dict)

final_abilities_df = pd.DataFrame(final_abilities)
final_abilities_df.to_csv("final_user_abilities.csv", index=False)