import pandas as pd
import numpy as np
from tqdm import tqdm

submissions = pd.read_csv("users_submissions.csv")
tasks_info = pd.read_csv("codeforces_problems.csv")

submissions = submissions.iloc[:int(len(submissions) * 0.01)].reset_index(drop=True)
submissions = submissions.iloc[::-1].reset_index(drop=True)
tasks_info.iloc[:, 1:] = tasks_info.iloc[:, 1:].astype(float)
task_types = tasks_info.columns[1:]
user_abilities = {}
learning_rate = 0.1
records = []

def update_abilities(handle, task_id, problem_rating, user_rating, result):
    global user_abilities
    if handle not in user_abilities:
        user_abilities[handle] = np.zeros(len(task_types), dtype=np.float64)
    task_row = tasks_info[tasks_info['task_id'] == task_id]
    if task_row.empty:
        return
    task_features = task_row.iloc[:, 1:].values.flatten().astype(np.float64)
    if result == 1:
        weight = max(0, (problem_rating - user_rating) / max(problem_rating, user_rating))
        user_abilities[handle] += learning_rate * task_features * weight

for _, row in tqdm(submissions.iterrows(), total=len(submissions), desc="Processing submissions"):
    handle = row['handle']
    task_id = row['id_of_submission_task']
    verdict = row['verdict']
    problem_rating = row['problem_rating']
    user_rating = row['rating_at_submission']
    result = 1 if verdict == "OK" else 0
    current_abilities = user_abilities.get(handle, np.zeros(len(task_types), dtype=np.float64)).tolist()
    records.append({
        "handle": handle,
        "task_id": task_id,
        "verdict": verdict,
        "rating_at_submission": user_rating,
        **{f"ability_{i}": ability for i, ability in enumerate(current_abilities)}
    })
    update_abilities(handle, task_id, problem_rating, user_rating, result)

new_database = pd.DataFrame(records)
new_database.to_csv("user_abilities_database.csv", index=False)
