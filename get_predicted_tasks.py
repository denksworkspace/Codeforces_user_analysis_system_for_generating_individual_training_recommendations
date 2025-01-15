import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

final_abilities_database = pd.read_csv("final_user_abilities_with_similarity.csv")

list_for_demonstration = None

for i in range(0, len(final_abilities_database)):
    print(f"For user '{final_abilities_database.iloc[i]['handle']}':")
    vector_ability = []
    for g in range(0, 37):
        vector_ability.append(final_abilities_database.iloc[i][f'ability_{g}'])
    vector_ability = np.array(vector_ability)
    vector_ability = vector_ability / np.linalg.norm(vector_ability)
    range_of_tasks = []
    user_rating = final_abilities_database.iloc[i]['rating_at_submission']
    for task in all_tasks:
        if not (0 <= problems_info.iloc[id_of_tasks[task]]['rating'] - user_rating <= 200):
            continue
        vector_themes_of_tasks = []
        for g in range(0, 37):
            vector_themes_of_tasks.append(problems_info.loc[id_of_tasks[task], themes_of_tasks[g]])
        vector_themes_of_tasks = np.array(vector_themes_of_tasks)
        similarity_for_task = np.dot(vector_ability, vector_themes_of_tasks)
        range_of_tasks.append([similarity_for_task / final_abilities_database.iloc[i][f'predicted_similarity'], task])
    range_of_tasks.sort(reverse=True)
    print(range_of_tasks[:5])
    print()
    if final_abilities_database.iloc[i]['handle'] == 'denk':
        list_for_demonstration = range_of_tasks[:5]

if list_for_demonstration:
    user_data = final_abilities_database[final_abilities_database['handle'] == 'denk'].iloc[0]
    abilities = [user_data[f'ability_{g}'] for g in range(37)]
    abilities = np.array(abilities) / np.linalg.norm(abilities)

    task_abilities = {task[1]: [] for task in list_for_demonstration}
    min_abilities = {task[1]: None for task in list_for_demonstration}
    for task in task_abilities:
        min_value = float('inf')
        for j, theme in enumerate(themes_of_tasks):
            if problems_info.loc[id_of_tasks[task], theme] == 1:
                task_abilities[task].append(j)
                if abilities[j] < min_value:
                    min_value = abilities[j]
                    min_abilities[task] = j

    plt.figure(figsize=(12, 6))
    plt.bar(range(37), abilities, alpha=0.7, label="User Abilities")

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, task in enumerate(task_abilities):
        if min_abilities[task] is not None:
            plt.scatter(min_abilities[task], abilities[min_abilities[task]], color=colors[idx],
                        s=50 * (len(task_abilities) - idx), label=f"Task {task} (min ability)")

    plt.xlabel("Ability Index")
    plt.ylabel("Normalized Ability Value")
    plt.title(f"Abilities of user with Task Associations (Min Highlighted)")
    plt.legend()
    plt.grid(True)
    plt.show()
