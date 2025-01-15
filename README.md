# Codeforces Task Recommendation Model

This project implements a model based on **XGBoost** aimed at providing Codeforces users with the most suitable tasks for learning. 

All information about the dataset and how it was collected can be found in the associated project on Hugging Face: [UsersCodeforcesSubmissionsEnd2024](https://huggingface.co/datasets/denkCF/UsersCodeforcesSubmissionsEnd2024).

## How to Use

Follow these steps to use the model:

1. Add the Codeforces handle of the user for whom you want to predict tasks to the `user_handles` list in the file `save_final_vector_abilities_for_predictions.py` (line 31).

2. Run the script `XGBoost_learning.py`.

3. Run the script `get_predicted_tasks.py`. The predicted tasks will be displayed in the standard output as a list.
