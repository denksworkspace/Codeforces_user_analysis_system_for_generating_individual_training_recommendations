import requests
import csv

def fetch_users_with_min_rating(min_rating=1200):
    users = []
    page = 1

    while True:
        print(f"Fetching users from rating page {page}...")
        url = f"https://codeforces.com/api/user.ratedList?activeOnly=true&from={(page - 1) * 1000 + 1}&count=1000"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch users from page {page}. Status code: {response.status_code}")
            break

        data = response.json()
        if data['status'] != "OK":
            print(f"Failed to fetch users from page {page}. Reason: {data.get('comment', 'Unknown error')}")
            break

        batch = [user for user in data['result'] if user['rating'] >= min_rating]
        if not batch:
            break

        users.extend(batch)
        if any(user['rating'] < min_rating for user in data['result']):
            break

        page += 1

    return users

def fetch_user_rating_history(handle):
    url = f"https://codeforces.com/api/user.rating?handle={handle}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch rating history for user {handle}. Status code: {response.status_code}")
        return []
    data = response.json()
    if data['status'] != "OK":
        print(f"Failed to fetch rating history for user {handle}. Reason: {data.get('comment', 'Unknown error')}")
        return []
    return data['result']

def fetch_user_submissions(handle):
    url = f"https://codeforces.com/api/user.status?handle={handle}&from=1&count=100000"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch submissions for user {handle}. Status code: {response.status_code}")
        return []
    data = response.json()
    if data['status'] != "OK":
        print(f"Failed to fetch submissions for user {handle}. Reason: {data.get('comment', 'Unknown error')}")
        return []
    return data['result']

def get_rating_at_time(rating_history, timestamp):
    for i in range(len(rating_history) - 1, -1, -1):
        if rating_history[i]['ratingUpdateTimeSeconds'] <= timestamp:
            return rating_history[i]['newRating']
    return None

def process_submissions(handle, submissions, rating_history):
    processed = []
    for submission in submissions:
        problem = submission.get("problem", {})
        user_rating_at_submission = get_rating_at_time(rating_history, submission['creationTimeSeconds'])
        processed.append({
            'id': submission['id'],
            'handle': handle,
            'rating_at_submission': user_rating_at_submission,
            'problem_rating': problem.get('rating', None),
            'id_of_submission_task': f"{problem.get('contestId', '')}{problem.get('index', '')}",
            'verdict': submission.get('verdict', None),
            'time': submission['creationTimeSeconds']
        })
    return processed

def save_submissions_to_csv(submissions, filename="users_submissions.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'id', 'handle', 'rating_at_submission', 'problem_rating',
            'id_of_submission_task', 'verdict', 'time'
        ])
        writer.writeheader()
        writer.writerows(submissions)

def main():
    print("Fetching users with rating >= 1200...")
    users = fetch_users_with_min_rating(min_rating=1200)
    print(f"Found {len(users)} users with rating >= 1200.")

    all_submissions = []
    for i, user in enumerate(users, start=1):
        handle = user['handle']
        print(f"[{i}/{len(users)}] Fetching submissions for user: {handle} (rating: {user['rating']})")
        try:
            rating_history = fetch_user_rating_history(handle)
            submissions = fetch_user_submissions(handle)
            processed = process_submissions(handle, submissions, rating_history)
            all_submissions.extend(processed)
        except Exception as e:
            print(f"Error fetching submissions for user {handle}: {e}")

    print("Saving all submissions to CSV...")
    save_submissions_to_csv(all_submissions)
    print("Submissions saved successfully!")

if __name__ == "__main__":
    main()
