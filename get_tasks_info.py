import requests
import csv


def fetch_problems():
    url = "https://codeforces.com/api/problemset.problems"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch problems. Status code: {response.status_code}")
    data = response.json()
    if data['status'] != "OK":
        raise Exception("Failed to fetch problems from Codeforces API.")
    return data['result']['problems'], data['result']['problemStatistics']


def index_problems(problems):
    indexed_problems = []
    all_tags = set(tag for problem in problems for tag in problem.get('tags', []))

    for problem in problems:
        problem_id = f"{problem['contestId']}{problem['index']}"
        rating = problem.get('rating', None)
        tags = {tag: tag in problem.get('tags', []) for tag in all_tags}
        indexed_problems.append({
            'task_id': problem_id,
            'rating': rating,
            **tags
        })

    return indexed_problems, list(all_tags)


def save_to_csv(problems, all_tags, filename="codeforces_problems.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['task_id', 'rating'] + all_tags)
        writer.writeheader()
        writer.writerows(problems)


def main():
    print("Fetching problems from Codeforces...")
    problems, _ = fetch_problems()

    print("Indexing problems...")
    indexed_problems, all_tags = index_problems(problems)

    print(f"Saving to CSV file...")
    save_to_csv(indexed_problems, all_tags)

    print(f"Data saved successfully!")


if __name__ == "__main__":
    main()
