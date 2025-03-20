import os
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

def get_user_exe_data(user_id, dataset_path):
    exe_data = []
    with open(os.path.join(dataset_path, "file.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if row[2] == user_id and row[4].endswith(".exe"):  # Check user and .exe files
                exe_data.append(row)
    return exe_data

def get_num_exe_per_week(user, exe_data):
    weekly_exe_counts = defaultdict(int)
    all_weeks = set()
    
    for row in exe_data:
        file_time = datetime.strptime(row[1], "%m/%d/%Y %H:%M:%S")
        week = file_time.strftime("%Y-%W")
        all_weeks.add(week)
        weekly_exe_counts[week] += 1
    
    # Ensure all weeks are included, even with 0 count
    min_week = min(all_weeks, default=None)
    max_week = max(all_weeks, default=None)
    
    if min_week and max_week:
        start_date = datetime.strptime(min_week + "-1", "%Y-%W-%w")
        end_date = datetime.strptime(max_week + "-1", "%Y-%W-%w")

        current_date = start_date
        complete_weeks = set()

        while current_date <= end_date:
            week_str = current_date.strftime("%Y-%W")
            complete_weeks.add(week_str)
            current_date += timedelta(days=7)

        weekly_counts = {week: weekly_exe_counts.get(week, 0) for week in complete_weeks}
    else:
        weekly_counts = {}
    
    output_list = [[user, week, count] for week, count in sorted(weekly_counts.items())]
    return pd.DataFrame(output_list, columns=["user", "week", "num_exe_files"])

# user = "ACM1770"
user = "DEO1964"
exe_data = get_user_exe_data(user, "r5.2/r5.2")
num_exe_files = get_num_exe_per_week(user, exe_data)
print(num_exe_files)
