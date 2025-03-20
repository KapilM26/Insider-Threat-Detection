# %%
import csv
import os
from collections import defaultdict
from datetime import datetime, time, timedelta

import pandas as pd


import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

def get_user_usb_data(user_id, dataset_path):
    usb_data = []
    with open(os.path.join(dataset_path, "device.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if row[2] == user_id and row[5] == "Connect":  # Check user and only "Connect" activity
                usb_data.append(row)
    return usb_data


def get_num_usb_insertions_per_week(user, usb_data):
    weekly_usb_counts = defaultdict(int)
    all_weeks = set()
    for row in usb_data:
        file_time = datetime.strptime(row[1], "%m/%d/%Y %H:%M:%S")
        week = file_time.strftime("%Y-%W")
        all_weeks.add(week)
        weekly_usb_counts[week] += 1
    
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

        weekly_counts = {week: weekly_usb_counts.get(week, 0) for week in complete_weeks}
    else:
        weekly_counts = {}
    
    output_list = [[user, week, count] for week, count in sorted(weekly_counts.items())]
    return pd.DataFrame(output_list, columns=["user", "week", "num_usb_insertions"])


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

# %%
def get_user_logon_data(user_id, dataset_path):
    logon_data = []
    with open(os.path.join(dataset_path, "logon.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[2] == user_id:
                logon_data.append(row)
    return logon_data

# %%
def get_user_pc(logon_data):
    pc_dict = {}
    for row in logon_data:
        pc_dict[row[3]] = 1 + pc_dict.get(row[3], 0)
    user_pc = max(pc_dict, key=pc_dict.get)
    return user_pc


# %%
def get_num_other_PC_per_week(user, user_pc, logon_data):
    weekly_pc_counts = defaultdict(set)  # Dictionary to store unique PCs per week
    all_weeks = set()  # Set to track all weeks where logons occurred

    for row in logon_data:
        logon_time = datetime.strptime(row[1], "%m/%d/%Y %H:%M:%S")  # Adjusted format
        week = logon_time.strftime("%Y-%W")  # Year-Week format
        all_weeks.add(week)  # Track all weeks

        if row[3] != user_pc:  # Check if PC is different from user's primary PC
            weekly_pc_counts[week].add(
                row[3]
            )  # Add PC to the week's set (unique values only)

    # Ensure all weeks are included, even with 0 count
    min_week = min(all_weeks)
    max_week = max(all_weeks)

    # Generate all weeks between min and max
    start_date = datetime.strptime(min_week + "-1", "%Y-%W-%w")
    end_date = datetime.strptime(max_week + "-1", "%Y-%W-%w")

    current_date = start_date
    complete_weeks = set()

    while current_date <= end_date:
        week_str = current_date.strftime("%Y-%W")
        complete_weeks.add(week_str)
        current_date += timedelta(days=7)

    # Ensure every week has a count (0 if no other PCs were accessed)
    weekly_counts = {
        week: len(weekly_pc_counts[week]) if week in weekly_pc_counts else 0
        for week in complete_weeks
    }

    # Convert to DataFrame
    output_list = [[user, week, count] for week, count in sorted(weekly_counts.items())]
    return pd.DataFrame(output_list, columns=["user", "week", "num_other_pc"])


def get_after_hours_logons(
    logon_data, user, business_start=time(9, 0, 0), business_end=time(17, 0, 0)
):
    """
    Aggregates after-hours logons per week for a specified user.

    :param logon_data: List of logon events in the format [id, date, user, pc, activity]
    :param user: The specific user to filter logon events for.
    :param business_start: Datetime.time representing start of business hours.
    :param business_end: Datetime.time representing end of business hours.
    :return: DataFrame with ['user', 'week', 'after_hours_logons']
    """

    after_hours_counts = defaultdict(int)

    # Track all weeks for the user
    all_weeks = set()

    for row in logon_data:
        logon_id, timestamp, logon_user, pc, activity = row  # Unpack columns

        if (
            activity.lower() == "logon" and logon_user == user
        ):  # Only process logons for the specified user
            try:
                logon_time = datetime.strptime(timestamp, "%m/%d/%Y %H:%M:%S")
                logon_week = logon_time.strftime("%Y-%W")  # Ensure same format

                # Store this week to ensure it's included in results
                all_weeks.add(logon_week)

                # Extract only the time component
                logon_hour = logon_time.time()

                # Check if the logon occurred outside business hours
                if logon_hour < business_start or logon_hour >= business_end:
                    after_hours_counts[logon_week] += 1

            except ValueError:
                continue  # Skip invalid timestamps

    # Ensure all weeks in range are included (like `get_num_other_PC_per_week`)
    if all_weeks:
        min_week = min(all_weeks)
        max_week = max(all_weeks)

        # Generate all weeks in range
        start_date = datetime.strptime(min_week + "-1", "%Y-%W-%w")
        end_date = datetime.strptime(max_week + "-1", "%Y-%W-%w")

        current_date = start_date
        complete_weeks = set()

        while current_date <= end_date:
            week_str = current_date.strftime("%Y-%W")
            complete_weeks.add(week_str)
            current_date += timedelta(days=7)

        # Fill in missing weeks with 0
        after_hours_counts = {
            week: after_hours_counts.get(week, 0) for week in complete_weeks
        }

    # Convert to DataFrame
    result_data = [
        (user, week, after_hours_counts[week])
        for week in sorted(after_hours_counts.keys())
    ]
    after_hours_df = pd.DataFrame(
        result_data, columns=["user", "week", "after_hours_logons"]
    )

    return after_hours_df


# %%
def find_insider_answers_file(user, insider_root):
    """
    Recursively searches for the insider CSV file for the given user in the `insider_root` directory.

    :param user: The user ID (e.g., "CWW1120")
    :param insider_root: The root folder containing multiple r5.2-* subfolders.
    :return: The full path to the user's insider CSV file if found, else None.
    """
    for root, _, files in os.walk(insider_root):
        for file in files:
            if file.startswith(f"r5.2-") and file.endswith(
                f"-{user}.csv"
            ):  # Match user file format
                return os.path.join(root, file)  # Return full file path if found
    return None  # Return None if no file is found


def extract_weeks_from_csv(file_path):
    """
    Reads a CSV file using `csv.reader` and extracts unique weeks from the timestamps (3rd column).

    :param file_path: Path to the insider CSV file.
    :return: A set of detected `Year-Week` values.
    """
    insider_weeks = set()

    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 3:  # Ensure the timestamp column exists
                    continue
                try:
                    logon_time = datetime.strptime(
                        row[2], "%m/%d/%Y %H:%M:%S"
                    )  # Parse timestamp
                    week = logon_time.strftime("%Y-%W")  # Convert to Year-Week format
                    insider_weeks.add(week)
                except ValueError:
                    continue  # Skip rows with invalid timestamps
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return insider_weeks


def label_insider_weeks(df, user, insider_root):
    """
    Adds an 'insider' column to the DataFrame by checking if the user's week exists in their insider file.

    :param df: DataFrame containing ['user', 'week', 'num_other_pc']
    :param user: The user ID for whom the dataframe is filtered.
    :param insider_root: Path to the folder containing multiple r5.2-* subfolders.
    :return: DataFrame with an 'insider' column.
    """

    # Locate the user's insider file
    insider_file = find_insider_answers_file(user, insider_root)

    # If no insider file exists for the user, mark all weeks as 0 (not insider)
    if not insider_file:
        df["insider"] = 0
        return df

    # Extract weeks from the insider CSV file
    insider_weeks = extract_weeks_from_csv(insider_file)

    # Label insider weeks in the user's dataframe
    df["insider"] = df["week"].apply(lambda w: 1 if w in insider_weeks else 0)

    return df


def combine_user_feature_data(user, dataset_path, insider_root):
    # Get data from different feature functions
    logon_data = get_user_logon_data(user, dataset_path)
    user_pc = get_user_pc(logon_data)
    num_other_pc = get_num_other_PC_per_week(user, user_pc, logon_data)
    after_hours_logons = get_after_hours_logons(logon_data, user)

    exe_data = get_user_exe_data(user, dataset_path)
    num_exe_files = get_num_exe_per_week(user, exe_data)

    usb_data = get_user_usb_data(user, dataset_path)
    num_usb = get_num_usb_insertions_per_week(user, usb_data)

    # Extract relevant columns
    after_hours_df = after_hours_logons[["week", "after_hours_logons"]]
    exe_df         = num_exe_files[["week", "num_exe_files"]]
    usb_df         = num_usb[["week", "num_usb_insertions"]]
    other_pc_df    = num_other_pc[["week", "num_other_pc"]]

    # Merge all dataframes on "week" using an outer join
    merged_df = after_hours_df.merge(exe_df, on="week", how="outer") \
                              .merge(usb_df, on="week", how="outer") \
                              .merge(other_pc_df, on="week", how="outer")

    # Replace NaN with 0 in all feature columns
    merged_df.fillna(0, inplace=True)

    # Add user column
    merged_df.insert(0, "user", user)
    labeled_df = label_insider_weeks(merged_df, user, insider_root)
    return labeled_df








# Example usage
dataset_path = os.path.join("/Users/kartikraveendrannair/Downloads", "r5.2")
user = "MIB0203"
# user = "ALT1465"
insider_root = os.path.join("/Users/kartikraveendrannair/Downloads", "answers")

final_df = combine_user_feature_data(user, dataset_path, insider_root)

# # Get list of users from answer folder
# import os
# import glob

# def get_all_user_ids(insider_root):
#     """
#     Traverses the `answers/r5.2-*` directories to extract unique user IDs.
    
#     :param insider_root: The root directory containing the r5.2-* subfolders.
#     :return: A list of unique user IDs.
#     """
#     user_ids = set()  # Use a set to ensure uniqueness
    
#     # Find all files matching the pattern
#     file_paths = glob.glob(os.path.join(insider_root, "r5.2-*", "r5.2-*-*.csv"))
    
#     for file in file_paths:
#         # Extract user ID from filename (last part before .csv)
#         user_id = os.path.basename(file).split("-")[-1].replace(".csv", "")
#         user_ids.add(user_id)
    
#     return sorted(user_ids)  # Return sorted list of unique user IDs


# # Example usage
# insider_root = "/Users/kartikraveendrannair/Downloads/answers"  # Change this path accordingly
# user_list = get_all_user_ids(insider_root)
# print(user_list)  # Print the list of user IDs



# final_df_list = []  # List to hold individual user DataFrames
# counter = 0
# # Loop through each user and combine the data
# for user in user_list:
#     print("==========================================================")
#     print(counter)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#     counter +=1
#     user_df = combine_user_feature_data(user, dataset_path, insider_root)
#     final_df_list.append(user_df)  # Append the user's DataFrame to the list

# # Concatenate all user DataFrames into a single DataFrame
# final_df = pd.concat(final_df_list, ignore_index=True)


print(final_df)



final_df.to_csv("MIB0203.csv", index=False)

# LOF

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

# Select feature columns (excluding 'user', 'week', and 'insider')
feature_cols = ["after_hours_logons", "num_exe_files", "num_usb_insertions", "num_other_pc"]
X = final_df[feature_cols]
y_true = final_df["insider"]  # Ground truth


"""
# Train LOF model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # Adjust contamination as needed
y_pred = lof.fit_predict(X)

# Convert LOF output (-1 for anomaly) to match 'insider' format (1 for anomaly)
y_pred = [1 if val == -1 else 0 for val in y_pred]

# Get decision function scores
decision_scores = lof.negative_outlier_factor_

# Add predictions and decision scores to final_df
final_df["lof_prediction"] = y_pred
final_df["decision_function"] = decision_scores

# Compute Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print(final_df[final_df["insider"]==1].head(20))
"""

import numpy as np

# Define your possible contamination values (thresholds)
contamination_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

# Loop through each contamination value and calculate performance metrics
for contamination in contamination_values:
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    
    # Fit LOF model and get predictions
    y_pred = lof.fit_predict(X)  # Assuming X_train is your feature matrix

    # Get decision function scores
    decision_scores = lof.negative_outlier_factor_

    # Add predictions and decision scores to final_df
    final_df["lof_prediction"] = y_pred
    final_df["decision_function"] = decision_scores

    # Get decision function scores (negative outlier factor)
    # decision_scores = lof.decision_function(X)  # Higher is inlier, lower is outlier

    # Define different threshold values for anomaly detection
    thresholds = np.percentile(decision_scores, [5, 10, 15, 20, 25])  # Try different percentiles

    # Loop through each threshold and compute precision, recall, F1-score
    for threshold in thresholds:
        y_pred = np.where(decision_scores < threshold, 1, 0)  # Label as "1" if below threshold (anomaly)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Threshold = {threshold:.4f}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}")



    # y_pred = [1 if pred == -1 else 0 for pred in y_pred]  # Convert LOF output (-1 for outliers, 1 for inliers)
    
    # # Calculate precision, recall, and F1 score for insider=1 (malicious)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    
    # print(f"Contamination = {contamination}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}")




# Save the updated dataframe
# final_df.to_csv("insider_threat_with_lof.csv", index=False)
# print("File saved as insider_threat_with_lof.csv")
