import pandas as pd

# Load datasets
# df_r3 = pd.read_csv("./r3.1/r3.1/psychometric.csv", usecols=["user_id"])
# df_r4 = pd.read_csv("./r4.2/r4.2/psychometric.csv", usecols=["user_id"])
# df_r5 = pd.read_csv("./r5.2/r5.2/psychometric.csv", usecols=["user_id"])

df_r3 = pd.read_csv("./mapped_psychometric_3.1.csv")#, usecols=["user_id"])
df_r4 = pd.read_csv("./mapped_psychometric_4.2.csv")#, usecols=["user_id"])
df_r5 = pd.read_csv("./mapped_psychometric_5.2.csv")#, usecols=["user_id"])

# Find common user IDs
# common_users = set(df_r3["user_id"]) & set(df_r4["user_id"]) & set(df_r5["user_id"])
# common_users = set(df_r3["user_id"]) & set(df_r4["user_id"]) 
# common_users = set(df_r3["user_id"]) & set(df_r5["user_id"])
# common_users = set(df_r4["user_id"]) & set(df_r5["user_id"])

# Print results
# print(f"Total unique users in r3.1: {df_r3['user_id'].nunique()}")
print(f"Total unique users in r4.2: {df_r4['user_id'].nunique()}")
print(f"Total unique users in r5.2: {df_r5['user_id'].nunique()}")
# print(f"Users present in both datasets: {len(common_users)}")

# Check duplicate count after merging
# merged_df = pd.concat([df_r3, df_r4, df_r5])
merged_df = pd.concat([df_r4, df_r5])
duplicate_count = merged_df.duplicated(subset=["user_id"]).sum()
print(f"Duplicate user IDs after merging: {duplicate_count}")

merged_df = merged_df.drop_duplicates(subset=["user_id"], keep="first")
print(merged_df.count())
# Save cleaned data to CSV
merged_df.to_csv("merged_psychometric.csv", index=False)

# print(common_users)