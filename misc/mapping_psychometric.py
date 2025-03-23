import os
import pandas as pd

def create_malicious_mapping(psychometric_file, answers_folder):
    # Load the psychometric dataset
    df_psychometric = pd.read_csv(psychometric_file)
    
    # Extract user IDs from psychometric.csv
    user_ids = set(df_psychometric['user_id'])  # Assuming 'UserID' is the column name
    
    # Get malicious user IDs from answers folder
    malicious_users = set()
    for root, dirs, files in os.walk(answers_folder):
        for dir_name in dirs:
            if dir_name.startswith("r3.1-"):
                dir_path = os.path.join(root, dir_name)
                for file in os.listdir(dir_path):
                    if file.endswith(".csv"):
                        user_id = file.split("-")[-1].replace(".csv", "")  # Extract user ID from filename
                        malicious_users.add(user_id)
    
    # Add malicious column to dataset
    df_psychometric['Malicious'] = df_psychometric['user_id'].apply(lambda x: 1 if str(x) in malicious_users else 0)
    
    # Save the new dataset
    output_file = "mapped_psychometric_3.1.csv"
    df_psychometric.to_csv(output_file, index=False)
    print(f"New dataset saved as {output_file}")

# Example usage
create_malicious_mapping("./r4.2/r4.2/psychometric.csv", "./")#"./answers/answers")
