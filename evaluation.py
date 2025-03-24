import csv
import os
import pickle

from anomaly_detection import predict_anomalies
from feature_extraction import get_user_feature_data
from flagging import get_flagged_users
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def get_gt_user_list(dataset_path, insider_root):
    if os.path.exists('all_users.pkl'):
        with open('all_users.pkl', 'rb') as f:
            all_users = pickle.load(f)
    else:
        all_users = set()
        with open(os.path.join(dataset_path, 'psychometric.csv')) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                all_users.add(row[1])
        all_users = list(all_users)
        with open('all_users.pkl', 'wb') as f:
            pickle.dump(all_users, f)

    if os.path.exists('malicious_users.pkl'):
        with open('malicious_users.pkl', 'rb') as f:
            malicious_users = pickle.load(f)    
    else:
        malicious_users = set()
        for root, dirs, files in os.walk(insider_root):
            for dir_name in dirs:
                if dir_name.startswith("r5.2-"):
                    dir_path = os.path.join(root, dir_name)
                    for file in os.listdir(dir_path):
                        if file.endswith(".csv"):
                            user_id = file.split("-")[-1].replace(".csv", "")  # Extract user ID from filename
                            malicious_users.add(user_id)
        malicious_users = list(malicious_users)
        with open('malicious_users.pkl', 'wb') as f:
            pickle.dump(malicious_users, f)
    return all_users, malicious_users  


def evaluate(dataset_path, insider_root, flagged_users, all_users, malicious_users, algorithm='lof', model_params={'n_neighbors':20, 'contamination':0.025}, decision_threshold=-1.5):
    total_users = set(all_users)
    total_malicious_users = set(malicious_users)
    total_normal_users = total_users - total_malicious_users

    flagged_users = set(flagged_users)

    final_flagged_users = set()
    flagged_malicious_instances = 0
    flagged_normal_instances = 0

    malicious_instances = 0
    normal_instances = 0

    for user in tqdm(flagged_users):
        user_data = get_user_feature_data(user, dataset_path, insider_root)
        predictions = predict_anomalies(user_data, algorithm, model_params, decision_threshold)

        user_data['anomaly_pred'] = predictions

        user_flagged_by_anomaly = user_data['anomaly_pred'].any()

        if user_flagged_by_anomaly:
            final_flagged_users.add(user)

        for idx, row in user_data.iterrows():
            is_malicious = int(row.get('insider', 0))
            anomaly_pred = int(row['anomaly_pred'])

            if is_malicious:
                malicious_instances += 1
            else:
                normal_instances += 1

            if anomaly_pred == 1 and is_malicious:
                flagged_malicious_instances += 1
            elif anomaly_pred == 1 and not is_malicious:
                flagged_normal_instances += 1

    user_TP = len(final_flagged_users & total_malicious_users)
    user_FP = len(final_flagged_users & total_normal_users)

    UDR = user_TP / len(total_malicious_users)
    UFPR = user_FP / len(total_normal_users)

    print(f"User-level Detection Rate (UDR): {UDR:.4f}")
    print(f"User-level False Positive Rate (UFPR): {UFPR:.4f}")

    instance_DR = flagged_malicious_instances / malicious_instances
    instance_FPR = flagged_normal_instances / normal_instances if normal_instances else 0

    print(f"Instance-level Detection Rate (DR): {instance_DR:.4f}")
    print(f"Instance-level False Positive Rate (FPR): {instance_FPR:.4f}")

def run_evaluation():
    dataset_path = 'Insider threat dataset\\r5.2'
    s2_vectorizer_path = 'models\\s2_vectorizer.pkl'
    s3_vectorizer_path = 'models\\s3_vectorizer.pkl'
    insider_root = 'Insider threat dataset\\answers'
    algorithms = {
        'iforest':({'contamination':0.25}, -0.025), 
        'lof':({'n_neighbors':20, 'contamination':0.01}, -1.0), 
        'ocsvm':({'kernel':"rbf", "gamma":"scale", "nu":0.99},-0.5)
        }
    flagged_users = get_flagged_users(dataset_path, s2_vectorizer_path, s3_vectorizer_path)
    all_users, malicious_users = get_gt_user_list(dataset_path, insider_root)
    for algorithm, (model_params, decision_threshold) in algorithms.items():
        print(f"Evaluating algorithm: {algorithm}")
        evaluate(dataset_path, insider_root, flagged_users, all_users, malicious_users, algorithm, model_params, decision_threshold)


if __name__=='__main__':
    run_evaluation()