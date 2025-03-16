import os
import csv
import pickle
from tqdm import tqdm

MALICIOUS_DOMAINS = ['wikileaks']

class RuleBasedClassifierS2:
    def __init__(self, vectorizer, min_matches=7):
        self.vectorizer = vectorizer
        self.min_matches = min_matches
        self.keywords = {'resume', 'strong', 'time', 'permanent', 'management', 'start', 'interview', 'years', 'growth', 'platform', 'hours', 'guidance', 'equivalent', 'notice', 'multitask', 'resign', 'skills', 'contribute', 'multiple', 'team', 'initiative', 'responsibilities', 'opportunity', 'degree', 'develop', 'concepts', 'key', 'recruiter', 'interface', 'process', 'dynamic', 'week', 'industry', 'resignation', 'technologies', 'letter', 'job', 'experience', 'opening', 'position', 'required', 'report', 'people', 'customer', 'passion', 'salary', 'sales', 'exit', 'benefits'}

    def classify_rule_based(self, text, keywords, analyzer, min_matches=2):
        tokens = set(analyzer(text))
        num_matches = len(tokens & keywords)
        return 1 if num_matches >= min_matches else 0
    
    def classify_email(self, text):
        analyzer = self.vectorizer.build_analyzer()
        pred = self.classify_rule_based(text, self.keywords, analyzer, min_matches=self.min_matches)

        return pred
    
class RuleBasedClassifierS3:
    def __init__(self, vectorizer, min_matches=9):
        self.vectorizer = vectorizer
        self.min_matches = min_matches
        self.keywords = {'suffer', 'schedule', 'gratitude', 'outraged', 'talk', 'angry', 'fed', 'hours', 'hard', 'appreciated', 'vacation', 'lets', 'complaints', 'employee', 'diligent', 'training', 'holidays', 'valued', 'operose', 'good', 'fault', 'seriously', 'today', 'company', 'work', 'faced', 'things', 'leave', 'job', 'irreplaceable', 'rest', 'demanding', 'bad', 'exacerbated'}

    def classify_rule_based(self, text, keywords, analyzer, min_matches=2):
        tokens = set(analyzer(text))
        num_matches = len(tokens & keywords)
        return 1 if num_matches >= min_matches else 0
    
    def classify_email(self, text):
        analyzer = self.vectorizer.build_analyzer()
        pred = self.classify_rule_based(text, self.keywords, analyzer, min_matches=self.min_matches)

        return pred

def detect_malicious_domain_users(dataset_path):
    malicious_users = set()
    with open(os.path.join(dataset_path, "http.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            url = row[4]
            for m in MALICIOUS_DOMAINS:
                if m in url:
                    user_id = row[2]
                    malicious_users.add(user_id)
                    break
    return malicious_users

def get_user_logon_data(user_id, dataset_path):
    logon_data = []
    with open(os.path.join(dataset_path, "logon.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[2] == user_id:
                logon_data.append(row)
    return logon_data

def get_user_pc(logon_data):
    pc_dict = {}
    for row in logon_data:
        pc_dict[row[3]] = 1 + pc_dict.get(row[3], 0)
    user_pc = max(pc_dict, key=pc_dict.get)
    return user_pc

def check_other_pc_login(logon_data):
    user_pc = get_user_pc(logon_data)
    for row in logon_data:
        if row[3] != user_pc:
            return True
    return False

def get_other_pc_users(dataset_path):
    other_pc_users = set()
    done_users = set()
    with open(os.path.join(dataset_path, "logon.csv"), "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            user_id = row[2]
            if user_id not in done_users:
                logon_data = get_user_logon_data(user_id, dataset_path)
                if check_other_pc_login(logon_data):
                    other_pc_users.add(user_id)
                done_users.add(user_id)
    return other_pc_users

def flag_s2_s3_users(dataset_path, vec_s2_path, vec_s3_path, org_domain='dtaa.com'):
    flagged_users = set()

    # Load your vectorizers explicitly
    with open(vec_s2_path, 'rb') as f:
        vectorizer_s2 = pickle.load(f)
    with open(vec_s3_path, 'rb') as f:
        vectorizer_s3 = pickle.load(f)

    # Instantiate your classifiers explicitly
    clf_s2 = RuleBasedClassifierS2(vectorizer=vectorizer_s2)
    clf_s3 = RuleBasedClassifierS3(vectorizer=vectorizer_s3)

    with open(os.path.join(dataset_path, 'email.csv'), newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header explicitly

        for row in tqdm(reader):
            user_id = row[2]
            recipients = (row[4]+';'+row[5]+';'+row[6]).split(';')
            recipients = [email.strip() for email in recipients if email.strip()]
            content = row[11].lower()
            activity = row[8].lower()

            if activity == 'send' and any(org_domain not in r for r in recipients):
                if clf_s2.classify_email(content) or clf_s3.classify_email(content):
                    flagged_users.add(user_id)

    return flagged_users

def get_flagged_users(dataset_path, s2_vectorizer_path, s3_vectorizer_path):
    print('getting other pc users..')
    other_pc_users = get_other_pc_users(dataset_path)
    with open('other_pc_users.pkl', 'wb') as f:
        pickle.dump(other_pc_users, f)
    print('other pc done')
    print('getting malicious domain users..')
    malicious_domain_users = detect_malicious_domain_users(dataset_path)
    with open('malicious_domain_users.pkl', 'wb') as f:
        pickle.dump(malicious_domain_users, f)
    print('mal domain done')
    print('getting email keyword users..')
    s2_s3_flagged = flag_s2_s3_users(dataset_path, s2_vectorizer_path, s3_vectorizer_path)
    with open('s2_s3_flagged.pkl', 'wb') as f:
        pickle.dump(s2_s3_flagged, f)
    print('email keywords done')

    all_flagged_users = other_pc_users.union(malicious_domain_users, s2_s3_flagged)
    return list(all_flagged_users)

dataset_path = 'Insider threat dataset\\r5.2'
s2_vectorizer_path = 'models\\s2_vectorizer.pkl'
s3_vectorizer_path = 'models\\s3_vectorizer.pkl'
flagged_users = get_flagged_users(dataset_path, s2_vectorizer_path, s3_vectorizer_path)



with open('flagged_users.pkl', 'wb') as f:
    pickle.dump(flagged_users, f)