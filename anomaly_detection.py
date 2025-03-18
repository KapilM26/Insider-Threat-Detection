from sklearn.neighbors import LocalOutlierFactor
import numpy as np

def predict_anomalies(user_data, algorithm='lof', model_params={'n_neighbors':20, 'contamination':0.025}, decision_threshold=-1.5):
    if algorithm=='lof':
        model = LocalOutlierFactor(**model_params)
    feature_cols = ["after_hours_logons", "num_exe_files", "num_usb_insertions", "num_other_pc"]
    X = user_data[feature_cols]
    y_pred = model.fit_predict(X)  # Assuming X_train is your feature matrix

    # Get decision function scores
    decision_scores = model.negative_outlier_factor_
    y_pred = np.where(decision_scores < decision_threshold, 1, 0)
    # Add predictions and decision scores to final_df
    return y_pred