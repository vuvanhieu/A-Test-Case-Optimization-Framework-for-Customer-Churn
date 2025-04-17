import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# ==== Risk Threshold Config ====
LOW_CREDIT_THRESHOLD = 580
HIGH_CREDIT_THRESHOLD = 700
LOW_BALANCE_THRESHOLD = 1000
HIGH_BALANCE_THRESHOLD = 100000
NEW_CUSTOMER_TENURE = 2
SENIOR_AGE_THRESHOLD = 60
# ===============================

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from collections import Counter
from sklearn.utils import resample
import tensorflow.keras.backend as K

import os
import shutil

def collect_files_for_report(result_folder, report_dir="reports"):
    """
    Copy all essential .csv, .txt, and .png files to a centralized 'reports' directory
    for paper writing.
    """
    os.makedirs(report_dir, exist_ok=True)

    multitask_dir = os.path.join(result_folder, "Multitask DNN")
    baselines_dir = os.path.join(result_folder, "Baselines")
    alert_dir = os.path.join(result_folder, "churn_risk_alert")

    important_csv_files = [
        os.path.join(multitask_dir, "gridsearch_multitask_results.csv"),
        os.path.join(multitask_dir, "test_case_list.csv"),
        os.path.join(multitask_dir, "explanation_results.csv"),
        os.path.join(multitask_dir, "churn_reason_statistics.csv"),
        os.path.join(baselines_dir, "baseline_comparison.csv"),
        os.path.join(alert_dir, "churn_risk_alert_log.csv"),
        os.path.join(alert_dir, "high_risk_customers_with_reasons.csv")
    ]

    important_txt_files = [
        os.path.join(multitask_dir, "focal_vs_nonfocal_comparison_report.txt"),
        os.path.join(result_folder, "best_model_summary.txt"),
        os.path.join(alert_dir, "churn_classification_report.txt")
    ]

    important_png_files = []

    # Add training history, confusion matrices, ROC, PR from each test case folder
    for root, dirs, files in os.walk(multitask_dir):
        for file in files:
            if file.endswith(('.png', '.txt')) and (
                "training_validation" in file or
                "confusion_matrix" in file or
                "roc_curve" in file or
                "pr_curve" in file
            ):
                important_png_files.append(os.path.join(root, file))
            elif file == "test_classification_report.txt":
                important_txt_files.append(os.path.join(root, file))

    # Add overall comparison plots
    for metric in ['accuracy', 'f1-score', 'precision', 'recall']:
        plot_path = os.path.join(result_folder, f"compare_{metric}.png")
        if os.path.exists(plot_path):
            important_png_files.append(plot_path)

    # Add before/after label distribution plots
    for f in os.listdir(result_folder):
        if f.startswith(("before_", "after_")) and f.endswith(".png"):
            important_png_files.append(os.path.join(result_folder, f))

    # Add churn reason and alert level plots
    extra_pngs = [
        os.path.join(multitask_dir, "churn_reason_barplot.png"),
        os.path.join(alert_dir, "churn_risk_level_distribution.png"),
        os.path.join(alert_dir, "high_risk_reason_barplot.png")
    ]
    important_png_files.extend([p for p in extra_pngs if os.path.exists(p)])

    # Copy all identified files to report folder
    for filepath in important_csv_files + important_txt_files + important_png_files:
        if os.path.exists(filepath):
            shutil.copy(filepath, os.path.join(report_dir, os.path.basename(filepath)))

    return f"✅ All important files copied to: {report_dir}"


def focal_loss_binary(alpha=0.75, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return focal_loss_fixed

def focal_loss_categorical(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))
    return focal_loss_fixed


def balance_labels(y_array, target_dist=None):
    """
    Apply undersampling to balance the label distribution of y_array.
    If target_dist is None, equalizes all classes.
    """
    df = pd.DataFrame({'label': y_array})
    counts = df['label'].value_counts()
    min_count = target_dist if target_dist else counts.min()
    balanced_df = pd.concat([
        df[df['label'] == label].sample(n=min_count, replace=False, random_state=42)
        for label in counts.index
    ])
    return balanced_df.index.values

def apply_balancing_all_tasks(X_train, y_train_dict):
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import NearestNeighbors

    # Step 1: SMOTE for churn
    smote = SMOTE(random_state=42)
    X_res, y_churn_res = smote.fit_resample(X_train, y_train_dict['churn'])

    # Step 2: Align other labels
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, indices = nbrs.kneighbors(X_res)
    indices = indices.flatten()

    y_score_res = y_train_dict['score'][indices]
    y_balance_res = y_train_dict['balance'][indices]

    # Step 3: Balance score (multiclass) and balance (binary)
    score_labels = np.argmax(y_score_res, axis=1)
    idx_score_bal = balance_labels(score_labels)
    idx_balance_bal = balance_labels(y_balance_res)

    # Intersect all indices (churn-balanced already)
    final_idx = np.intersect1d(idx_score_bal, idx_balance_bal)

    # Final output
    return X_res[final_idx], {
        'churn': y_churn_res[final_idx],
        'score': y_score_res[final_idx],
        'balance': y_balance_res[final_idx]
    }


def plot_all(y_true, y_pred, y_proba, task_name, save_dir, model_name=""):
    """
    Plot and save:
    - Confusion matrix (regular and normalized)
    - ROC curve (per class)
    - Precision-Recall curve
    - AUC score
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    os.makedirs(save_dir, exist_ok=True)
    postfix = f"_{model_name}" if model_name else ""

    # ✅ Ensure y_true is label (not one-hot)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # ✅ Ensure y_proba has proper shape for binary classification
    if y_proba.ndim == 1:
        y_proba = np.vstack([1 - y_proba, y_proba]).T
    elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = np.hstack([1 - y_proba, y_proba])

    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Confusion Matrix (regular)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # plt.title(f"Confusion Matrix - {task_name} {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix{postfix}.png"))
    plt.close()

    # Confusion Matrix (normalized)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    df_cm_norm = pd.DataFrame(cm_norm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # plt.title(f"Normalized Confusion Matrix - {task_name} {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix_normalized{postfix}.png"))
    plt.close()

    # ❗ Skip ROC/PR/AUC if only one class in y_true
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ Skipping ROC/PR/AUC for {task_name} {model_name} (only one class in y_true)")
        return

    # ROC Curve
    plt.figure(figsize=(8, 6))
    try:
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {label} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.title(f"ROC Curve - {task_name} {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_roc_curve{postfix}.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ ROC plot error for {task_name} {model_name}: {e}")

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    try:
        if y_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            for i, label in enumerate(labels):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f"Class {label} (AUC = {pr_auc:.2f})")
        # plt.title(f"Precision-Recall Curve - {task_name} {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_pr_curve{postfix}.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ PR plot error for {task_name} {model_name}: {e}")

    # AUC Score
    try:
        if y_proba.shape[1] == 2:
            auc_score = roc_auc_score(y_true, y_proba[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            auc_score = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        with open(os.path.join(save_dir, f"{task_name}_AUC{postfix}.txt"), "w") as f:
            f.write(f"AUC Score ({task_name} - {model_name}): {auc_score:.4f}\n")
    except Exception as e:
        print(f"⚠️ Could not calculate AUC for {task_name} {model_name}: {e}")



def plot_label_distribution(y_dict, save_dir=".", prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    for task_name, y in y_dict.items():
        if y.ndim == 2:  # One-hot encoded labels
            y = np.argmax(y, axis=1)

        label_counts = pd.Series(y).value_counts().sort_index()
        total = len(y)
        percentages = label_counts / total * 100

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette="Set2")

        for i, count in enumerate(label_counts.values):
            percent = percentages.iloc[i]
            ax.text(i, count + 0.5, f"{count} ({percent:.1f}%)", ha='center', va='bottom', fontsize=10)

        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        # Gộp prefix nếu có
        task_tag = f"{prefix}_{task_name}" if prefix else task_name
        file_path = os.path.join(save_dir, f"{task_tag}_distribution.png")
        plt.savefig(file_path)
        plt.close()

        print(f"📊 Saved label distribution with % for {task_name} to: {file_path}")


def plot_high_risk_reason_distribution(csv_path, output_path="high_risk_reason_barplot.png"):
    """
    Generate a barplot showing the distribution of churn reasons among high-risk customers.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    df = pd.read_csv(csv_path)

    if "Churn_Reason_Suggestion" not in df.columns:
        print("❌ 'Churn_Reason_Suggestion' column not found.")
        return

    # Split and count reasons
    reason_list = df["Churn_Reason_Suggestion"].dropna().str.split(", ")
    counter = Counter()
    for reasons in reason_list:
        counter.update(reasons)

    reason_df = pd.DataFrame(counter.items(), columns=["Reason", "Count"]).sort_values("Count", ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=reason_df, x="Count", y="Reason", palette="mako", hue="Reason", legend=False)
    # plt.title("Top Churn Reasons among High-Risk Customers")
    plt.xlabel("Frequency")
    plt.ylabel("Churn Reason")

    plt.savefig(output_path)
    plt.close()

    print(f"📊 High-risk churn reason plot saved to: {output_path}")

def plot_training_history(history, output_dir):
    """
    Plot training and validation loss/accuracy from a Keras history object.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    for key in history.history.keys():
        if 'accuracy' in key:
            plt.plot(history.history[key], label=key)
    # plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    
    acc_path = os.path.join(output_dir, 'training_validation_accuracy.png')
    plt.savefig(acc_path)
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    for key in history.history.keys():
        if 'loss' in key:
            plt.plot(history.history[key], label=key)
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()

    loss_path = os.path.join(output_dir, 'training_validation_loss.png')
    plt.savefig(loss_path)
    plt.close()
    
    return acc_path, loss_path
    
def infer_reason(row):
    reasons = []
    if row["CreditScore"] < LOW_CREDIT_THRESHOLD:
        reasons.append("Low credit score")
    elif row["CreditScore"] > HIGH_CREDIT_THRESHOLD:
        reasons.append("High credit score")

    if row["Balance"] > HIGH_BALANCE_THRESHOLD:
        reasons.append("High balance")
    elif row["Balance"] < LOW_BALANCE_THRESHOLD:
        reasons.append("Very low balance")

    if row["Tenure"] <= NEW_CUSTOMER_TENURE:
        reasons.append("New customer")

    if row["Age"] >= SENIOR_AGE_THRESHOLD:
        reasons.append("Older customer")

    return ", ".join(reasons)

def generate_high_risk_recommendations(alert_csv_path, output_path="high_risk_customers_with_reasons.csv"):
    df = pd.read_csv(alert_csv_path)
    high_risk_df = df[df["Risk_Level"] == "High"].copy()
    high_risk_df["Churn_Reason_Suggestion"] = high_risk_df.apply(infer_reason, axis=1)
    high_risk_df.to_csv(output_path, index=False)
    print(f"✅ High risk customers with suggested churn reasons saved to: {output_path}")



def apply_smote_to_main_task(X_train, y_train_dict):
    y_churn = y_train_dict['churn']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_churn)

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, indices = nbrs.kneighbors(X_res)
    indices = indices.flatten()

    y_train_dict_new = {
        'churn': y_res,
        'score': y_train_dict['score'][indices],
        'balance': y_train_dict['balance'][indices]
    }
    return X_res, y_train_dict_new

def load_clean_data_multitask(filepath):
    df = pd.read_csv(filepath)
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    le = LabelEncoder()
    df['Geography'] = le.fit_transform(df['Geography'])
    df['Gender'] = le.fit_transform(df['Gender'])
    df['CreditScoreClass'] = pd.cut(df['CreditScore'], bins=[0, 580, 700, 850], labels=[0, 1, 2]).astype(int)
    df['HighBalanceFlag'] = (df['Balance'] > 100000).astype(int)
    X = df.drop(['Exited', 'CreditScoreClass', 'HighBalanceFlag'], axis=1).values
    X = StandardScaler().fit_transform(X)
    y_dict = {
        'churn': df['Exited'].values,
        'score': to_categorical(df['CreditScoreClass'].values, num_classes=3),
        'balance': df['HighBalanceFlag'].values
    }
    return X, y_dict, df

def build_multitask_dnn(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out_churn = Dense(1, activation='sigmoid', name='churn')(x)
    out_score = Dense(3, activation='softmax', name='score')(x)
    out_balance = Dense(1, activation='sigmoid', name='balance')(x)
    model = Model(inputs=inputs, outputs=[out_churn, out_score, out_balance])
    return model

def train_multitask_model(model, X_train, y_train_dict,
                          X_val, y_val_dict,
                          X_test, y_test_dict,
                          epochs=30, batch_size=64,
                          loss_weights=None,
                          use_focal_churn=False,
                          use_focal_score=False,
                          use_focal_balance=False,
                          save_path=None, save_dir=".",
                          model_name="Multitask DNN"):

    loss_fn = {
        'churn': focal_loss_binary(alpha=0.75, gamma=2.0) if use_focal_churn else 'binary_crossentropy',
        'score': focal_loss_categorical(alpha=0.25, gamma=2.0) if use_focal_score else 'categorical_crossentropy',
        'balance': focal_loss_binary(alpha=0.75, gamma=2.0) if use_focal_balance else 'binary_crossentropy'
    }

    if loss_weights is None:
        loss_weights = {'churn': 1.0, 'score': 0.5, 'balance': 0.5}

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        loss_weights=loss_weights,
        metrics={
            'churn': ['accuracy', Precision(name='precision'), Recall(name='recall')],
            'score': 'accuracy',
            'balance': 'accuracy'
        }
    )

    callbacks = []
    if save_path:
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            monitor='val_churn_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)

    history = model.fit(
        X_train,
        y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_dict),
        verbose=1,
        callbacks=callbacks
    )

    best_epoch = int(np.argmax(history.history.get('val_churn_recall', history.history['val_churn_accuracy'])) + 1)
    print(f"✅ Best Epoch: {best_epoch}")

    # Load lại model tốt nhất
    custom_objects = {}
    if use_focal_churn:
        custom_objects['focal_loss_fixed'] = focal_loss_binary(alpha=0.75, gamma=2.0)
    if use_focal_score:
        custom_objects['focal_loss_fixed'] = focal_loss_categorical(alpha=0.25, gamma=2.0)
    if use_focal_balance:
        custom_objects['focal_loss_fixed'] = focal_loss_binary(alpha=0.75, gamma=2.0)

    best_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)

    y_pred = best_model.predict(X_test)
    y_true_dict = {
        'churn': y_test_dict['churn'],
        'score': y_test_dict['score'].argmax(axis=1),
        'balance': y_test_dict['balance']
    }
    y_pred_dict = {
        'churn': (y_pred[0] > 0.5).astype(int).flatten(),
        'score': y_pred[1].argmax(axis=1),
        'balance': (y_pred[2] > 0.5).astype(int).flatten()
    }

    # === Save report & plots ===
    task_metrics = []
    report_txt_path = os.path.join(save_dir, "test_classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"✅ Best Epoch (val_churn_recall): {best_epoch}\n\n")
        for task in ['churn', 'score', 'balance']:
            y_true = y_true_dict[task]
            y_pred_task = y_pred_dict[task]

            acc = accuracy_score(y_true, y_pred_task)
            prec = precision_score(y_true, y_pred_task, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred_task, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred_task, average='weighted', zero_division=0)
            cls_report = classification_report(y_true, y_pred_task)

            f.write(f"=== {task.upper()} ===\n")
            f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\n")
            f.write(cls_report + "\n\n")

            task_metrics.append({
                'Epochs': epochs,
                'Batch Size': batch_size,
                'Task': task,
                'Accuracy': round(acc, 4),
                'F1-score': round(f1, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'Model': model_name
            })

            plot_all(
                y_true=y_true,
                y_pred=y_pred_task,
                y_proba=y_pred[0 if task == 'churn' else 1 if task == 'score' else 2],
                task_name=task,
                save_dir=save_dir,
                model_name=model_name
            )

    print(f"📄 Saved test report and metrics to: {report_txt_path}")
    return history, task_metrics

def generate_simple_explanations(y_pred_churn, y_pred_score, y_pred_balance, metadata_df, max_samples=None):
    explanations = []
    for i in range(len(y_pred_churn)):
        if y_pred_churn[i] == 1:
            score_class = y_pred_score[i]
            balance_flag = y_pred_balance[i]
            geo = metadata_df.iloc[i]['Geography']
            age = metadata_df.iloc[i]['Age']

            reasons = []
            if score_class == 0:
                reasons.append("low credit score")
            elif score_class == 2:
                reasons.append("high credit score")

            reasons.append("high balance" if balance_flag == 1 else "low balance")
            explanation = f"Churn = 1 → {', '.join(reasons)} | Age: {age}, Country: {geo}"
        else:
            explanation = "No churn"
        explanations.append(explanation)
        if max_samples and len(explanations) >= max_samples:
            break
    return explanations

def save_predictions_with_explanations(y_churn, y_score, y_balance, metadata_df, output_path):
    explanations = generate_simple_explanations(y_churn, y_score, y_balance, metadata_df)
    df_exp = metadata_df.copy()
    df_exp['Pred_Exited'] = y_churn
    df_exp['Pred_ScoreClass'] = y_score
    df_exp['Pred_HighBalance'] = y_balance
    df_exp['Explanation'] = explanations
    df_exp.to_csv(output_path, index=False)
    print(f"✅ Saved explanations to: {output_path}")
    
def generate_churn_reason_statistics(explanation_csv_path, output_dir="."):
    df = pd.read_csv(explanation_csv_path)
    churn_df = df[df["Pred_Exited"] == 1]

    # Tách các nguyên nhân trong cột Explanation
    reasons_all = churn_df["Explanation"].str.extract(r"→ (.+) \|")[0]
    reasons_split = reasons_all.str.split(", ")

    # Đếm tần suất từng lý do
    from collections import Counter
    reason_counter = Counter()
    for reasons in reasons_split.dropna():
        reason_counter.update(reasons)

    reason_df = pd.DataFrame(reason_counter.items(), columns=["Reason", "Count"])
    reason_df = reason_df.sort_values("Count", ascending=False)

    # Lưu và vẽ biểu đồ
    csv_out = os.path.join(output_dir, "churn_reason_statistics.csv")
    fig_out = os.path.join(output_dir, "churn_reason_barplot.png")
    reason_df.to_csv(csv_out, index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=reason_df, x="Count", y="Reason", palette="viridis", hue="Reason", legend=False)
    # plt.title("Top Reasons for Predicted Churn")
    plt.xlabel("Count")
    plt.ylabel("Reason")
    
    
    plt.savefig(fig_out)
    plt.close()
    
    print(f"✅ Saved churn reason statistics to: {csv_out}")
    print(f"✅ Barplot saved to: {fig_out}")

  
def run_baseline_models(X_train, X_test, y_train, y_test, task_name, save_dir="."):
    from sklearn.preprocessing import label_binarize
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        # 'Random Forest': RandomForestClassifier(n_estimators=100),
        # 'Random Forest': RandomForestClassifier(
        #     n_estimators=50, 
        #     max_depth=5, 
        #     min_samples_split=10, 
        #     random_state=42
        # ),

        'MLP (Single)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, kernel='rbf'),
        # 'XGBoost': XGBClassifier(eval_metric='logloss')
        # 'XGBoost': XGBClassifier(
        #     eval_metric='logloss',
        #     n_estimators=50,
        #     max_depth=4,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     learning_rate=0.1,
        #     random_state=42
        # )
    }

    results = []

    # Nếu dữ liệu là one-hot encoded, chuyển về nhãn số
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)

    # Tạo thư mục riêng cho từng mô hình
    baseline_subdirs = {}
    for name in models.keys():
        subdir_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        subdir_path = os.path.join(save_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        baseline_subdirs[name] = subdir_path

    for name, model in models.items():
        print(f"▶️ Training baseline: {name} on task {task_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Dự đoán xác suất
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 1:
                # Thêm cột xác suất đối lập cho binary classification
                y_proba = np.hstack([1 - y_proba, y_proba])
        except AttributeError:
            # Nếu model không hỗ trợ predict_proba
            classes = np.unique(y_train)
            y_proba = np.zeros((len(y_pred), len(classes)))
            for i, label in enumerate(classes):
                y_proba[:, i] = (y_pred == label).astype(int)

        # ✅ Gọi plot_all
        plot_all(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            task_name=task_name,
            save_dir=baseline_subdirs[name],
            model_name=name.replace(" ", "_")
        )

        results.append({
            'Task': task_name,
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 3),
            'F1-score': round(f1_score(y_test, y_pred, average='weighted'), 3),
            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 3)
        })

    return results

# def plot_model_comparison(df_compare, metric='F1-score', save_dir="."):
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(data=df_compare, x='Task', y=metric, hue='Model')
#     for container in ax.containers:
#         ax.bar_label(container, fmt='%.2f', label_type='edge')
#     # plt.title(f'So sánh {metric} giữa các mô hình')
#     plt.ylabel(metric)

#     plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
#     plt.tight_layout()
    
#     file_path = os.path.join(save_dir, f'compare_{metric.lower()}.png')
#     plt.savefig(file_path)
#     plt.close()

def plot_model_comparison(df_compare, metric='F1-score', save_dir="."):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    num_bars = len(df_compare)
    fig_width = max(12, num_bars * 0.6)

    plt.figure(figsize=(fig_width, 6))
    ax = sns.barplot(data=df_compare, x='Task', y=metric, hue='Model')

    # Gắn nhãn trên từng cột, xoay dọc
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', rotation=90, padding=3)

    plt.ylabel(metric)
    plt.xlabel("Task")

    # Bỏ spines trên và phải, giữ trái
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Di chuyển legend ra ngoài và điều chỉnh lề
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.subplots_adjust(left=0.08, right=0.82, top=0.95, bottom=0.15)

    file_path = os.path.join(save_dir, f'compare_{metric.lower()}.png')
    plt.savefig(file_path)
    plt.close()



def select_best_model(result_csv="gridsearch_multitask_results.csv", metric='F1-score', save_dir="."):
    df = pd.read_csv(result_csv)
    grouped = df.groupby(['Epochs', 'Batch Size'])[metric].mean().reset_index()
    best_row = grouped.loc[grouped[metric].idxmax()]
    best_config = {
        'Epochs': int(best_row['Epochs']),
        'Batch Size': int(best_row['Batch Size']),
        f'Avg {metric}': round(best_row[metric], 3)
    }
    summary = (
        f"\n🏆 Best Configuration based on average {metric} across tasks:\n"
        f"  - Epochs: {best_config['Epochs']}\n"
        f"  - Batch Size: {best_config['Batch Size']}\n"
        f"  - Avg {metric}: {best_config[f'Avg {metric}']}\n"
    )
    print(summary)
    with open(os.path.join(save_dir, "best_model_summary.txt"), "w") as f:
        f.write(summary)
    return best_config

def create_directories(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def classify_alert(score):
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'

def run_churn_risk_alert_pipeline(filepath, result_folder):
    output_dir = os.path.join(result_folder, "churn_risk_alert")
    output_dir = create_directories(output_dir)
    # Load dataset
    df = pd.read_csv(filepath)

    # Encode categorical variables
    df['Geography'] = pd.factorize(df['Geography'])[0]
    df['Gender'] = pd.factorize(df['Gender'])[0]

    # Feature engineering
    df['CreditScoreClass'] = pd.cut(df['CreditScore'], bins=[0, 580, 700, 850], labels=[0, 1, 2]).astype(int)
    df['HighBalanceFlag'] = (df['Balance'] > 100000).astype(int)

    # Target variable: churn
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict churn probability
    probs = model.predict_proba(X_test)[:, 1]
    churn_risk_score = (probs * 100).astype(int)
    alert_levels = [classify_alert(score) for score in churn_risk_score]

    # Create alert log
    alert_df = X_test.copy()
    alert_df['Churn_Prob'] = probs
    alert_df['Churn_Risk_Score'] = churn_risk_score
    alert_df['Risk_Level'] = alert_levels

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    alert_log_path = os.path.join(output_dir, "churn_risk_alert_log.csv")
    alert_df.to_csv(alert_log_path, index=False)
    print(f"✅ Alert log saved to {alert_log_path}")

    # Evaluate performance
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    print(report)
    report_path = os.path.join(output_dir, "churn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📝 Classification report saved to {report_path}")

    # Plot alert level distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=alert_df, x='Risk_Level', order=['Low', 'Medium', 'High'], palette='Set2', hue='Risk_Level', legend=False)

    # plt.title("Churn Risk Level Distribution")
    plt.ylabel("Number of Customers")
    plt.xlabel("Risk Level")



    barplot_path = os.path.join(output_dir, "churn_risk_level_distribution.png")
    plt.savefig(barplot_path)
    print(f"📊 Saved {barplot_path}")

    # Generate recommendations for high risk customers
    generate_high_risk_recommendations(alert_log_path, os.path.join(output_dir, "high_risk_customers_with_reasons.csv"))

from itertools import product
import pandas as pd
import os

def generate_test_cases(
    batch_sizes=[32],
    epochs_list=[150],
    use_focal_churn_options=[False, True],
    use_focal_score_options=[False],
    use_focal_balance_options=[False],
    loss_weights_list=[(2.0, 0.5, 0.5)],
    save_path=None
):
    """
    Generate test case configurations for multitask DNN training.
    Each test case can control focal loss usage per task.

    Args:
        batch_sizes (list): List of batch sizes.
        epochs_list (list): List of epoch values.
        use_focal_churn_options (list): List of bools for focal loss on churn.
        use_focal_score_options (list): List of bools for focal loss on score.
        use_focal_balance_options (list): List of bools for focal loss on balance.
        loss_weights_list (list): List of (churn, score, balance) weight tuples.
        save_path (str): Optional .csv file path to save the test case configurations.

    Returns:
        List of dictionaries representing test case configurations.
    """
    test_cases = list(product(
        batch_sizes, epochs_list,
        use_focal_churn_options,
        use_focal_score_options,
        use_focal_balance_options,
        loss_weights_list
    ))

    labeled_test_cases = []
    for i, (bs, ep, focal_churn, focal_score, focal_balance, lw) in enumerate(test_cases):
        config = {
            "test_case_id": f"TC_{i:03d}",
            "batch_size": bs,
            "epochs": ep,
            "use_focal_churn": focal_churn,
            "use_focal_score": focal_score,
            "use_focal_balance": focal_balance,
            "loss_weights": {
                "churn": lw[0],
                "score": lw[1],
                "balance": lw[2]
            }
        }
        labeled_test_cases.append(config)

    # Optional CSV export
    if save_path:
        df_save = pd.DataFrame([
            {
                "test_case_id": tc["test_case_id"],
                "batch_size": tc["batch_size"],
                "epochs": tc["epochs"],
                "use_focal_churn": tc["use_focal_churn"],
                "use_focal_score": tc["use_focal_score"],
                "use_focal_balance": tc["use_focal_balance"],
                "churn_weight": tc["loss_weights"]["churn"],
                "score_weight": tc["loss_weights"]["score"],
                "balance_weight": tc["loss_weights"]["balance"]
            }
            for tc in labeled_test_cases
        ])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_save.to_csv(save_path, index=False)
        print(f"✅ Saved test case list to: {save_path}")

    return labeled_test_cases

from datetime import datetime

def generate_focal_comparison_report(result_csv_path, test_case_csv_path, output_txt_path, mode='w'):
    """
    Generate a focal vs non-focal comparison report based on gridsearch and test case configs.

    Args:
        result_csv_path (str): Path to the results .csv file.
        test_case_csv_path (str): Path to the test case definitions .csv file.
        output_txt_path (str): Path to output the report .txt file.
        mode (str): 'w' = overwrite, 'a' = append.
    """
    df_result = pd.read_csv(result_csv_path)
    df_testcase = pd.read_csv(test_case_csv_path)

    # Merge to get focal flags per model
    df = df_result.merge(df_testcase, left_on='Model', right_on='test_case_id', how='left')

    # Tổng hợp theo churn (nhiệm vụ chính)
    df_churn = df[df['Task'] == 'churn'].copy()
    df_churn["Score"] = df_churn["F1-score"] + df_churn["Recall"]

    # Tìm mô hình tốt nhất tổng thể
    best_row = df_churn.loc[df_churn["Score"].idxmax()]

    # Tổng hợp theo cấu hình focal
    pivot = df_churn.groupby(['use_focal_churn', 'use_focal_score', 'use_focal_balance'])[
        ['Accuracy', 'F1-score', 'Precision', 'Recall']
    ].mean().round(4).reset_index()

    # Ghi báo cáo
    with open(output_txt_path, mode) as f:
        f.write("🎯 Best Churn Model Based on F1 + Recall:\n")
        f.write(f"- Model ID: {best_row['Model']}\n")
        f.write(f"- Epochs: {best_row['Epochs']}, Batch Size: {best_row['Batch Size']}\n")
        f.write(f"- F1-score: {best_row['F1-score']}, Recall: {best_row['Recall']}\n")
        f.write(f"- Accuracy: {best_row['Accuracy']}, Precision: {best_row['Precision']}\n\n")
        f.write("📊 Average Performance by Focal Configuration (churn task only):\n")
        f.write(pivot.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n\n")

    return output_txt_path

def generate_focal_comparison_allTask_report(result_csv_path, test_case_csv_path, output_txt_path, mode='w'):
    """
    Generate a focal vs non-focal comparison report across all tasks.
    Prioritize churn (main task) but include score and balance as well.
    """

    df_result = pd.read_csv(result_csv_path)
    df_testcase = pd.read_csv(test_case_csv_path)

    # Merge test case metadata
    df = df_result.merge(df_testcase, left_on='Model', right_on='test_case_id', how='left')

    report_lines = []
    report_lines.append("📘 Focal vs Non-Focal Comparison Report (All Tasks)\n")

    # === CHURN: Main task ===
    df_churn = df[df['Task'] == 'churn'].copy()
    df_churn["Score"] = df_churn["F1-score"] + df_churn["Recall"]
    best_churn_row = df_churn.loc[df_churn["Score"].idxmax()]

    report_lines.append("🎯 Best Churn Model Based on F1 + Recall:")
    report_lines.append(f"- Model ID: {best_churn_row['Model']}")
    report_lines.append(f"- Epochs: {best_churn_row['Epochs']}, Batch Size: {best_churn_row['Batch Size']}")
    report_lines.append(f"- F1-score: {best_churn_row['F1-score']}, Recall: {best_churn_row['Recall']}")
    report_lines.append(f"- Accuracy: {best_churn_row['Accuracy']}, Precision: {best_churn_row['Precision']}\n")

    pivot_churn = df_churn.groupby(
        ['use_focal_churn', 'use_focal_score', 'use_focal_balance']
    )[['Accuracy', 'F1-score', 'Precision', 'Recall']].mean().round(4).reset_index()

    report_lines.append("📊 Average Performance by Focal Configuration (CHURN):")
    report_lines.append(pivot_churn.to_string(index=False))
    report_lines.append("\n" + "="*80 + "\n")

    # === SCORE: Auxiliary task ===
    df_score = df[df['Task'] == 'score'].copy()
    df_score["Score"] = df_score["F1-score"] + df_score["Recall"]
    best_score_row = df_score.loc[df_score["Score"].idxmax()]

    report_lines.append("📌 Best Score Model Based on F1 + Recall:")
    report_lines.append(f"- Model ID: {best_score_row['Model']}")
    report_lines.append(f"- Epochs: {best_score_row['Epochs']}, Batch Size: {best_score_row['Batch Size']}")
    report_lines.append(f"- F1-score: {best_score_row['F1-score']}, Recall: {best_score_row['Recall']}")
    report_lines.append(f"- Accuracy: {best_score_row['Accuracy']}, Precision: {best_score_row['Precision']}\n")

    pivot_score = df_score.groupby(
        ['use_focal_churn', 'use_focal_score', 'use_focal_balance']
    )[['Accuracy', 'F1-score', 'Precision', 'Recall']].mean().round(4).reset_index()

    report_lines.append("📊 Average Performance by Focal Configuration (SCORE):")
    report_lines.append(pivot_score.to_string(index=False))
    report_lines.append("\n" + "="*80 + "\n")

    # === BALANCE: Auxiliary task ===
    df_balance = df[df['Task'] == 'balance'].copy()
    df_balance["Score"] = df_balance["F1-score"] + df_balance["Recall"]
    best_balance_row = df_balance.loc[df_balance["Score"].idxmax()]

    report_lines.append("📌 Best Balance Model Based on F1 + Recall:")
    report_lines.append(f"- Model ID: {best_balance_row['Model']}")
    report_lines.append(f"- Epochs: {best_balance_row['Epochs']}, Batch Size: {best_balance_row['Batch Size']}")
    report_lines.append(f"- F1-score: {best_balance_row['F1-score']}, Recall: {best_balance_row['Recall']}")
    report_lines.append(f"- Accuracy: {best_balance_row['Accuracy']}, Precision: {best_balance_row['Precision']}\n")

    pivot_balance = df_balance.groupby(
        ['use_focal_churn', 'use_focal_score', 'use_focal_balance']
    )[['Accuracy', 'F1-score', 'Precision', 'Recall']].mean().round(4).reset_index()

    report_lines.append("📊 Average Performance by Focal Configuration (BALANCE):")
    report_lines.append(pivot_balance.to_string(index=False))
    report_lines.append("\n" + "="*80 + "\n")

    # === WRITE TO FILE ===
    with open(output_txt_path, mode) as f:
        f.write("\n".join(report_lines))

    print(f"✅ Extended focal comparison report saved to: {output_txt_path}")
    return output_txt_path


def main():
    directory_work = os.getcwd()
    model_name = "Optimization_TestCase_Customer_Churn_XAI_v1_3"
    result_folder = create_directories(os.path.join(directory_work, model_name))
    filepath = os.path.join(directory_work, 'Bank_Customer_Churn.csv')
    df = pd.read_csv(filepath)
    print(df['Exited'].value_counts(normalize=True))

    # Load và chuẩn hóa dữ liệu
    X, y_dict, df = load_clean_data_multitask(filepath)
    plot_label_distribution(y_dict, save_dir=result_folder, prefix="before")

    # Train-test split
    X_train_full, X_test, y_churn_train_full, y_churn_test, y_score_train_full, y_score_test, y_balance_train_full, y_balance_test = train_test_split(
        X, y_dict['churn'], y_dict['score'], y_dict['balance'], test_size=0.2, random_state=42)

    # Validation split
    X_train, X_val, y_churn_train, y_churn_val, y_score_train, y_score_val, y_balance_train, y_balance_val = train_test_split(
        X_train_full, y_churn_train_full, y_score_train_full, y_balance_train_full, test_size=0.2, random_state=42)

    y_train_dict = {'churn': y_churn_train, 'score': y_score_train, 'balance': y_balance_train}
    y_val_dict = {'churn': y_churn_val, 'score': y_score_val, 'balance': y_balance_val}
    y_test_dict = {'churn': y_churn_test, 'score': y_score_test, 'balance': y_balance_test}

    # Balancing
    X_train, y_train_dict = apply_balancing_all_tasks(X_train, y_train_dict)
    plot_label_distribution(y_train_dict, save_dir=result_folder, prefix="after")

    metadata_test = df.iloc[len(df) - len(X_test):].reset_index(drop=True)
    
    multitask_dir = create_directories(os.path.join(result_folder, "Multitask DNN"))
    baseline_dir = create_directories(os.path.join(result_folder, "Baselines"))
    report_dir = create_directories(os.path.join(result_folder, "reports"))
    
    # ✅ Tạo test cases
    test_cases = generate_test_cases(
        batch_sizes=[32],
        epochs_list=[100, 150],
        use_focal_churn_options=[True, False],
        use_focal_score_options=[False],
        use_focal_balance_options=[False],
        loss_weights_list=[(2.0, 0.5, 0.5), (1.5, 1.0, 1.0)],
        save_path=os.path.join(multitask_dir, "test_case_list.csv")
    )

    all_results = []
    model_paths = {}
  
    # Vòng lặp huấn luyện cho từng test case
    for config in test_cases:
        tc_id = config["test_case_id"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        loss_weights = config["loss_weights"]

        print(f"🚀 Training {tc_id}: epochs={epochs}, batch_size={batch_size}")
        config_dir = create_directories(os.path.join(multitask_dir, f"{tc_id}_ep{epochs}_bs{batch_size}"))
        model = build_multitask_dnn(X.shape[1])
        model_path = os.path.join(config_dir, f"model_{tc_id}.h5")

        history, test_metrics = train_multitask_model(
            model=model,
            X_train=X_train,
            y_train_dict=y_train_dict,
            X_val=X_val,
            y_val_dict=y_val_dict,
            X_test=X_test,
            y_test_dict=y_test_dict,
            epochs=epochs,
            batch_size=batch_size,
            loss_weights=loss_weights,
            use_focal_churn=config["use_focal_churn"],
            use_focal_score=config["use_focal_score"],
            use_focal_balance=config["use_focal_balance"],
            save_path=model_path,
            save_dir=config_dir,
            model_name=tc_id
        )

        plot_training_history(history, config_dir)
        model_paths[tc_id] = model_path
        all_results.extend(test_metrics)

    # Sau vòng lặp, lưu kết quả tổng hợp ra file
    df_all = pd.DataFrame(all_results)
    gridsearch_path = os.path.join(multitask_dir, "gridsearch_multitask_results.csv")
    df_all.to_csv(gridsearch_path, index=False)

    # Gọi báo cáo tổng hợp focal
    focal_report_path = os.path.join(report_dir, "focal_vs_nonfocal_comparison_report.txt")
    generate_focal_comparison_report(
        result_csv_path=gridsearch_path,
        test_case_csv_path=os.path.join(multitask_dir, "test_case_list.csv"),
        output_txt_path=focal_report_path,
        mode='w'
    )

    # Gọi báo cáo mở rộng cho cả 3 nhiệm vụ
    focal_alltask_report_path = os.path.join(report_dir, "focal_alltask_vs_nonfocal_comparison_report.txt")
    generate_focal_comparison_allTask_report(
        result_csv_path=gridsearch_path,
        test_case_csv_path=os.path.join(multitask_dir, "test_case_list.csv"),
        output_txt_path=focal_alltask_report_path,
        mode='w'
    )

    print(f"📄 Final focal vs non-focal summary saved to: {focal_report_path}")
    print(f"✅ Extended focal comparison report saved to: {focal_alltask_report_path}")

    # Chọn mô hình tốt nhất theo nhiệm vụ CHURN
    df_churn = df_all[df_all['Task'] == 'churn'].copy()
    df_churn["Score"] = df_churn["Recall"] + df_churn["F1-score"]
    best_row = df_churn.loc[df_churn["Score"].idxmax()]
    best_id = best_row["Model"]

    # ✅ Lấy lại config để xác định custom_objects
    config_best = [tc for tc in test_cases if tc["test_case_id"] == best_id][0]
    custom_objects = {}
    if config_best["use_focal_churn"]:
        custom_objects['focal_loss_fixed'] = focal_loss_binary(alpha=0.75, gamma=2.0)
    if config_best["use_focal_score"]:
        custom_objects['focal_loss_fixed'] = focal_loss_categorical(alpha=0.25, gamma=2.0)
    if config_best["use_focal_balance"]:
        custom_objects['focal_loss_fixed'] = focal_loss_binary(alpha=0.75, gamma=2.0)

    best_model = tf.keras.models.load_model(model_paths[best_id], custom_objects=custom_objects)
    best_model.summary()

    y_pred = best_model.predict(X_test)
    y_churn = (y_pred[0] > 0.5).astype(int).flatten()
    y_score = y_pred[1].argmax(axis=1)
    y_balance = (y_pred[2] > 0.5).astype(int).flatten()

    explanation_path = os.path.join(multitask_dir, "explanation_results.csv")
    save_predictions_with_explanations(y_churn, y_score, y_balance, metadata_test, explanation_path)
    generate_churn_reason_statistics(explanation_path, multitask_dir)

    # Baseline models
    baseline_results = []
    for task in ['churn', 'score', 'balance']:
        baseline_results += run_baseline_models(X_train, X_test, y_train_dict[task], y_test_dict[task], task, save_dir=baseline_dir)

    df_baselines = pd.DataFrame(baseline_results)
    df_baselines.to_csv(os.path.join(baseline_dir, "baseline_comparison.csv"), index=False)

    # So sánh tổng hợp của các mô hình
    df_combined = pd.concat([df_all, df_baselines], axis=0)
    for metric in ['Accuracy', 'F1-score', 'Precision', 'Recall']:
        plot_model_comparison(df_combined, metric, save_dir=result_folder)

    # Cảnh báo churn
    run_churn_risk_alert_pipeline(filepath, result_folder)
    plot_high_risk_reason_distribution(
        csv_path=os.path.join(result_folder, "churn_risk_alert", "high_risk_customers_with_reasons.csv"),
        output_path=os.path.join(result_folder, "churn_risk_alert", "high_risk_reason_barplot.png")
    )

    # sao chép các file cần thiết
    collect_files_for_report(result_folder=result_folder, report_dir=report_dir)
    print(f"📄 All important files are gathered in the report folder: {report_dir}")

if __name__ == '__main__':
    main()

