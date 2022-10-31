import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import yaml
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import glob
import tqdm
import matplotlib.pyplot as plt

from utils.paths import *

fig = None
ax = None

def analysis(csv_path: str) -> dict:
    """
    Analysis and visualize a result csv.
    Args:
        csv_path:       /path/to/the/csv
    Methods:
        Aggregate KPI-level result to change case-level result by the 95th percentile.
        Calculate the best F1-score and the AUC.
        Draw P-R curve and ROC curve for KPI-level and change case-level results under "result/analysis_fig/".
    """

    global fig, ax
    exp_name = os.path.splitext(os.path.basename(csv_path))[0]

    result_df = pd.read_csv(csv_path)
    cols = result_df.columns
    label = result_df['label']
    dist_names = cols.values[4:]

    ax_roc = ax[0][0]
    ax_pr = ax[1][0]
    ax_roc_case = ax[0][1]
    ax_pr_case = ax[1][1]
    ax_roc.cla()
    ax_pr.cla()
    ax_roc_case.cla()
    ax_pr_case.cla()

    ax_roc.set_xlim((-0.1, 1.1))
    ax_roc.set_ylim((-0.1, 1.1))
    ax_roc.grid(color='gray', alpha=0.4)
    ax_pr.set_xlim((-0.1, 1.1))
    ax_pr.set_ylim((-0.1, 1.1))
    ax_pr.grid(color='gray', alpha=0.4)
    ax_roc_case.set_xlim((-0.1, 1.1))
    ax_roc_case.set_ylim((-0.1, 1.1))
    ax_roc_case.grid(color='gray', alpha=0.4)
    ax_pr_case.set_xlim((-0.1, 1.1))
    ax_pr_case.set_ylim((-0.1, 1.1))
    ax_pr_case.grid(color='gray', alpha=0.4)

    result_dict = {}
    for dist_name in dist_names:
        pred = result_df[dist_name].to_numpy().copy()
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0
        fpr, tpr, thres = roc_curve(label, pred, pos_label=1)
        auc_value = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{dist_name} - AUC:{auc_value:.4f}')
        result_dict[f'{dist_name} AUC'] = auc_value

        prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
        f1_scores = (2 * prec * recall) / (prec + recall)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        result_dict[f'{dist_name} F1'] = best_f1_score
        result_dict[f'{dist_name} P'] = prec[best_f1_score_index]
        result_dict[f'{dist_name} R'] = recall[best_f1_score_index]
        result_dict[f'{dist_name} thres'] = thres[best_f1_score_index]
        ax_pr.plot(prec, recall, label=f'{dist_name} - F1:{best_f1_score:.4f}')

    cases = result_df.groupby('case_id')
    # [case_id, case_label, dists...]
    case_results = []
    for c in cases:
        df = c[1]
        res = [int(df.iloc[0]['case_label'])]
        for dist_name in dist_names:
            d = np.array(df[[dist_name]])
            # The score of a software change case is the 95th percentile of the ones of its related KPIs.
            res.append(np.nanpercentile(d, 95))
        case_results.append(res)
    case_results = np.array(case_results)

    label = case_results[:, 0]
    for id, dist_name in enumerate(dist_names):
        pred = case_results[:, id+1].copy()
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0
        fpr, tpr, thres = roc_curve(label, pred, pos_label=1)
        auc_value = auc(fpr, tpr)
        ax_roc_case.plot(fpr, tpr, label=f'{dist_name} - AUC:{auc_value:.4f}')
        result_dict[f'{dist_name} case AUC'] = auc_value

        prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
        f1_scores = (2 * prec * recall) / (prec + recall)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        result_dict[f'{dist_name} case F1'] = best_f1_score
        result_dict[f'{dist_name} case P'] = prec[best_f1_score_index]
        result_dict[f'{dist_name} case R'] = recall[best_f1_score_index]
        result_dict[f'{dist_name} case thres'] = thres[best_f1_score_index]
        ax_pr_case.plot(prec, recall, label=f'{dist_name} - F1:{best_f1_score:.4f}')

    yaml_path = os.path.join(exp_info_save_path, f'{exp_name[exp_name.find("_") + 1:]}.yaml')
    exp_info = '\n'
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as fin:
            exp_info_dict = yaml.safe_load(fin)
        for k, v in exp_info_dict.items():
            exp_info += f'{k}: {v}\n'

    fig.suptitle(f'Experiment: {exp_name}{exp_info}')

    ax_roc.set_title(f'kpi ROC figure')
    ax_roc.legend()
    ax_roc.set_xlabel('FP rate')
    ax_roc.set_ylabel('TP rate')

    ax_pr.set_title(f'kpi P-R figure')
    ax_pr.legend()
    ax_pr.set_xlabel('Precision')
    ax_pr.set_ylabel('Recall')

    ax_roc_case.set_title(f'case ROC figure')
    ax_roc_case.legend()
    ax_roc_case.set_xlabel('FP rate')
    ax_roc_case.set_ylabel('TP rate')

    ax_pr_case.set_title(f'case P-R figure')
    ax_pr_case.legend()
    ax_pr_case.set_xlabel('Precision')
    ax_pr_case.set_ylabel('Recall')

    fig.tight_layout()
    os.makedirs(analysis_fig_path, exist_ok=True)
    plt.savefig(os.path.join(analysis_fig_path, f'{exp_name}.jpg'))

    return result_dict

def generate_analysis():
    """
    Process all the result csv files.
    Collect the results of all the experiments and save them to "result/analysis/report.csv".
    """

    global fig, ax
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    result_df = pd.DataFrame(columns=['experiment', 'criterion', 'value'])
    for result_csv in tqdm.tqdm(glob.glob(f'{csv_save_path}/*.csv')):
        analy_res = analysis(result_csv)
        for k, v in analy_res.items():
            result_df.loc[len(result_df)] = [os.path.splitext(os.path.basename(result_csv))[0], k, v]
    os.makedirs(os.path.dirname(analysis_report_path), exist_ok=True)
    result_df.to_csv(analysis_report_path, index=False, float_format='%.3f', sep=',')

    csv_set = set()
    for result_csv in glob.glob(f'{csv_save_path}/*.csv'):
        f = os.path.splitext(os.path.basename(result_csv))[0]
        f = f[f.find('_')+1:]
        csv_set.add(f)

def aggregate_results(a: str, b: str, alpha: float=2.5):
    """
    Aggregate a P model result and an LS model result.
    Args:
        a:      Filename (*.csv) of the P model result.
        b:      Filename (*.csv) of the LS model result.
        alpha:  Hyper-parameter alpha in our paper.
    """

    def get_information(a: str) -> tuple:
        a = os.path.join(csv_save_path, a)

        result_df = pd.read_csv(a)
        cols = result_df.columns
        label = result_df['label']
        dist_names = cols.values[4:]
        dist_name = dist_names[0]

        pred = result_df[dist_name].to_numpy().copy()
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0

        prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
        f1_scores = (2 * prec * recall) / (prec + recall)

        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])

        return best_f1_score, pred

    def get_case_information(a: str) -> tuple:
        a = os.path.join(csv_save_path, a)

        result_df = pd.read_csv(a)
        cols = result_df.columns
        dist_names = cols.values[4:]
        cases = result_df.groupby('case_id')
        # [case_id, case_label, dists...]
        case_results = []
        for c in cases:
            df = c[1]
            res = [int(df.iloc[0]['case_label'])]
            for dist_name in dist_names:
                d = np.array(df[[dist_name]])
                res.append(np.nanpercentile(d, 95))
            case_results.append(res)
        case_results = np.array(case_results)

        label = case_results[:, 0]
        pred = case_results[:, 1]

        prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
        f1_scores = (2 * prec * recall) / (prec + recall)

        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        return best_f1_score, pred, label

    a_f1, a_pred = get_information(a)
    b_f1, b_pred = get_information(b)
    pred = np.array(a_pred + b_pred * alpha, dtype=float)

    df = pd.read_csv(os.path.join(csv_save_path, a))
    label = df['label']
    prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
    f1_scores = (2 * prec * recall) / (prec + recall)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    p = prec[best_f1_score_index]
    r = recall[best_f1_score_index]
    print(f'KPI: {a_f1:.3f} {b_f1:.3f} -> {best_f1_score:.3f} {p:.3f} {r:.3f}')

    a_c_f1, a_c_pred, _ = get_case_information(a)
    b_c_f1, b_c_pred, label = get_case_information(b)
    pred = np.array(a_c_pred + b_c_pred * 0.1, dtype=float)
    prec, recall, thres = precision_recall_curve(label, pred, pos_label=1)
    f1_scores = (2 * prec * recall) / (prec + recall)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    p = prec[best_f1_score_index]
    r = recall[best_f1_score_index]
    print(f'case: {a_c_f1:.3f} {b_c_f1:.3f} -> {best_f1_score:.3f} {p:.3f} {r:.3f}')

if __name__ == '__main__':
    generate_analysis()