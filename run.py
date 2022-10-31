from time import time

from kontrast.experiment import *
from utils.io import get_dataset
from utils.result_analysis import generate_analysis
from utils.paths import *
from utils.gpu_util import get_free_gpu
free_gpu = get_free_gpu()

def read_config(yaml_path: str=config_path):
    """
    Read a experimental config under "config/", default "config/experiment.yaml".
    Args:
        yaml_path:      /path/to/the/yaml.
    Returns:
        dict
    """

    import yaml
    with open(yaml_path, 'r') as fin:
        configs = yaml.safe_load(fin)
    return configs

def build_experiments(config: dict) -> list:
    """
    Build experiment instances based on config dict.
    Args:
        config:     Input config dict.
    Returns:
        list[Experiment]
    """

    exps = []
    common_args = config['common_args']
    config = config['experiments']
    for k, v in config.items():
        for item in v:
            item.update(common_args)
            if 'omega' in item:
                item['omega'] = datetime.timedelta(minutes=item['omega'])
            if 'period' in item:
                item['period'] = datetime.timedelta(minutes=item['period'])

            if k == 'kontrast':
                exps.append(Experiment(device=free_gpu, **item))
    return exps

def run_experiment(dataset_name: str, experiment: Experiment):
    """
    Run a experiment using a built Experiment instance.
    Args:
        dataset_name:       Filename of the dataset file under "dataset/config/" (ext name excluded).
        experiment:         Experiment instance.

    Methods:
        Save the results to a csv file located in "result/csv/", columns:
            id:             "id" is a randomly generated attribute unifying each KPI.
            label:          Label of the KPI. 0 (normal) / 1 (erroneous).
            case_id:        Id of the software change case to which this KPI attach.
            case_label:     Label of this software change case. 0 (normal) / 1 (erroneous).
            "criterion(s)": Algorithm result(s) for this KPI.
        This result csv file will be aggregated and processed in next step.

        Time cost per KPI is also displayed under "dataset/config/", ending by "_avgtime.txt", shown in seconds per KPI.
    """

    experiment.train()
    t0 = time()
    result_df = experiment.test()
    t1 = time()
    total_time_cost = t1 - t0
    total_kpi_count = len(get_dataset(dataset_name))

    os.makedirs(csv_save_path, exist_ok=True)
    result_df.sort_values(by='id', inplace=True)
    result_df['id'] = result_df['id'].astype(int)
    result_df['label'] = result_df['label'].astype(int)
    result_df['case_id'] = result_df['case_id'].astype(int)
    result_df['case_label'] = result_df['case_label'].astype(int)
    result_df.to_csv(os.path.join(csv_save_path, f'{dataset_name}_{experiment}.csv'), index=False)

    with open(os.path.join(csv_save_path, f'{dataset_name}_{experiment}_avgtime.txt'), 'w') as fout:
        if total_kpi_count > 0:
            fout.write(f'{total_time_cost / total_kpi_count}\n')
        else:
            fout.write(f'0\n')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    configs = read_config()
    dataset_name = configs['common_args']['dataset_name']
    experiments = build_experiments(configs)

    print('Experiments: ')
    for exp in experiments:
        print(exp)
    print()

    while len(experiments) > 0:
        exp = experiments[0]
        run_experiment(dataset_name, exp)
        experiments.pop(0)

    generate_analysis()