# Kontrast

Kontrast is a self-supervised, generic and adaptive tool for identifying erroneous software changes.

## Dependencies

python 3.9

    $ pip install -r requirements.txt

## Usage

    $ python run.py

`run.py` is the entrance of the whole project.
It builds experiments based on the config file `config/experiment.yaml`.
Experiments are executed sequentially without parallel.

Kontrast will automatically find an applicable GPU to perform training and testing.
If none, it will be executed on the CPU.

After all the experiments are finished, a result aggregation is carried out, calculating the best F1-score metrics and collecting all the results into a report file `result/analysis/report.csv`.

### Use dataset A in our paper

Our default setup in `config/experiment.yaml` is the P model of the best configuration.
Simply run the code will start training and testing the model.

### Use your own dataset

By adding dataset meta file `dataset/config/DATASET.csv`, you can add a new dataset named `DATASET`.
Columns in the csv should be the same as in the default setup.
Detailed information can be found in the comments of `utils/io.py`.

For the raw KPI data in your new dataset, you should put all your KPIs under `dataset/data/DATASET/`, obeying the naming rule: `instance_name$metric.csv`.
Each KPI csv file should contain two columns: `timestamp` and `value`.
Data point missing is not permitted, thus please do preprocessing before running Kontrast.

To use `DATASET`, remember to modify the `dataset_name` argument in `experiment.yaml`.

### Aggregate the results of P and LS model

Please refer to `aggregate_results()` in `utils/result_analysis.py` for detailed usage.

### Get the results

After experiments, results can be found in `result/`.
Here are the contents in each folder:

- `analysis`: `report.csv` is the overall result records.
- `analysis_fig`: P-R curve and ROC curve visualization of each experiment.
- `csv`: Raw prediction results and time cost per KPI of each experiment.
- `exp_info`: Record of the arguments of experiment.

### Arguments

- `period`: The period of KPI time series in the dataset. $T$ in our paper. Note period in our project is a TimeDelta.
- `omega`: $\omega$ in our paper, length of the inspection window. Note omega in our project is a TimeDelta.
- `mode`: "P" or "LS", indicating the model in the experiment is a P model or an LS model.
- `K`: Number of the categories of noise intensity classifier.
- `dataset_size`: Number of KPI time series segment pairs in the dataset to generate per label (positive/negative). For example, if `dataset_size=40000`, we will generate 40000 positive pairs and 40000 negative pairs.
- `epoch`: Epochs to train.
- `lstm_hidden_dim`: The hidden dimension of LSTM.
- `lstm_output_dim`: The output dimension of LSTM.
- `lr`: Learning rate.
- `from_ckpt` (Optional): Whether to start from a checkpoint (name_stamp of another experiment). If so, we only test the model.
- `batch_size` (Optional): Batch size of the dataset.
- `rate` (Optional): Sampling interval of the dataset (unit: second). For example, if the monitor service collects data once per minute, `rate=60`.
- `ignore` (Optional): Whether to ignore the ongoing period in LS model, default True. No effect in P model experiments.

Sample:
```yaml
common_args:
  period: 1440
  dataset_name: aiops2018

experiments:
  kontrast:
    - omega: 120
      mode: 'P'
      K: 5
      dataset_size: 40000
      epoch: 100
      batch_size: 10000
      lstm_hidden_dim: 30
      lstm_output_dim: 30
      lr: 0.001
```

## Project Structure

- `config`: The experiment configs.
- `dataset`: Dataset meta files and raw KPI data.
- `kontrast`: Algorithm.
  - `dataset/`: Data preprocess, training data generation, pair selection, noise injection.
  - `config.py`: Experiment config.
  - `experiment.py`: Experiment implementation.
  - `model.py`: Training and testing phase implementation of Kontrast.
  - `nn.py`: Neural network implementation.
- `models`: Useful basic models.
- `utils`: Tools that are used by Kontrast.