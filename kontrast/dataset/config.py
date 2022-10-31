class DatasetConfig:
    def __init__(self,
                 K: int,
                 mode: str,
                 ignore: bool=True,
                 batch_size: int=1000,
                 dataset_size: int=10000):
        """
        Configs of dataset building.
        Args:
            K:                  Number of noise intensity levels. K in our paper.
            mode:               "LS" or "P", indicating this dataset is for an LS model or a P model.
            ignore:             Whether to ignore the ongoing period, default True.
            batch_size:         Batch size of the dataset.
            dataset_size:       Number of KPI time series segment pairs in the dataset to generate per label (positive/negative).
        """

        assert K in [1, 3, 5]
        self.K = K
        assert mode in ['LS', 'P']
        self.mode = mode
        self.ignore = ignore

        if self.K == 1:
            self.std_thres = [1]
        elif self.K == 3:
            self.std_thres = [0.005, 0.1, 1]
        elif self.K == 5:
            self.std_thres = [0.005, 0.03, 0.1, 0.3, 1]

        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def get_category(self, std: float) -> int:
        """
        Find the corresponding noise intensity category id by the standard deviation.
        Args:
            std:        Standard deviation of a KPI.
        Returns:
            int
        """

        for id, value in enumerate(self.std_thres):
            if value > std:
                return id
        return len(self.std_thres) - 1