from __future__ import annotations
import numpy as np
from .time_span import TimeSpan

class TimeSeries:
    def __init__(self, timestamps: list, values: list, standardize: bool=True) -> None:
        """
        Args:
            timestamps:     List of 10-digit timestamps.
            values:         List of the Corresponding values.
            standardize:    Whether to standardize first.
        """

        self.timestamps = timestamps
        if self.timestamps and self.timestamps[0] > 1e11:
            self.timestamps = list(np.array(self.timestamps) / 1000)
        self.values = values
        self.check_data_valid()
        if standardize:
            self.standardize()

    def __len__(self) -> int:
        return len(self.timestamps)

    def check_data_valid(self):
        """
        Check the integrity of a time series.
        """

        n_times, n_values = len(self.timestamps), len(self.values)
        if n_times != n_values:
            raise Exception(
                'TimeSeries must have the same number of timestamps({}) and values({}).'.format(n_times, n_values)
            )

    @classmethod
    def merge(cls, a: TimeSeries, b: TimeSeries) -> TimeSeries:
        """
        Merge two TimeSeries.
        Args:
            a, b:   TimeSeries.
        Returns:
            TimeSeries
        """

        if not (a.timestamps[0] > b.timestamps[-1] or a.timestamps[-1] < b.timestamps[0]):
            raise Exception(
                'When merging, TimeSeries must not overlap.'
            )
        if a.timestamps[0] > b.timestamps[-1]:
            timestamps = b.timestamps + a.timestamps
            values = b.values + a.values
        else:
            timestamps = a.timestamps + b.timestamps
            values = a.values + b.values

        return TimeSeries(timestamps, values, standardize=False)

    @classmethod
    def extract(cls, a: TimeSeries, range: TimeSpan) -> TimeSeries:
        """
        Extract a selected segment by the time span.
        Args:
            a:          Input time series.
            range:      The time span.
        Returns:
            TimeSeries
        """

        range.start = max(range.start, a.timestamps[0])
        range.end = min(range.end, a.timestamps[-1])

        start_index = np.searchsorted(a.timestamps, range.start)
        end_index = np.searchsorted(a.timestamps, range.end)
        return TimeSeries(a.timestamps[start_index:end_index], a.values[start_index:end_index], standardize=False)

    def standardize(self):
        """
        Standardize the time series, making the majority of data located in [-1, 1].
        """

        q = np.percentile(self.values, 95)
        if q > 0:
            self.values = list(np.array(self.values) / q)
        self.values = list((np.array(self.values) - 0.5) * 2)
