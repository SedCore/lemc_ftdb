import sys
import os
import logging
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor

### Suppress undesirable print statements temporarily
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


### Load the dataset
def load_dataset(dataset_name, subject_id=None):
    if dataset_name=="BCICIV_2a":
        raw_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_id) # all subjects
        # Check the sampling frequency consistency across all datasets included in the raw data
        sfreq = raw_dataset.datasets[0].raw.info['sfreq'] 
        assert all([ds.raw.info['sfreq'] == sfreq for ds in raw_dataset.datasets])

        # Preprocess the dataset
        print("Signal preprocessing...")
        preprocessors = [Preprocessor('pick', picks=['eeg'])] # only use eeg(stim channels must be removed)
        preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=1e-3))
        with SuppressPrint(): preprocessed_dataset = preprocess(raw_dataset, preprocessors=preprocessors)

        # Create windows in the dataset based on events
        trial_start_offset_seconds = -0.5 # 0.5s before the beginning of the MI task.
        trial_stop_offset_seconds = 0 # Right at the end of the MI task
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)
        print("Creating windows from events...")
        with SuppressPrint(): windows_dataset = create_windows_from_events(
                    preprocessed_dataset,
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=trial_stop_offset_samples,
                    window_size_samples=None,
                    window_stride_samples=None,
                    preload=True,
                    mapping=None
                )

        # Classification parameters
        n_channels = windows_dataset[0][0].shape[0] # 22
        n_times = windows_dataset[0][0].shape[1] # 1125
        n_classes = 4
        return windows_dataset, n_channels, n_times, n_classes, sfreq
    else:
        raise ValueError(f'Dataset {dataset_name} is unknown.')
