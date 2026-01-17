"""
Load and split the raw dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR



def load_and_split_data(
        raw_path: str | Path = RAW_DATA_DIR, 
        output_path: str | Path = PROCESSED_DATA_DIR,
        seed: int = 123
):
    ''' Load raw data and split into train/val/test splits '''

    rawpath = Path(raw_path)
    outpath = Path(output_path)
    
    rawpath.mkdir(parents=True, exist_ok=True)
    outpath.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print('Start loading data .. ')
    raw_train_df = pd.read_csv(rawpath / 'train.csv')
    raw_train_df = raw_train_df.drop(columns=['id'])

    # Split into train/valid/test (80/10/10)
    n_observations = len(raw_train_df)

    valid_ratio = 0.1
    test_ratio = 0.1
    print(f'Train / Val / Test : {100*(1-valid_ratio-test_ratio)} / {100*valid_ratio} / {100*test_ratio}')

    n_valid = int(n_observations * valid_ratio)
    n_test = int(n_observations * valid_ratio)

    rng = np.random.default_rng(seed)   # For reproducible results

    # Sample valid split
    valid_split_df = raw_train_df.sample(n = n_valid, random_state=rng)
    train_split_df = raw_train_df.drop(valid_split_df.index)

    # Sample test split
    test_split_df = train_split_df.sample(n = n_test, random_state=rng)
    train_split_df = train_split_df.drop(test_split_df.index)

    print('Size of training   split : ', len(train_split_df))
    print('Size of validation split : ', len(valid_split_df))
    print('Size of test       split : ', len(test_split_df))    


    # Export train/val/test split
    train_split_df.to_csv(outpath / 'train_split.csv', index=False)
    valid_split_df.to_csv(outpath / 'valid_split.csv', index=False)
    test_split_df.to_csv(outpath / 'test_split.csv', index=False)

    return train_split_df, valid_split_df, test_split_df




if __name__ == '__main__':
    load_and_split_data()

