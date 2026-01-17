"""
    EDA and Data Cleaning
"""


import pandas as pd
import numpy as np
from pathlib import Path
from src.config.paths import PROCESSED_DATA_DIR 



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    ''' Drop indentical rows '''
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns, keep='first')
    after = df.shape[0]
    print(f'Dropped {after - before} rows')
    return df



def summarize_cat(df: pd.DataFrame, output_path: Path | str = PROCESSED_DATA_DIR):
    ''' Summarize categorical columns '''
    cat_cols = [col for col in df.columns if col.startswith('cat')]
    summary_cat_df = pd.DataFrame(
        {
        "n_cats" : df[cat_cols].nunique(),
        "most_freq_cat" : df[cat_cols].agg(lambda x: x.value_counts().idxmax()),
        "most_freq_count" : df[cat_cols].agg(lambda x: x.value_counts().max())
        },
        index = cat_cols
    )
    outpath = Path(output_path)
    summary_cat_df.to_csv(outpath / "summary_cat.csv")
    return summary_cat_df



def drop_loss_outlier(df: pd.DataFrame) -> pd.DataFrame:
    ''' Drop potential outliers of loss '''
    log_loss = np.log(df['loss'])
    outlier_id = df[log_loss < 0].index
    print(f'Log_loss outlier id : {outlier_id}')
    print('Train set size before removing outliers: ', len(df))
    df = df.drop(outlier_id)
    print('Train set size after removing outliers: ', len(df))
    return df



def run_eda_cleaning(
    splits: tuple[str] = ('train', 'valid', 'test'),
    pro_path: Path | str = PROCESSED_DATA_DIR
):
    propath = Path(pro_path)
    propath.mkdir(parents=True, exist_ok=True)

    cleaned_splits = []
    summary_cat_df = None

    for split in splits:
        split_df = pd.read_csv(propath / f'{split}_split.csv')
        # Eligible action on all splits
        split_df = drop_duplicates(split_df)

        # Only for train split
        if split == 'train':
            summary_cat_df = summarize_cat(split_df, output_path=propath)
            split_df = drop_loss_outlier(split_df)

        cleaned_splits.append(split_df)

        # Export processed splits
        split_df.to_csv(propath / f'cleaned_{split}.csv', index=False)

    return cleaned_splits[0], cleaned_splits[1], cleaned_splits[2], summary_cat_df



if __name__ == "__main__":
    run_eda_cleaning()


