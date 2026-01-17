"""
    Inference Pipeline

    - Take raw input
    - Apply transformations and encoders (saved in model directory)
    - Return predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import pickle
from src.config.paths import MODEL_DIR, PROCESSED_DATA_DIR
from src.feature.eda_data_cleaning import drop_duplicates, drop_loss_outlier
from src.feature.build_train_features import transform_rank


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = MODEL_DIR,
    pro_path: Path | str = PROCESSED_DATA_DIR
):
    """ Return predictions produced by the final model """

    modelpath = Path(model_path)
    propath = Path(pro_path)

    # 1. Preprocess input
    input_df = drop_duplicates(input_df)

    # 2. Add features and encodings
    with open(modelpath / "retrained" / "retrain_feature_engineer_dict.pkl", "rb") as f:
        feature_dict = pickle.load(f)

    summary_cat_df = pd.read_csv(propath / "summary_cat.csv", index_col=0)
    cat_cols = [col for col in input_df.columns if col.startswith('cat')]
    cont_cols = [col for col in input_df.columns if col.startswith('cont')]

    # Frequency encoding
    cat_counts = feature_dict["cat_counts"]
    for cat, count in zip(cat_cols, cat_counts):
        input_df[f'{cat}_freq'] = input_df[cat].map(count).fillna(0)
        input_df[f'{cat}_log_freq'] = np.log1p(input_df[f'{cat}_freq'])     # take log to compress extreme counts
        input_df[f'{cat}_norm_freq'] = input_df[f'{cat}_freq'] / count.sum()

    # Ordinal encoding
    ordinal_encoder = feature_dict["ordinal_encoder"]
    input_df[cat_cols] = ordinal_encoder.transform(input_df[cat_cols])

    # Onehot encoder
    onehot_encoder = feature_dict["onehot_encoder"]
    low_card_cols = summary_cat_df[summary_cat_df['n_cats'] <= 3].index.to_list()
    input_onehot_encoded = onehot_encoder.transform(input_df[low_card_cols])
    input_onehot_df = pd.DataFrame(input_onehot_encoded, columns=onehot_encoder.get_feature_names_out(low_card_cols))
    input_df = pd.concat([input_df, input_onehot_df], axis=1)

    # Group stats
    groups = feature_dict["groups"]
    cont_stats = feature_dict["cont_stats"]
    for cat, group in zip(cat_cols, groups):
        for cont in cont_cols:
            global_mean, global_med, global_std = cont_stats[cont][0], cont_stats[cont][1], cont_stats[cont][2]
            input_df[f'{cat}_{cont}_mean'] = input_df[cat].map(group[cont].mean()).fillna(global_mean)
            input_df[f'{cat}_{cont}_med'] = input_df[cat].map(group[cont].median()).fillna(global_med)
            input_df[f'{cat}_{cont}_std'] = input_df[cat].map(group[cont].std()).fillna(global_std)

    # Numeric transformations
    uniq = feature_dict["uniq"]
    ranks = feature_dict["ranks"]
    l, u = feature_dict["low_bound"], feature_dict["up_bound"]
    for cont in cont_cols:
        # Log transform
        input_df[f'log_{cont}'] = np.log1p(input_df[cont])
        # Rank transform
        input_df[f'{cont}_rank'] = transform_rank(input_df[cont], uniq, ranks)
        # Winsorization
        input_df[f'{cont}_cap'] = input_df[cont].clip(l, u)


    # 3. Predict
    model = lgb.Booster(model_file = modelpath / "retrained"/ "final_lgb.txt")
    log_loss_pred = model.predict(input_df)
    loss_pred = np.exp(log_loss_pred)

    print(f"Loss prediction: {loss_pred}")

    return loss_pred



if __name__ == "__main__":
    predict()
    

