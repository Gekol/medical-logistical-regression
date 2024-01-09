import wandb
import pandas as pd
from typing import List
from joblib import dump, load
from sklearn.model_selection import train_test_split


def create_table(artifact, data: pd.DataFrame, column_names: List[str], table_name: str):
    """Create wandb table from dataframe and save it in an artifact"""
    table = wandb.Table(columns=column_names)
    for row in list(data.itertuples(index=False, name=None)):
        table.add_data(*row)
    artifact.add(table, table_name)


def drop_records_with_many_nulls(data: pd.DataFrame):
    """Drop records with values that have too many nulls"""
    row_num, col_num = data.shape

    delete_flags = []
    for index, row in data.iterrows():
        null_count = data.loc[[index]].isna().sum().sum()
        if null_count >= (col_num // 10):
            delete_flags.append(True)
        else:
            delete_flags.append(False)
    data["delete_flag"] = delete_flags
    data = data.drop(data[data.delete_flag == True].index)
    data = data.drop(['delete_flag'], axis=1)
    return data


def update_columns_with_nulls(data: pd.DataFrame):
    columns_to_be_deleted = []
    columns_to_be_updated = []
    n = len(data)
    for series_name, series in data.items():
        if series.isna().sum() >= n // 10:
            columns_to_be_deleted.append(series_name)
        elif series.isna().sum() > 0:
            columns_to_be_updated.append(series_name)
    # print(columns_to_be_deleted)
    # print(columns_to_be_updated)
    data = data.drop(columns_to_be_deleted, axis=1)
    for col in columns_to_be_updated:
        data[col] = data[col].fillna(data[col].mean())
    return data


def split_data(data: pd.DataFrame, target_column: str, test_size: float):
    y = data[[target_column]]
    x = data.drop(target_column, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y[target_column],
        test_size=test_size
    )

    return x_train, x_test, pd.Series(y_train), pd.Series(y_test)


def separate_x_from_y(data, target_column):
    y = data[target_column]
    x = data.drop(target_column, axis=1)
    return x, y


def save_model(model, prefix, artifact, model_name):
    path = f"{prefix}/{model_name}.joblib"
    serialized_model = dump(model, path)
    artifact.add_file(path, model_name)
    return artifact

