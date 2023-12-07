import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--target-column', type=str)

def delete_records_with_missing_values(data):
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


def process_columns_with_missing_values(data):
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


def binarize_quality_features(data):
    data = pd.get_dummies(data, columns=quality_features_columns, drop_first=True)
    return data


def split_data(data):
    Y = data[TARGET_COLUMN]
    X = data.drop(TARGET_COLUMN, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.7, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prefix, key = '/data', 'data.xlsx'
    data_location = f'{prefix}/{key}'
    # Read the data
    data = pd.read_excel(data_location)
    target_column = args.target_column
    quality_features_columns = [
        'группа', 'подгруппа', 'операция', 'стенокардия ФК', 'СН ФК', 'ЦАГ перетоки', 'КТ очаг ишемии'
    ]

    # Drop records with missing values
    data = delete_records_with_missing_values(data)

    # Delete columns with too many nulls and update the remaining nulls with the column mean value
    data = process_columns_with_missing_values(data)

    # Binarization of the quality features
    data = binarize_quality_features(data)

    # Split the data into train and test datasets
    X_train, X_test, y_train, y_test = split_data(data)

    # Write the datasets into dataframes
    base_dir = "/opt/ml/train"
    pd.DataFrame(X_train).to_csv(
        f"{base_dir}/X_train.csv",
        header=True,
        index=False
    )
    pd.DataFrame(X_test).to_csv(
        f"{base_dir}/X_test.csv",
        header=True,
        index=False
    )
    pd.DataFrame(y_train).to_csv(
        f"{base_dir}/y_train.csv",
        header=True,
        index=False
    )
    pd.DataFrame(y_test).to_csv(
        f"{base_dir}/y_test.csv",
        header=True,
        index=False
    )