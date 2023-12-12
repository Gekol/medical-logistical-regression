import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def delete_records_with_missing_values(data):
    row_num, col_num = data.shape
    delete_flags = []
    for index, row in data.iterrows():
        null_count = data.loc[[index]].isna().sum().sum()
        if null_count >= (col_num // 10):
            delete_flags.append(True)
        else:
            delete_flags.append(False)
    data["delete_flag"] = delete_flags
    data = data.drop(data[data.delete_flag is True].index)
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
    data = data.drop(columns_to_be_deleted, axis=1)
    for col in columns_to_be_updated:
        data[col] = data[col].fillna(data[col].mean())
    return data


def binarize_quality_features(data, quality_features_columns):
    data = pd.get_dummies(data, columns=quality_features_columns, drop_first=True)
    return data


def split_data(data, target_column):
    y = data[target_column]
    x = data.drop(target_column, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.7, random_state=42
    )
    return x_train, x_test, y_train, y_test


def main():
    prefix, key = '/data', 'data.xlsx'
    data_location = f'{prefix}/{key}'
    # Read the data
    data = pd.read_excel(data_location)

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-column', type=str)
    args = parser.parse_args()

    # Drop records with missing values
    data = delete_records_with_missing_values(data)

    # Delete columns with too many nulls and update the remaining nulls with the column mean value
    data = process_columns_with_missing_values(data)

    # Binarization of the quality features
    quality_features_columns = [
        'группа', 'подгруппа', 'операция', 'стенокардия ФК', 'СН ФК', 'ЦАГ перетоки', 'КТ очаг ишемии'
    ]
    data = binarize_quality_features(data, quality_features_columns)

    # Split the data into train and test datasets
    target_column = args.target_column
    x_train, x_test, y_train, y_test = split_data(data, target_column)

    # Write the datasets into dataframes
    train_data_dir = "/opt/ml/train"
    pd.DataFrame(x_train).to_csv(
        f"{train_data_dir}/x_data.csv",
        header=True,
        index=False
    )
    pd.DataFrame(x_test).to_csv(
        f"{train_data_dir}/x_data.csv",
        header=True,
        index=False
    )
    test_data_dir = "/opt/ml/test"
    pd.DataFrame(y_train).to_csv(
        f"{test_data_dir}/y_data.csv",
        header=True,
        index=False
    )
    pd.DataFrame(y_test).to_csv(
        f"{test_data_dir}/y_data.csv",
        header=True,
        index=False
    )


if __name__ == "__main__":
    main()
