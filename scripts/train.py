import argparse
from sklearn.linear_model import LogisticRegression


def main():
    model = LogisticRegression(solver='liblinear', C=2).fit(X_train, y_train)

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--regularisation_parameter", type=int, default=1)
    X_train = pd.read_csv(
        f"{base_dir}/input/abalone-dataset.csv",
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    )
    y_train

    model = LogisticRegression(solver='liblinear', C=2).fit(X_train, y_train)



if __name__ == "__main__":
    main()
