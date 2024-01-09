import os
# import json
import joblib
# import cgi
# from io import StringIO
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector


# from sagemaker_containers.beta.framework import encoders, worker


# def input_fn(input_data, content_type):
#     """Parse input raw_data.

#     We currently only take csv input. Since we need to process both labelled
#     and unlabelled raw_data we first determine whether the label column is present
#     by looking at how many columns were provided.
#     """
#     # cgi.parse_header extracts all arguments after ';' as key-value pairs
#     # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
#     # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
#     content_type, params = cgi.parse_header(content_type.lower())

#     if content_type == 'text/csv':
#         # Read the raw input raw_data as CSV.
#         df = pd.read_csv(StringIO(input_data), header=None)
#         return df
#     else:
#         raise NotImplementedError(
#             f"content_type '{content_type}' not implemented!")


# def predict_fn(input_data, model):
#     return model.predict(input_data)


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


# def output_fn(prediction, accept):
#     """Format prediction output.

#     The default accept/content-type between containers for serial inference is JSON.
#     We also want to set the ContentType or mimetype as the same value as accept so the next
#     container can read the response payload correctly.
#     """
#     # cgi.parse_header extracts all arguments after ';' as key-value pairs
#     # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
#     # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
#     accept, params = cgi.parse_header(accept.lower())

#     if accept == 'text/csv':
#         return worker.Response(encoders.encode(prediction, accept), mimetype=accept)

#     elif accept == "application/json":
#         # https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html
#         instances = []
#         for row in prediction.tolist():
#             instances.append({"features": row})

#         json_output = {"instances": instances}

#         return worker.Response(json.dumps(json_output), mimetype=accept)

#     else:
#         raise NotImplementedError(f"accept '{accept}' not implemented!")

def train(x_train, y_train, n_features_to_select):
    # Find the regularisation parameter based on the train parameter
    regularisation_tests = {
        i: LogisticRegression(
            solver='liblinear', C=i
        ).fit(x_train, y_train).score(x_train, y_train)
        for i in range(1, 1001)
    }
    tests_results = pd.DataFrame({
        'C': regularisation_tests.keys(),
        'score': regularisation_tests.values()
    })
    max_preciseness = tests_results['score'].max()
    regularisation_parameter = tests_results.loc[
        tests_results['score'] == max_preciseness
        ]['C'].iloc[0]

    model = LogisticRegression(
        solver='liblinear', C=regularisation_parameter
    )
    feature_selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features_to_select,
        direction='backward',
        scoring='log_loss'
    )
    new_x_train = feature_selector.transform(x_train)

    new_trained_model = model.fit(new_x_train, y_train)

    return new_trained_model


def main():
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments.
    # Defaults are set in the environment variables.
    parser.add_argument(
        '--output-raw_data-dir',
        type=str,
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ['SM_CHANNEL_TRAIN']
    )

    # Parse hyperparameters
    parser.add_argument(
        "--n_features_to_select",
        type=int,
        default=3
    )

    args = parser.parse_args()

    # Read the train raw_data
    x_train = pd.read_csv(
        f"{args.train}/x_train.csv",
    )
    y_train = pd.read_csv(
        f"{args.train}/y_train.csv",
    )

    # Read the test raw_data
    x_test = pd.read_csv(
        f"{args.train}/x_test.csv",
    )
    y_test = pd.read_csv(
        f"{args.train}/y_test.csv",
    )

    # Create and train the model
    train_score, test_score = 0, 0
    trained_model = None
    while train_score <= 0.5 and test_score <= 0.5:
        trained_model = train(x_train, y_train, args.n_features_to_select)
        train_score, test_score = trained_model.score(x_train, y_train), trained_model.score(x_test, y_test)

    # Store tar.gz file
    print("Saving model")
    joblib.dump(trained_model, os.path.join(args.model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
