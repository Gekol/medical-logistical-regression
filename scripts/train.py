import os
import json
import joblib
import cgi
from io import StringIO
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sagemaker_containers.beta.framework import encoders, worker


def input_fn(input_data, content_type):
    """Parse input data.

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    # cgi.parse_header extracts all arguments after ';' as key-value pairs
    # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
    # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
    content_type, params = cgi.parse_header(content_type.lower())

    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        return df
    else:
        raise NotImplementedError(
            f"content_type '{content_type}' not implemented!")


def predict_fn(input_data, model):
    return model.predict(input_data)


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


def output_fn(prediction, accept):
    """Format prediction output.

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    # cgi.parse_header extracts all arguments after ';' as key-value pairs
    # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
    # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
    accept, params = cgi.parse_header(accept.lower())

    if accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)

    elif accept == "application/json":
        # https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)

    else:
        raise NotImplementedError(f"accept '{accept}' not implemented!")


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularisation_parameter", type=int, default=1)
    args = parser.parse_args()

    # Read the train data
    base_dir = "/opt/ml/train"
    x_train = pd.read_csv(
        f"{base_dir}/x_train.csv",
    )
    y_train = pd.read_csv(
        f"{base_dir}/y_train.csv",
    )

    # Create and train the model
    model = LogisticRegression(solver='liblinear', C=args.regularisation_parameter).fit(x_train, y_train)
    model.fit(x_train, y_train)

    # Store tar.gz file
    print("Saving model")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))



if __name__ == "__main__":
    main()
