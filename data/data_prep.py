import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import  train_test_split


def get_data(drug='cetuximab', folder='data/GDSC/', test_size=0.25, variance_thold = 0.1, seed=42):
    DATA_PATH_Y = os.path.join(folder, drug.lower() + "_response.tsv")
    DATA_PATH_X = os.path.join(folder, drug.lower() + "_exprs.tsv")

    df_y_label = pd.read_csv(DATA_PATH_Y, sep="\t", header=0)
    df_y_label.response = df_y_label.response.replace('R', 1)
    df_y_label.response = df_y_label.response.replace('S', 0)

    df_x_train = pd.read_csv(DATA_PATH_X, sep="\t", index_col=0, decimal=",")
    df_x_train = df_x_train[[str(c) for c in df_y_label.sample_name.to_list()]]
    df_x_train = pd.DataFrame.transpose(df_x_train)

    selector = VarianceThreshold(threshold=variance_thold)
    selector.fit_transform(df_x_train)
    df_x_train = df_x_train.iloc[:, selector.get_support(indices=True)]

    X_train, X_test, y_train, y_test = train_test_split(df_x_train,
                                                        df_y_label,
                                                        test_size=test_size,
                                                        random_state=seed)
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    # y_train = y_train.response.reset_index(drop=True)
    # y_test = y_test.response.reset_index(drop=True)

    return X_train, y_train, X_test, y_test