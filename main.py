
from data.dataloader import Dataset

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def mlp_classify(x_train, x_test, y_train, y_test):
    mlp_tanh = MLPClassifier(activation='tanh', solver='adam', alpha=1e-1,
                             hidden_layer_sizes=(10, 2), random_state=1,
                             warm_start=True, verbose=0)
    mlp_tanh.fit(x_train, y_train)
    y_pred = mlp_tanh.predict(x_test)
    print("======={} prediction report=======:".format(mlp_classify.__name__))
    print(metrics.confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))
    print(metrics.classification_report(y_test, y_pred, digits=3))


def random_forest(x_train, x_test, y_train, y_test):
    r_forest_gini = RandomForestClassifier(n_estimators=100, criterion='gini', max_features=None,
                                           min_samples_split=0.05, min_samples_leaf=0.001)
    r_forest_gini.fit(x_train, y_train)
    r_forest_gini_pred = r_forest_gini.predict(x_test)
    print("======={} prediction report=======:".format(random_forest.__name__))
    print(metrics.confusion_matrix(y_test.values.argmax(axis=1), r_forest_gini_pred.argmax(axis=1)))
    print(metrics.classification_report(y_test, r_forest_gini_pred, digits=3))


def prepare_data_within_domain(data_path):
    # ========  Prepare data  ========
    data = Dataset(filename=data_path, column_names=None)

    x_train, x_test, y_train, y_test = train_test_split(data.feature,
                                                        data.onehot_encoded_output,
                                                        test_size=0.3)

    return x_train, x_test, y_train, y_test


def prepare_data_across_domain(data_1, data_2):
    # ========  Prepare data  ========
    data_1 = Dataset(filename=data_1, column_names=None)
    data_2 = Dataset(filename=data_2, column_names=None)

    x_train, _, y_train, _ = train_test_split(data_1.feature,
                                              data_1.onehot_encoded_output,
                                              test_size=0.001)
    _, x_test, _, y_test = train_test_split(data_2.feature,
                                            data_2.onehot_encoded_output,
                                            test_size=0.999)

    return x_train, x_test, y_train, y_test


def main(data_all, data_1x1, data_4x3, data_9x10, cross_domain=False):
    # ========  Prepare data  ========
    if cross_domain:
        x_train, x_test, y_train, y_test = prepare_data_across_domain(data_1x1, data_4x3)
    else:
        x_train, x_test, y_train, y_test = prepare_data_within_domain(data_all)

    # ========  Scale data  ========
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # ========  MLP ========
    mlp_classify(x_train, x_test, y_train, y_test)

    # ========  Random forest ========
    random_forest(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    data_1x1 = "data//PV_Data_1x1.csv"
    data_4x3 = "data//PV_Data_4x3.csv"
    data_9x10 = "data//PV_Data_9x10.csv"
    data_all = "data//All_data.csv"
    main(data_all, data_1x1, data_4x3, data_9x10)
