from typing import Union
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error


# https://github.com/lachhebo/pyclustertend/blob/master/pyclustertend/hopkins.py
def hopkins(data_frame: Union[np.ndarray, pd.DataFrame], sampling_size: int) -> float:
    """Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express
    no clusterability and a score tending to 0 express a high cluster tendency.

    Examples
    --------
    >>> from sklearn import datasets
    >>> from pyclustertend import hopkins
    >>> X = datasets.load_iris().data
    >>> hopkins(X,150)
    0.16
    """

    if type(data_frame) == np.ndarray:
        data_frame = pd.DataFrame(data_frame)

    data_frame_sample = sample_observation_from_dataset(data_frame, sampling_size)

    sample_distances_to_nearest_neighbours = get_distance_sample_to_nearest_neighbours(
        data_frame, data_frame_sample
    )

    uniformly_selected_observations_df = simulate_df_with_same_variation(
        data_frame, sampling_size
    )

    df_distances_to_nearest_neighbours = get_nearest_sample(
        data_frame, uniformly_selected_observations_df
    )

    x = sum(sample_distances_to_nearest_neighbours)
    y = sum(df_distances_to_nearest_neighbours)

    if x + y == 0:
        raise Exception("The denominator of the hopkins statistics is null")

    return x / (x + y)[0]


def get_nearest_sample(df: pd.DataFrame, uniformly_selected_observations: pd.DataFrame):
    tree = BallTree(df, leaf_size=2)
    dist, _ = tree.query(uniformly_selected_observations, k=1)
    uniformly_df_distances_to_nearest_neighbours = dist
    return uniformly_df_distances_to_nearest_neighbours


def simulate_df_with_same_variation(
    df: pd.DataFrame, sampling_size: int
) -> pd.DataFrame:
    max_data_frame = df.max()
    min_data_frame = df.min()
    uniformly_selected_values_0 = np.random.uniform(
        min_data_frame[0], max_data_frame[0], sampling_size
    )
    uniformly_selected_values_1 = np.random.uniform(
        min_data_frame[1], max_data_frame[1], sampling_size
    )
    uniformly_selected_observations = np.column_stack(
        (uniformly_selected_values_0, uniformly_selected_values_1)
    )
    if len(max_data_frame) >= 2:
        for i in range(2, len(max_data_frame)):
            uniformly_selected_values_i = np.random.uniform(
                min_data_frame[i], max_data_frame[i], sampling_size
            )
            to_stack = (uniformly_selected_observations, uniformly_selected_values_i)
            uniformly_selected_observations = np.column_stack(to_stack)
    uniformly_selected_observations_df = pd.DataFrame(uniformly_selected_observations)
    return uniformly_selected_observations_df


def get_distance_sample_to_nearest_neighbours(df: pd.DataFrame, data_frame_sample):
    tree = BallTree(df, leaf_size=2)
    dist, _ = tree.query(data_frame_sample, k=2)
    data_frame_sample_distances_to_nearest_neighbours = dist[:, 1]
    return data_frame_sample_distances_to_nearest_neighbours


def sample_observation_from_dataset(df, sampling_size: int):
    if sampling_size > df.shape[0]:
        raise Exception("The number of sample of sample is bigger than the shape of D")
    data_frame_sample = df.sample(n=sampling_size)
    return data_frame_sample



class RsaGaussianNB:
    
    def __init__(self, X :pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

        self.C = 1
        self.cv_results = pd.DataFrame()

        # specify range of hyperparameters
        # Set the parameters by cross-validation
        self.hyper_params = [
                        {
                            'priors': [None],
                            'var_smoothing': [0.00000001, 0.000000001, 0.00000001]
                        }]
        
        
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

    def ScaleModel(self):
        # Create and initialise an object sc by calling a method StandardScaler()
        sc = StandardScaler()

        # Train the model by calling a method fit_transform()
        self.X_train = sc.fit_transform(self.X_train)

        # Transforms the test data
        self.X_test = sc.transform(self.X_test)

    def Predict(self):
        # Fitting Naive Bayes Classification to the Training set with linear kernel
        # Create and initialise an object sc by calling a method GaussianNB()
        self.model = GaussianNB()

        # Train the model by calling a method fit()
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

        # accuracy
        print("accuracy", metrics.accuracy_score(self.y_test, self.y_pred))
        # precision
        print("precision", metrics.precision_score(self.y_test, self.y_pred))
        # recall/sensitivity
        print("recall", metrics.recall_score(self.y_test, self.y_pred))
    
    def PlotPrecisionRecall(self):
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_pred)
        disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.show()

    def ParameterTunning(self):
        # creating a KFold object with 5 splits 
        folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

        # specify model
        model = GaussianNB()

        # set up GridSearchCV()
        self.model_cv = GridSearchCV(estimator = model, 
                                param_grid = self.hyper_params, 
                                scoring= 'accuracy', 
                                cv = folds, 
                                verbose = 1,
                                return_train_score=True)      

        # fit the model
        self.model_cv.fit(self.X_train, self.y_train)
        # cv results
        self.cv_results = pd.DataFrame(self.model_cv.cv_results_)
        self.cv_results.sort_values(by=["rank_test_score"])

    def ConfusionMatrix(self):
        # confusion matrix
        return confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)

    def CrossValidation(self):
        return cross_val_score(SVC(kernel='rbf'), self.X, self.y, scoring='accuracy', cv = 20)
    
    def plot_parameters_tuning(self):
        # converting C to numeric type for plotting on x-axis
        self.cv_results['param_C'] = self.cv_results['param_C'].astype('int')

        # # plotting
        gamma_arr = self.hyper_params[0]['gamma'];
        plt.figure(figsize=(16,6))
        plt_base_id = f"1{len(gamma_arr)}"
        count = 0
        for g_value in gamma_arr:
            count += 1
            plt_id = f"{plt_base_id}{count}"
            # subplot 1/3
            plt.subplot(int(plt_id))
            gamma_01 = self.cv_results[self.cv_results['param_gamma']==g_value]

            plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
            plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title(f"Gamma={g_value}")
            plt.ylim([0.80, 1])
            plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
            plt.xscale('log')



class RsaSupportVector:
    def __init__(self, X :pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

        self.C = 1
        self.gamma=0.1
        self.kernel='rbf'
        self.cv_results = pd.DataFrame()

        # specify range of hyperparameters
        # Set the parameters by cross-validation
        self.gammas = [0.1, 1, 10, 100]
        self.hyper_params = [
                        {
                            'gamma': [1e-2, 1e-3, 1e-4],
                            'C': self.gammas
                        }]
        
        
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

    def ScaleModel(self):
        # Create and initialise an object sc by calling a method StandardScaler()
        sc = StandardScaler()

        # Train the model by calling a method fit_transform()
        self.X_train = sc.fit_transform(self.X_train)

        # Transforms the test data
        self.X_test = sc.transform(self.X_test)

    def Predict(self):
        self.model = SVC(C = self.C, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        # accuracy
        print("accuracy", metrics.accuracy_score(self.y_test, self.y_pred))
        # precision
        print("precision", metrics.precision_score(self.y_test, self.y_pred))
        # recall/sensitivity
        print("recall", metrics.recall_score(self.y_test, self.y_pred))
    
    def PlotPrecisionRecall(self):
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_pred)
        disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.show()

    def ParameterTunning(self):
        # creating a KFold object with 5 splits 
        folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

        # specify model
        model = SVC(kernel='rbf')

        # set up GridSearchCV()
        self.model_cv = GridSearchCV(estimator = model, 
                                param_grid = self.hyper_params, 
                                scoring= 'accuracy', 
                                cv = folds, 
                                verbose = 1,
                                return_train_score=True)      

        # fit the model
        self.model_cv.fit(self.X_train, self.y_train)
        # cv results
        self.cv_results = pd.DataFrame(self.model_cv.cv_results_)
        self.cv_results.sort_values(by=["rank_test_score"])

    def ConfusionMatrix(self):
        # confusion matrix
        return confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)

    def CrossValidation(self):
        nvclassifier = GaussianNB()
        return cross_val_score(nvclassifier, self.X, self.y, scoring='accuracy', cv = 20)
    
    def plot_parameters_tuning(self):
        # converting C to numeric type for plotting on x-axis
        self.cv_results['param_C'] = self.cv_results['param_C'].astype('int')

        # # plotting
        gamma_arr = self.hyper_params[0]['gamma'];
        plt.figure(figsize=(16,6))
        plt_base_id = f"1{len(gamma_arr)}"
        count = 0
        for g_value in gamma_arr:
            count += 1
            plt_id = f"{plt_base_id}{count}"
            # subplot 1/3
            plt.subplot(int(plt_id))
            gamma_01 = self.cv_results[self.cv_results['param_gamma']==g_value]

            plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
            plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title(f"Gamma={g_value}")
            plt.ylim([0.80, 1])
            plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
            plt.xscale('log')


class RandomForest:
    def __init__(self, X :pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

        self.cv_results = pd.DataFrame()

        # specify range of hyperparameters
        # Set the parameters by cross-validation
        self.hyper_params = [{
                        'bootstrap': [True],
                        'max_depth': [80, 90],
                        'max_features': [2, 3],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [50, 100]
                    }]
        
        
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

    def ScaleModel(self):
        # Create and initialise an object sc by calling a method StandardScaler()
        sc = StandardScaler()

        # Train the model by calling a method fit_transform()
        self.X_train = sc.fit_transform(self.X_train)

        # Transforms the test data
        self.X_test = sc.transform(self.X_test)

    def Predict(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        # accuracy
        print("accuracy", metrics.accuracy_score(self.y_test, self.y_pred))
        # precision
        print("precision", metrics.precision_score(self.y_test, self.y_pred))
        # recall/sensitivity
        print("recall", metrics.recall_score(self.y_test, self.y_pred))
    
    def PlotPrecisionRecall(self):
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_pred)
        disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.show()

    def ProbabilityPrection(self):
        #Receiver Operating Characteristic Curve
        step_factor = 0.05 
        threshold_value = 0.2 
        roc_score=0
        predicted_proba = self.model.predict_proba(self.X_test) #probability of prediction
        while threshold_value <=0.8: #continue to check best threshold upto probability 0.8
            temp_thresh = threshold_value
            predicted = (predicted_proba [:,1] >= temp_thresh).astype('int') #change the class boundary for prediction
            print('Threshold',temp_thresh,'--', roc_auc_score(self.y_test, predicted))
            if roc_score<roc_auc_score(self.y_test, predicted): #store the threshold for best classification
                roc_score = roc_auc_score(self.y_test, predicted)
                thrsh_score = threshold_value
            threshold_value = threshold_value + step_factor
        print('---Optimum Threshold ---',thrsh_score,'--ROC--',roc_score)

    def ParameterTunning(self):

        # creating a KFold object with 5 splits 
        folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

        # specify model
        model = RandomForestClassifier()

        # set up GridSearchCV()
        self.model_cv = GridSearchCV(estimator = model, 
                                param_grid = self.hyper_params, 
                                scoring= 'accuracy', 
                                cv = folds, 
                                verbose = 1,
                                return_train_score=True)      

        # fit the model
        self.model_cv.fit(self.X_train, self.y_train)
        # cv results
        self.cv_results = pd.DataFrame(self.model_cv.cv_results_)
        self.cv_results.sort_values(by=["rank_test_score"])

    def ConfusionMatrix(self):
        # confusion matrix
        return confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)

    def CrossValidation(self):
        nvclassifier = GaussianNB()
        return cross_val_score(nvclassifier, self.X, self.y, scoring='accuracy', cv = 20)
    
    def plot_parameters_tuning(self):
        # converting C to numeric type for plotting on x-axis
        self.cv_results['param_C'] = self.cv_results['param_C'].astype('int')

        # # plotting
        gamma_arr = self.hyper_params[0]['gamma'];
        plt.figure(figsize=(16,6))
        plt_base_id = f"1{len(gamma_arr)}"
        count = 0
        for g_value in gamma_arr:
            count += 1
            plt_id = f"{plt_base_id}{count}"
            # subplot 1/3
            plt.subplot(int(plt_id))
            gamma_01 = self.cv_results[self.cv_results['param_gamma']==g_value]

            plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
            plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title(f"Gamma={g_value}")
            plt.ylim([0.80, 1])
            plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
            plt.xscale('log')

    def plot_sse(self):
        self.model.n_estimators = 1
        for iter in range(50):
            y_train_predicted = self.model.predict(self.X_train)
            y_test_predicted = self.model.predict(self.X_test)
            mse_train = mean_squared_error(self.y_train, y_train_predicted)
            mse_test = mean_squared_error(self.y_test, y_test_predicted)
            print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
            self.model.n_estimators += 1

__all__ = ['hopkins','RsaGaussianNB','RsaSupportVector']

