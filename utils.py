# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
import shap

import warnings
warnings.filterwarnings("ignore")

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def consitency_mod(metric, col_position,n_neighbors=5):
    """Calculate consistency defined in: 
    https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf. 
    
    Adaptation of the consistency metrics implemented in: 
    https://aif360.mybluemix.net/

    Return consistency result    
    
    Parameters
    ----------
    metric: aif360.metrics.BinaryLabelDatasetMetric
    col_position: int
        Column position of the sensitive group in the dataset        
    n_neighbors: int
        Number of neighbors to use in KNN
    """
    X = metric.dataset.features
    X = np.delete(X, col_position, 1)
    X = StandardScaler().fit_transform(X) 
    num_samples = X.shape[0]
    y = metric.dataset.labels

    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)

    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency/num_samples

    return consistency
    
def compute_metrics(model, X_test, y_test, X_train, y_train, dataset_test, 
                    dataset_name, model_name, unprivileged_groups, 
                    privileged_groups, position):
    """
    Calculate and return: model accuracy and fairness metrics
    
    Parameters
    ----------
    model: scikit-learn classifier    
    X_test: numpy 2d array
    y_test: numpy 1d array
    X_train: numpy 2d array
    y_train: numpy 1d array
    dataset_test: aif360.datasets.BinaryLabelDataset
    dataset_name: string
        Dataset name used in the analysis
    model_name: string
    unprivileged_groups: list<dict>
        Dictionary where the key is the name of the sensitive column in the 
        dataset, and the value is the value of the unprivileged group in the
        dataset
    privileged_groups: list<dict>
        Dictionary where the key is the name of the sensitive column in the 
        dataset, and the value is the value of the privileged group in the
        dataset
    position: int
        Column position of the sensitive group in the dataset 
    """
    
    y_pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    print("Test accuracy: ", acc_test)
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    print("Train accuracy: ", acc_train)
    
    dataset_pred = dataset_test.copy()
    dataset_pred.labels = y_pred_test

    bin_metric = BinaryLabelDatasetMetric(dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    disparate_impact_bin = bin_metric.disparate_impact()
    print('Disparate impact: ', disparate_impact_bin)
    mean_difference = bin_metric.mean_difference()
    print('Mean difference: ', mean_difference)

    classif_metric = ClassificationMetric(dataset_test, dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    classif_disparete_impact = classif_metric.disparate_impact()
    avg_odds = classif_metric.average_odds_difference()
    print('Average odds difference:', avg_odds)
    equal_opport = classif_metric.equal_opportunity_difference()
    print('Equality of opportunity:', equal_opport)
    false_discovery_rate = classif_metric.false_discovery_rate_difference()
    print('False discovery rate difference:', false_discovery_rate)
    entropy_index = classif_metric.generalized_entropy_index()
    print('Generalized entropy index:', entropy_index)

    cons_comp = consitency_mod(bin_metric, position,n_neighbors=5)
    print('Consistency: ', cons_comp)
    
    result = (dataset_name, model_name, acc_test, disparate_impact_bin, 
              mean_difference, classif_disparete_impact, avg_odds, 
              equal_opport, false_discovery_rate, entropy_index, cons_comp)

    return result
    
def tree_shap_results(model, model_name, X_train, X_test, dataset_test, 
                      name_protect, position):
    """
    Generate SHAP grapahs, computes SHAP values and evaluate SHAP results for 
    a tree base model
    
    Parameters
    ----------
    model: Tree base model. Models supported: XGBoost, LightGBM, CatBoost, 
        and scikit-learn
    model_name: string
    X_train: numpy 2d array
    X_test: numpy 2d array
    dataset_test: aif360.datasets.BinaryLabelDataset
    name_protect: string
    position: int
        Column position of the sensitive group in the dataset  
    """
                          
    explainer = shap.TreeExplainer(model=model, data=X_train)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values,list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_test, 
                      feature_names=dataset_test.feature_names)
    
    shap.summary_plot(shap_values, X_test, 
                      feature_names=dataset_test.feature_names, 
                      plot_type="bar")
    
    df_feat = pd.DataFrame({'feature_names': dataset_test.feature_names, 
                            'value': np.abs(shap_values).mean(axis=0)})
    df_feat = df_feat.sort_values(by='value', ascending=False).reset_index(drop=True)
    feat_import = df_feat.loc[df_feat['feature_names'] == name_protect, 'value'].iloc[0]
    feature_pos = df_feat[df_feat['feature_names'] == name_protect].index[0]
    
    shap.dependence_plot(position, shap_values, dataset_test.features, 
                         dataset_test.feature_names)
    
    df = pd.DataFrame({'shap': shap_values[:, position], 
                       'feat_value': dataset_test.features[:, position]})
    unpriv_value = df.loc[df['feat_value'] == 0, 'shap'].mean()
    priv_value = df.loc[df['feat_value'] == 1, 'shap'].mean()
    print('Mean SHAP value unprivileged class: ', unpriv_value)
    print('Mean SHAP value privileged class: ', priv_value)
    
    results = stats.ttest_ind(df.loc[df['feat_value'] == 1, 'shap'], 
                              df.loc[df['feat_value'] == 0, 'shap'], 
                              equal_var=False)
    alpha = 0.05
    if (results[0] > 0) & (results[1]/2 < alpha):
        print("reject null hypothesis, mean of group privilegiad is greater than mean of unprivilegiad")
        priv_greater_unpriv = 1
    else:
        print("accept null hypothesis, mean of group privilegiad is NOT greater than mean of unprivilegiad")
        priv_greater_unpriv = 0   
        
    shap_results = (model_name, feat_import, feature_pos, unpriv_value, 
                    priv_value, priv_greater_unpriv)
    return df_feat, shap_results
    
def linear_shap_results(model, model_name, X_train, X_test, dataset_test, 
                        name_protect, position):
    """
    Generate SHAP grapahs, computes SHAP values and evaluate SHAP results for 
    a linear model
    
    Parameters
    ----------
    model: Linear model of scikit-learn
    model_name: string
    X_train: numpy 2d array
    X_test: numpy 2d array
    dataset_test: aif360.datasets.BinaryLabelDataset
    name_protect: string
    position: int
        Column position of the sensitive group in the dataset  
    """
    
    explainer = shap.LinearExplainer(model, X_train, feature_dependence="correlation")
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=dataset_test.feature_names)
    
    shap.summary_plot(shap_values, X_test, feature_names=dataset_test.feature_names, 
                      plot_type="bar")
    
    df_feat = pd.DataFrame({'feature_names': dataset_test.feature_names, 
                            'value': np.abs(shap_values).mean(axis=0)})
    df_feat = df_feat.sort_values(by='value', ascending=False).reset_index(drop=True)
    feat_import = df_feat.loc[df_feat['feature_names'] == name_protect, 'value'].iloc[0]
    feature_pos = df_feat[df_feat['feature_names'] == name_protect].index[0]

    shap.dependence_plot(position, shap_values, dataset_test.features, 
                         dataset_test.feature_names)

    df = pd.DataFrame({'shap': shap_values[:, position], 
                       'feat_value': dataset_test.features[:, position]})
    unpriv_value = df.loc[df['feat_value'] == 0, 'shap'].mean()
    priv_value = df.loc[df['feat_value'] == 1, 'shap'].mean()
    print('Mean SHAP value unprivileged class: ', unpriv_value)
    print('Mean SHAP value privileged class: ', priv_value)
    
    results = stats.ttest_ind(df.loc[df['feat_value'] == 1, 'shap'], 
                          df.loc[df['feat_value'] == 0, 'shap'], equal_var=False)
    alpha = 0.05
    if (results[0] > 0) & (results[1]/2 < alpha):
        print("reject null hypothesis, mean of group privilegiad is greater than mean of unprivilegiad")
        priv_greater_unpriv = 1
    else:
        print("accept null hypothesis, mean of group privilegiad is NOT greater than mean of unprivilegiad")
        priv_greater_unpriv = 0
        
    shap_results = (model_name, feat_import, feature_pos, unpriv_value, 
                    priv_value, priv_greater_unpriv)
    return df_feat, shap_results