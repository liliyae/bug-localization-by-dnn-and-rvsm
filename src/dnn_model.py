from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed, cpu_count
from math import ceil
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

def oversample(samples):
    """ Oversamples the features for label "1" 
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    samples_ = []

    # oversample features of buggy files
    for i, sample in enumerate(samples):
        samples_.append(sample)
        if i % 51 == 0:
            for _ in range(9):
                samples_.append(sample)

    return samples_


def features_and_labels(samples):
    """ Returns features and labels for the given list of samples
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    features = np.zeros((len(samples), 5))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        labels[i] = float(sample["match"])

    return features, labels


def kfold_split_indexes(k, len_samples):
    """ Returns list of tuples for split start(inclusive) and 
        finish(exclusive) indexes.
    
    Arguments:
        k {integer} -- the number of folds
        len_samples {interger} -- the length of the sample list
    """
    step = ceil(len_samples / k)

    ret_list = [(start, start + step) for start in range(0, len_samples, step)]
    return ret_list


def kfold_split(bug_reports, samples, start, finish):
    """ Returns train samples and bug reports for test
    
    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
    """
    train_samples = samples[:start] + samples[finish:]
    test_samples = samples[start:finish]

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports if br["id"] in test_br_ids]

    return train_samples, test_bug_reports

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out =self.predict(out)

        return out

def train_dnn(
    i, num_folds, samples, start, finish, sample_dict, bug_reports, br2files_dict
):
    """ Trains the dnn model and calculates top-k accuarcies
    
    Arguments:
        i {interger} -- current fold number for printing information
        num_folds {integer} -- total fold number for printing information
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        bug_reports {list of dictionaries} -- list of all bug reports
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv" pairs
    """
    print("Fold: {} / {}".format(i + 1, num_folds), end="\r")

    train_samples, test_bug_reports = kfold_split(bug_reports, samples, start, finish)
    train_samples = oversample(train_samples)
    np.random.shuffle(train_samples)
    X_train, y_train = features_and_labels(train_samples)
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()

    """clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )"""
    #clf = RandomForestRegressor(max_depth=2, random_state=0)
    #clf = GradientBoostingRegressor(random_state=0)
    #clf = LogisticRegression(random_state=0)
    """clf = AdaBoostRegressor(random_state=0, n_estimators=100)
    clf.fit(X_train, y_train.ravel())
    acc_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=clf)"""

    net = Net(5, 10, 1)
    #print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    for t in range(100):
        prediction = net(X_train)
        loss = loss_func(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=net)
    return acc_dict


def dnn_model_kfold(k=10):
    """ Run kfold cross validation in parallel
    
    Keyword Arguments:
        k {integer} -- the number of folds (default: {10})
    """
    samples = csv2dict("../data/Tomcat_features.csv")

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples)

    np.random.shuffle(samples)

    # K-fold Cross Validation in parallel
    """acc_dicts = Parallel(n_jobs=-2)(  # Uses all cores but one
        delayed(train_dnn)(
            i, k, samples, start, step, sample_dict, bug_reports, br2files_dict
        )
        for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    )"""

    """acc_dicts = []
    for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples))):
        acc_dicts.append(train_dnn( i, k, samples, start, step, sample_dict, bug_reports, br2files_dict))"""


    acc_dicts = Parallel(cpu_count() - 1)(  # Uses all cores but one
        delayed(train_dnn)(
            i, k, samples, start, step, sample_dict, bug_reports, br2files_dict
        )
        for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    )

    # Calculating the average accuracy from all folds
    avg_acc_dict = {}
    for key in acc_dicts[0].keys():
        avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)

    return avg_acc_dict
