import numpy as np
from DataLoader import DataLoader
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

configs_knn = {
        'n_neighbors': [3, 5],
        'metric': ['euclidean'],
    }

configs_rforest={
        'criterion': ['gini', 'entropy'],
        'max_depth': [3]
    }

configs_svm={
        'kernel': ['rbf'],
        'C': [1, 10]
    }
configs_dtree={
        'criterion': ['gini', 'entropy'],
        'max_depth': [3]
    }

result_knn = {
        "hyperparameter_train_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }
result_svm= {
        "hyperparameter_train_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }
result_dtree={
        "hyperparameter_train_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }
result_forest = {
        "hyperparameter_train_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }  


def normalize_plus_minus1(data):
    data=(2*(data-np.min(data)))/(np.max(data)-np.min(data)) -1
    return data
   

knn=KNeighborsClassifier()
fold_num=1
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
for train_index, test_index in outer_cv.split(dataset, labels):

    # Get the partitions for the current fold
    X_train, X_test = normalize_plus_minus1(dataset[train_index]), normalize_plus_minus1(dataset[test_index])
    y_train, y_test = labels[train_index], labels[test_index]

    grid_search=GridSearchCV(KNeighborsClassifier(), configs_knn, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    param_num=0
    for mean_score,params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if str(params) not in result_knn['hyperparameter_train_scores']:
            result_knn["hyperparameter_train_scores"][str(params)]=[mean_score]
        else:
            result_knn["hyperparameter_train_scores"][str(params)].append(mean_score)
        param_num+=1
    param_num=0
    best_knn=grid_search.best_estimator_
    y_pred=best_knn.predict(X_test)
    result_knn["mean_test_score"].append(accuracy_score(y_test,y_pred))
    result_knn["f1_score"].append(f1_score(y_test,y_pred))

    grid_search=GridSearchCV(SVC(), configs_svm, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    param_num=0
    for mean_score,params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if str(params) not in result_svm['hyperparameter_train_scores']:
            result_svm["hyperparameter_train_scores"][str(params)]=[mean_score]
        else:
            result_svm["hyperparameter_train_scores"][str(params)].append(mean_score)
        param_num+=1
    param_num=0
    best_svm=grid_search.best_estimator_
    y_pred=best_svm.predict(X_test)
    result_svm["mean_test_score"].append(accuracy_score(y_test,y_pred))
    result_svm["f1_score"].append(f1_score(y_test,y_pred))  

    grid_search=GridSearchCV(DecisionTreeClassifier(), configs_dtree, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    param_num=0
    for mean_score,params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if str(params) not in result_dtree['hyperparameter_train_scores']:
            result_dtree["hyperparameter_train_scores"][str(params)]=[mean_score]
        else:
            result_dtree["hyperparameter_train_scores"][str(params)].append(mean_score)
        param_num+=1
    param_num=0
    best_dtree=grid_search.best_estimator_
    y_pred=best_dtree.predict(X_test)
    result_dtree["mean_test_score"].append(accuracy_score(y_test,y_pred))
    result_dtree["f1_score"].append(f1_score(y_test,y_pred))

    config_means = []
    for i in range(5):
        grid_search = GridSearchCV(RandomForestClassifier(), configs_rforest, cv=inner_cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        config_means.append(grid_search.cv_results_['mean_test_score'])

    config_means = np.mean(config_means, axis=0)
    param_num=0
    for mean_score,params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if str(params) not in result_forest['hyperparameter_train_scores']:
            result_forest["hyperparameter_train_scores"][str(params)]=[mean_score]
        else:
            result_forest["hyperparameter_train_scores"][str(params)].append(mean_score)
        param_num+=1
    param_num=0
    best_rf = grid_search.cv_results_['params'][np.argmax(np.array(config_means))]
    print(best_rf)
    rf = RandomForestClassifier(criterion=str(best_rf['criterion']),max_depth=int(best_rf['max_depth']))
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_test)
    result_forest["mean_test_score"].append(accuracy_score(y_test,y_pred))
    result_forest["f1_score"].append(f1_score(y_test,y_pred))
    print("Fold Num "+str(fold_num)+" Done")
    fold_num+=1

for config in result_knn['hyperparameter_train_scores']:
    mean_value = np.mean(result_knn['hyperparameter_train_scores'][config])
    std_value = np.std(result_knn['hyperparameter_train_scores'][config])
    interval_size = 1.96 * std_value / np.sqrt(len(result_knn['hyperparameter_train_scores'][config]))
    print(f"{config} - {mean_value} +/- {interval_size}")

print('mean_test_score')
mean_value = np.mean(result_knn['mean_test_score'])
std_value = np.std(result_knn['mean_test_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_knn['mean_test_score']))
print(f"{mean_value} +/- {interval_size}")

print('f1_score')
mean_value = np.mean(result_knn['f1_score'])
std_value = np.std(result_knn['f1_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_knn['f1_score']))
print(f"{mean_value} +/- {interval_size}")

for config in result_svm['hyperparameter_train_scores']:
    mean_value = np.mean(result_svm['hyperparameter_train_scores'][config])
    std_value = np.std(result_svm['hyperparameter_train_scores'][config])
    interval_size = 1.96 * std_value / np.sqrt(len(result_svm['hyperparameter_train_scores'][config]))
    print(f"{config} - {mean_value} +/- {interval_size}")

print('mean_test_score')
mean_value = np.mean(result_svm['mean_test_score'])
std_value = np.std(result_svm['mean_test_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_svm['mean_test_score']))
print(f"{mean_value} +/- {interval_size}")

print('f1_score')
mean_value = np.mean(result_svm['f1_score'])
std_value = np.std(result_svm['f1_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_svm['f1_score']))
print(f"{mean_value} +/- {interval_size}")

for config in result_dtree['hyperparameter_train_scores']:
    mean_value = np.mean(result_dtree['hyperparameter_train_scores'][config])
    std_value = np.std(result_dtree['hyperparameter_train_scores'][config])
    interval_size = 1.96 * std_value / np.sqrt(len(result_dtree['hyperparameter_train_scores'][config]))
    print(f"{config} - {mean_value} +/- {interval_size}")

print('mean_test_score')
mean_value = np.mean(result_dtree['mean_test_score'])
std_value = np.std(result_dtree['mean_test_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_dtree['mean_test_score']))
print(f"{mean_value} +/- {interval_size}")

print('f1_score')
mean_value = np.mean(result_dtree['f1_score'])
std_value = np.std(result_dtree['f1_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_dtree['f1_score']))
print(f"{mean_value} +/- {interval_size}")

for config in result_forest['hyperparameter_train_scores']:
    mean_value = np.mean(result_forest['hyperparameter_train_scores'][config])
    std_value = np.std(result_forest['hyperparameter_train_scores'][config])
    interval_size = 1.96 * std_value / np.sqrt(len(result_forest['hyperparameter_train_scores'][config]))
    print(f"{config} - {mean_value} +/- {interval_size}")

print('mean_test_score')
mean_value = np.mean(result_forest['mean_test_score'])
std_value = np.std(result_forest['mean_test_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_forest['mean_test_score']))
print(f"{mean_value} +/- {interval_size}")

print('f1_score')
mean_value = np.mean(result_forest['f1_score'])
std_value = np.std(result_forest['f1_score'])
interval_size = 1.96 * std_value / np.sqrt(len(result_forest['f1_score']))
print(f"{mean_value} +/- {interval_size}")


    
    
