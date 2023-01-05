import pickle
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def confidence_interval(data):
    print(data)
    m=np.mean(data)
    s=np.std(data)
    lower_bound=m-1.96*(s/np.sqrt(5))
    upper_bound=m+1.96*(s/np.sqrt(5))
    print("mean (%.3f), confidence Interval (%.3f,%.3f)"%(m,lower_bound,upper_bound))


# Load the dataset and labels
dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

# Define the parameter grid for grid search
param_grid = {'C': [1, 10],
              'kernel': ['rbf', 'poly']}

# Create the model
model = SVC()

# Create the outer cross-validation object
outer_cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=12)

# Create the grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,scoring="accuracy")

cross_score=np.zeros((4,10))
mean_scores=[]
iter_num=0
iter_mean=0
for train_index, test_index in outer_cv.split(dataset, labels):
    X_train, X_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    processed_data=np.concatenate((X_train,X_test))
    shuffled_labels=np.concatenate((y_train,y_test))
    # Fit the grid search object to the data
    scaler = StandardScaler()
    grid_search.fit(scaler.fit_transform(processed_data), shuffled_labels)

    # Evaluate the model's performance
    param_num=0
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        cross_score[param_num][iter_mean]=mean_score
        param_num+=1
    iter_num+=1
    iter_mean+=1

    if (iter_mean==10):
        mean_scores.append(np.mean(cross_score,axis=1))
        iter_mean=0
        cross_score=np.zeros((4,10))
        

# Calculate the average performance of the model over all outer folds
param_num=0
mean_scores=np.array(mean_scores)
print(mean_scores)
for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    data=mean_scores[:,param_num]
    print("Parameter configuration:", params)
    confidence_interval(data)
    param_num+=1
