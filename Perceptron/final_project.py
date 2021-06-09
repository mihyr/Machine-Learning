import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

#Import training and testing data
train_features = pd.read_csv('UCI HAR Dataset/train/X_train.txt',delimiter="\s+",header=None)
train_labels = pd.read_csv('UCI HAR Dataset/train/y_train.txt', delimiter = "\s+",header=None)
test_features = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delimiter = "\s+",header=None)
test_labels = pd.read_csv('UCI HAR Dataset/test/y_test.txt', delimiter = "\s+",header=None)

#Shape of training features and labels 
print(f'Shape of training features {train_features.shape},and labels{train_labels.shape}')
#Shape of test features and labels 
print(f'Shape of test features {test_features.shape},and labels{test_labels.shape}')

#Find corelation matrix
#corrMatrix = train_features.corr()

#Init some constants
feature_size = train_features.shape[1]
training_size = train_features.shape[0]
testing_size = test_features.shape[0]
class_size = 6

#Convert data to nparray
train_features= train_features.to_numpy()
train_labels= train_labels.to_numpy()
test_features= test_features.to_numpy()
test_labels= test_labels.to_numpy()

#Init Helper Functions

def multi_class_weighted_sum(feature,weights):
	weighted_sum = [0 for x in range(class_size)]
	
	for index, weight in enumerate(weights):
		#print(index, weight)
		for i in range(feature_size):
			weighted_sum[index]+= weight[i] * feature[i]
	
	return weighted_sum

def multi_class_activation(weighted_sum):
    '''
    note: following numbers in training_label and test_label data refer to corresponding human actions
        1: Walking, 2: Walking_Upstairs, 3: Walking_downstairs, 4: Sitting, 5: Standing, 6:Laying
        In our weights we start from 0, hence we return 'index+1'
    '''
    max_weight = max(weighted_sum)
    index = weighted_sum.index(max_weight)

    return index + 1

def fit_perceptron(train_features,train_labels,tolerance, max_iteration):
    #Init weights of size feature_size x class_size
    weights = [[1 for x in range(feature_size)] for x in range(class_size)]

    #print(weights)

    training_size = train_features.shape[0]
    count = 0
    trigger = True
    error_array = []
    accuracy_array = []
    while trigger:
        error_counter = 0
        count +=1
        #Iterate through all training data
        for i in range(training_size):

            #extract features and labels
            bias_feature = train_features[i]
            label = train_labels[i][0]
            
            #Prediction
            w_sum = multi_class_weighted_sum(bias_feature,weights)
            output = multi_class_activation(w_sum)
            output_index = output-1
            label_index = label -1

            #Evaluation, if wrong update weights
            if output != label:
                
                #Update weights and error counter
                error_counter+=1
                
                for i in range(feature_size):
                    #subtract feature from incorrect weight and add feature to correct weight
                    weights[output_index][i] -= bias_feature[i]
                    weights[label_index][i] +=  bias_feature[i]

        
        print(f'iteration: {count}, No. of errors: {error_counter}')
        accuracy = (training_size-error_counter)*100/training_size
        error_array.append(error_counter)
        accuracy_array.append(accuracy)
        #If error is less than tolerance or loop exceed max iteration, break the loop
        if error_counter<=tolerance or max_iteration<=count:
            #print correct weights
            #print(f'Correct weights {weights}')
            trigger = False
            best_accuracy = (training_size-error_counter)*100/training_size
            return weights, accuracy_array, error_array


def predict_perceptron(test_features,test_labels,perceptron_weights):
    #Testing
    errors = 0
    testing_size = test_features.shape[0]
    predict = np.zeros(testing_size)

    #Iterate through all testing data
    for i in range(testing_size):
        bias_feature = test_features[i]
        test_label = test_labels[i][0]

        #prediction
        w_sum = multi_class_weighted_sum(bias_feature,perceptron_weights)
        output = multi_class_activation(w_sum)
        #update output to dataset
        predict[i] = output

        if output!=test_label:
            errors+=1
        
    accuracy = (testing_size-errors)*100/testing_size
    return predict, accuracy

def plot_confusion(actual_labels, predicted_labels):
    plt.title('Confusion Matrix')
    c_matrix = confusion_matrix(actual_labels, predicted_labels) 
    c_matrix_normalized = c_matrix / c_matrix.astype(np.float).sum(axis=1)
    axis_labels = ['Walking','Walking_Upstairs' ,'Walking_downstairs' ,'Sitting', 'Standing' ,'Laying']
    sn.heatmap(c_matrix, annot=True,xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel("Predicted") 
    plt.ylabel("Actual")
    #plt.rcParams['font.size'] = '5'
    plt.tight_layout()
    plt.show()
    plt.title('Normalized Confusion Matrix')
    axis_labels = ['Walking','Walking_Upstairs' ,'Walking_downstairs' ,'Sitting', 'Standing' ,'Laying']
    sn.heatmap(c_matrix_normalized, annot=True,xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel("Predicted") 
    plt.ylabel("Actual")
    #plt.rcParams['font.size'] = '5'
    plt.tight_layout()
    plt.show()

def plot_error(error_array,fit_accuracy):
    plt.plot(error_array)
    plt.xlabel("Iterations") 
    plt.ylabel("Number of Errors")
    plt.title('Errors vs. Epoch')
    plt.xticks(range(len(error_array)))
    plt.show()
    plt.plot(fit_accuracy)
    plt.xlabel("Iterations") 
    plt.ylabel("Accuracy")
    plt.title('Accuracy vs. Epoch')
    plt.xticks(range(len(fit_accuracy)))
    plt.show()


#Find weights using train data         
perceptron_weights, fit_accuracy, error_array = fit_perceptron(train_features,train_labels,tolerance=200,max_iteration=200)
#Predict using test data
perceptron_predict,test_accuracy = predict_perceptron(test_features,test_labels,perceptron_weights)

#Check accuracy of fit and test data
print(f'fit accuracy: {fit_accuracy[-1]}%, test accuracy: {test_accuracy}%')

#classification_report
print(classification_report(test_labels, perceptron_predict))

#Plot Error vs Iterations
plot_error(error_array,fit_accuracy)
#Plot Confusion matrix
plot_confusion(test_labels, perceptron_predict)
