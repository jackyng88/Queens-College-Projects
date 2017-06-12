import numpy as np
from sklearn import datasets
import scipy.optimize as opt
import math as math
import random
from random import randrange
from numpy import linalg as la
import copy
import matplotlib.pyplot as plt
import time



def sigmoid (x):
    return 1 / (1 + np.exp(-x))

def cost_function(w, x, y):
    '''
    w = np.matrix(w)
    x = np.matrix (x)
    y = np.matrix (y)
    '''

    first = sigmoid(np.dot(x, w))
    final = (-y) * np.log(first) - (1 - y) * np.log(1 - first)

    return np.mean(final)

def calculate_gradient (w,x,y):
    first = sigmoid(np.dot(x, w))
    error = first - y
    gradient = np.dot(error, first) / y.size

    #print (gradient.shape)
    return gradient



def steepest_descent (w, x, y, alpha):
    threshold = 0.0001
    #rows = x.shape[0]
    #params = x.shape[1]

    #theta_values = np.zeros(1, params)
    #theta_values = copy.copy(w)
    theta_values = w
    #theta_values = np.array(theta_values)
    #theta_values = np.matrix(theta_values)
    #theta_values = w.shape[0]
    current_cost = cost_function (w, x, y)

    cost_iterations = np.array([])
    cost_iterations = np.append(cost_iterations, current_cost)

    step_increase = alpha * 1.01

    print ("W's values")
    print (w)

    print ("Theta_Values")
    print (theta_values)
    print ("Shape of Theta Values " + str(theta_values.shape))


    #w = w - alpha * gradient
    #final_calc = first_calc.T.dot(x)

    #gradient_one = sigmoid(x @ w.T) - y
    #gradient = gradient_one.T.dot(x)

    #gradient = np.matrix(gradient)

    #print ("Shape of Gradient " + str(gradient.shape))
    iteration = 0

    cost_difference = 100
    iteration_performance = np.array([])

    while (current_cost != 0 and cost_difference > threshold and alpha != 0):
        #iteration += 1
        st = time.time()
        #theta_values = theta_values - (alpha * gradient_function(theta_values, x, y))
        print (theta_values)


        #current_cost = cost_function (theta_values, x, y)

        previous_cost = current_cost
        previous_theta = theta_values
        previous_alpha = alpha

        #cost_iterations = np.append(cost_iterations, current_cost)

        current_cost = cost_function(theta_values, x, y)
        theta_values = theta_values - (alpha * calculate_gradient(theta_values, x, y))
        next_cost = cost_function (theta_values, x, y)

        cost_difference = abs(current_cost - next_cost)

        print ("Iteration # " + str(iteration))
        print ("Current Cost = " + str(current_cost))
        print ("Next Cost = " + str(next_cost))
        print ("Difference in Cost = " + str(cost_difference))
        print ("Current Value of Alpha = " + str(alpha))

        if (current_cost >= next_cost):
            iteration += 1
            alpha *= 1.01
            #cost_difference = abs(current_cost - next_cost)
            #cost_iterations = np.append(cost_iterations, current_cost)
            #cost_iterations = np.append(cost_iterations, next_cost)

            current_cost = next_cost

            cost_iterations = np.append(cost_iterations, current_cost)

            theta_values = theta_values - (alpha * calculate_gradient (theta_values, x, y))
            et = time.time()
            total_time = et - st
            iteration_performance = np.append(iteration_performance, total_time)

            #theta_values = theta_values - (alpha * gradient_function(theta_values, x, y))


        elif current_cost < next_cost:
            alpha = previous_alpha
            alpha *= 0.5

            theta_values = previous_theta
            #theta_values = theta_values - (alpha * calculate_gradient (theta_values, x, y))
            current_cost = previous_cost
            et = time.time()
            total_time = et - st
            iteration_performance = np.append(iteration_performance, total_time)


            #theta_values = theta_values - (alpha * calculate_gradient (theta_values, x, y))
            #theta_values = theta_values - (alpha * gradient_function(theta_values, x, y))


    #cost_iterations = np.array(cost_iterations)
    return theta_values, cost_iterations, iteration, iteration_performance


def predict (w, x):
    m,n = x.shape
    probability = np.zeros(shape=(m, 1));
    h = sigmoid(x.dot(w.T))
    #probability = 1 * (h >= 0.5)
    probability = np.where (h >= 0.5, 1, 0)

    return probability


def k_folds_cv (w,x,y, n):
    blockSize = len(x) / n
    #100 / 5 = 20.

#LOADING INPUT DATA
iris = datasets.load_iris()
x = iris.data[:100, :4]
y = iris.target[:100]

initial_w = np.random.normal (loc = 0.0, scale = 0.66, size = (len(x[0])))
#initial_w = np.array(initial_w)
print ("Initial values of W = " + str(initial_w))
print ("Y's shape = " + str(y.shape))

#cost_values = []

trained_w, cost_values, num_iterations, iteration_performance = steepest_descent (initial_w, x, y, 5)
#print ("Shape of Trained Weights = " + str(trained_w.shape))
print ("Trained W values = ")
#trained_w = trained_w[0]
print(trained_w)

#trained_w = np.insert (trained_w, 0, values = np.random.normal (loc = 0.0, scale = 0.77),axis = 0)
trained_w = np.insert (trained_w, 0, values = 0, axis = 0)


test_x = x
test_x = np.insert(test_x, 0, values=np.random.normal(loc = 0.0, scale = 0.77), axis=1)
#np.random.shuffle(test_x)
#test_x = np.insert(test_x, 0, values = 1, axis=1)


print("Shape of Test X = " + str(test_x.shape))
print("Shape of Trained W = " + str(trained_w.shape))

start_time = time.time()
prediction = predict (trained_w, test_x)
end_time = time.time()

steep_total_time = end_time - start_time
print (steep_total_time)
'''
correct = [1 if a == b else 0 for (a, b) in zip(prediction, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
'''

print ('Accuracy: %f' % ((y[np.where(prediction == y)].size / float(y.size)) * 100.0));


print (cost_values)
print (num_iterations)

quasi_newton_weights = np.zeros (shape = len(x[0]))
optimized_qn_weights = opt.fmin_bfgs(cost_function, quasi_newton_weights, args=(x, y));
optimized_qn_weights = np.insert(optimized_qn_weights, 0, values=np.random.normal(loc = 0.0, scale = 0.77), axis=0)

qn_test_x = x
qn_test_x = np.insert(qn_test_x, 0, values=np.random.normal(loc = 0.0, scale = 0.77), axis=1)

start_time = time.time()
quasi_newton_prediction= predict (optimized_qn_weights, qn_test_x)
end_time = time.time()
quasi_total_time = end_time - start_time


print ('Accuracy: %f' % ((y[np.where(quasi_newton_prediction == y)].size / float(y.size)) * 100.0));


print ("Optimized Steepest Descent Weight Vector" , trained_w)
print ("Optimized Quasi-Newton Weight Vector", optimized_qn_weights)

print (len(x))


#kfold_weights = cross_validate_kfold (trained_w, test_x, y, 10)
#kfold_predictions = predict (kfold_weights, x)
#print ('Accuracy: %f' % ((y[np.where(kfold_predictions == y)].size / float(y.size)) * 100.0));

#kfold_data = k_folds_split(test_x, 5)
#print (kfold_data)


#scores = evaluate_algorithm(trained_w, test_x, y, 1.01)

plt.plot(range(len(cost_values)), cost_values, 'ro')
plt.axis([0, len(cost_values), 0, 5])
plt.savefig('Cost Iterations.pdf')

print ("Steepest Prediction took ", steep_total_time, " seconds")
print ("Quasi Prediction took ", quasi_total_time, " seconds")
print ("Iteration Time Vector" , iteration_performance)
print ("Average Time of Iterations ", np.mean(iteration_performance))

#print (sigmoid(5))


