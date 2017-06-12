import numpy as np
import scipy
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy.optimize as opt
import loadcifar10 as lcf
from sklearn.decomposition import PCA



def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis = 1)).T
    return sm


def compute_cost(w, x, y, lam):
    #m = Number of features
    m = x.shape[0]
    #print ("M's size", m)

    y_oneH = one_hot_convert(y)
    #print ("One Hot Converted Y = ", y_oneH)

    prob = softmax(np.dot(x, w))
    #print ("Probability", prob)

    loss = (-1 / m) * np.sum(y_oneH * np.log(prob)) + (lam / 2) * np.sum(w * w)
    return loss

def compute_gradient (w, x, y, lam):
    m = x.shape[0]
    y_oneH = one_hot_convert(y)
    #scores = np.dot(x, w)
    prob = softmax(np.dot(x, w))
    grad = (-1 / m) * np.dot(x.T,(y_oneH - prob)) + lam * w

    return grad

def one_hot_convert(y):
    #m = y.shape[0]
    #oneH = scipy.sparse.csr_matrix((np.ones(m), (y, np.array(range(m)))))
    #oneH = np.array(oneH.todense()).T
    oneH = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)

    return oneH


def compute_probability (w, x):
    prob = softmax (np.dot (x, w))
    return prob

def predict(w, x):
    prediction = np.argmax(compute_probability(w, x), axis = 1)
    return prediction

def compute_accuracy(w, x, y):
    prob = compute_probability(w, x)
    pred = predict(w, x)

    accuracy = sum(pred == y)/(float(len(y)))
    return accuracy

def steepest_descent (w, x, y, alpha):
    threshold = 0.0001
    theta_values = w
    iteration = 0
    cost_difference = 100

    current_cost = compute_cost (w, x, y, 1)
    cost_iterations = np.array([])
    cost_iterations = np.append(cost_iterations, current_cost)

    '''
    step_increase = alpha * 1.01
    step_decrease = alpha * 0.5
    '''

    #print ("W's values")
    #print (w)
    #print ("Theta_Values")
    #print (theta_values)
    #print ("Shape of Theta Values " + str(theta_values.shape))

    iteration_performance = np.array([])

    while (current_cost != 0 and cost_difference > threshold and alpha != 0):
        #print (theta_values)
        previous_cost = current_cost
        previous_theta = theta_values
        previous_alpha = alpha

        current_cost = compute_cost(theta_values, x, y, 1)
        theta_values = theta_values - (alpha * compute_gradient(theta_values, x, y, 1))
        next_cost = compute_cost (theta_values, x, y, 1)

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
            theta_values = theta_values - (alpha * compute_gradient (theta_values, x, y, 1))
            #iteration_performance = np.append(iteration_performance)
            #theta_values = theta_values - (alpha * gradient_function(theta_values, x, y))

        elif current_cost < next_cost:
            iteration += 1
            alpha = previous_alpha
            alpha *= 0.5

            theta_values = previous_theta
            #theta_values = theta_values - (alpha * calculate_gradient (theta_values, x, y))
            current_cost = previous_cost
            #iteration_performance = np.append(iteration_performance)
            #theta_values = theta_values - (alpha * calculate_gradient (theta_values, x, y))
            #theta_values = theta_values - (alpha * gradient_function(theta_values, x, y))


    #return theta_values, cost_iterations, iteration, iteration_performance
    return theta_values, cost_iterations

#Load the Digit Dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target



print ("Shape of X" , x.shape)
print ("Shape of Y" , y.shape)

'''
Initializes a weight matrix to the shape of m x n.
For example for iris. W is (4 x 3). 4 Features, 3 classes.
w first takes the shape of of x's shape on axis 1 (number of columns)
it then takes the length of the number of unique values in y. In this case,
there are 3 classes. SO 3 unique values.
'''
w = np.ones([x.shape[1],len(np.unique(y))])
#w = np.random.normal (loc = 0.0, scale = 0.66, size = (x.shape[1], len(np.unique(y))))

print ("Shape of W", w.shape)

alpha = 0.1
digit_test_x = x
digit_theta = w

digit_test_x = np.insert (digit_test_x, 0, values = 1, axis = 1)
digit_theta = np.insert (w, 0, values = np.random.normal(loc = 0.0, scale = 0.77), axis = 0)

#digits_theta, digit_losses = steepest_descent(w, x, y, alpha)
digits_theta, digit_losses = steepest_descent (digit_theta, digit_test_x, y, alpha)


#print ("Digits Dataset Theta Values", digits_theta)
probability = compute_probability(digits_theta, digit_test_x)
predictions = predict (digits_theta, digit_test_x)

#print ("Probability Matrix", probability)
print ("Our Digits Data Set Predictions", predictions)

#print ('Digits Data Set Training Accuracy: ', compute_accuracy(digits_theta,x,y))
print ('Digits Data Set Training Accuracy: ', compute_accuracy(digits_theta, digit_test_x, y))


#Loading the Iris Dataset
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
iris_w = np.ones([iris_x.shape[1],len(np.unique(iris_y))])

iris_theta, iris_losses = steepest_descent(iris_w, iris_x, iris_y, alpha)

iris_prob = compute_probability (iris_theta, iris_x)
iris_pred = predict (iris_theta, iris_x)

print ("Our Iris Data Set Predictions", iris_pred)
print ('Iris Data Set Training Accuracy: ', compute_accuracy(iris_theta, iris_x, iris_y))


#Digit Dataset Steepest Descent Graph
plt.title('Digit Dataset - Steepest Descent')
plt.plot(range(len(digit_losses)), digit_losses)
plt.xlabel('Iterations')
plt.ylabel('Cost Values')
plt.savefig('Digit Dataset Steepest.png')
plt.show()

#Iris Dataset Steepest Descent Graph
plt.title('Iris Dataset - Steepest Descent')
plt.plot(range(len(iris_losses)), iris_losses)
plt.xlabel('Iterations')
plt.ylabel('Cost Values')
plt.savefig('Iris Dataset Steepest.png')
plt.show()

'''
quasi_newton_weights = np.zeros (shape = len(iris_x[0]))
optimized_qn_weights = opt.fmin_bfgs(compute_cost, quasi_newton_weights, args = (iris_x, iris_y, 1));
optimized_qn_weights = np.insert(optimized_qn_weights, 0, values = np.random.normal(loc = 0.0, scale = 0.77), axis=0)
qn_test_x = x
qn_test_x = np.insert(qn_test_x, 0, values = np.random.normal(loc = 0.0, scale = 0.77), axis=1)
quasi_newton_prediction = predict (optimized_qn_weights, qn_test_x)
print ("Our Iris Data Set Quasi-Newton Predictions", quasi_newton_prediction)
print ('Iris Data Set Quasi-Newton Training Accuracy: ', compute_accuracy(optimized_qn_weights, qn_test_x, iris_y))
'''

#test_x = x
#digits_test_theta = digits_theta

'''
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = lcf.get_CIFAR10_data()
lcf.visualize_sample(X_train_raw, y_train_raw, classes)
subset_classes = ['plane', 'car']
X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = lcf.subset_classes_data(subset_classes)
X_train, y_train, X_val, y_val, X_test, y_test = lcf.preprocessing_CIFAR10_data(X_train_raw, y_train_raw, X_val_raw,
                                                                            y_val_raw, X_test_raw, y_test_raw)
'''


#print (lcf.y_train)

#lcf.X_train = lcf.X_train.T

#lcf.X_train = lcf.X_train[:1768]
print('Train data shape: ', lcf.X_train_raw.shape)
print ('Train labels shape: ', lcf.y_train_raw.shape)

#lcf.X_train = lcf.X_train.T
#print('Train data transposed: ', lcf.X_train.shape)



#cifar_weight = np.ones([lcf.X_train_raw.shape[1],len(np.unique(lcf.y_train_raw))])
#print ("Cifar weight vector shape", cifar_weight.shape)





pca = PCA(n_components = 100)
pca.fit(lcf.X_train_raw)
new_X = pca.transform(lcf.X_train_raw)
lcf.X_train_raw = new_X
print (new_X)


cifar_weight = np.ones([lcf.X_train_raw.shape[1],len(np.unique(lcf.y_train_raw))])
cifar_weight, cifar_losses = steepest_descent(cifar_weight, lcf.X_train_raw, lcf.y_train_raw, alpha)
cifar_prob = compute_probability (cifar_weight, lcf.X_train_raw)
cifar_pred = predict (cifar_weight, lcf.X_train_raw)

print ("Our Cifar10 Data Set Predictions", cifar_pred)
print ('Cifar10 Data Set Training Accuracy: ', compute_accuracy(cifar_weight, lcf.X_train_raw, lcf.y_train_raw))

'''
cifar_weight = np.ones([new_X.shape[1],len(np.unique(lcf.y_train))])
print ("Cifar weight vector shape", cifar_weight.shape)
cifar_weight, cifar_losses = steepest_descent(cifar_weight, new_X, lcf.y_train, alpha)
cifar_prob = compute_probability (cifar_weight, new_X)
cifar_pred = predict (cifar_weight, new_X)
print ("Our Cifar10 Data Set Predictions", cifar_pred)
print ('Cifar10 Data Set Training Accuracy: ', compute_accuracy(cifar_weight, new_X, lcf.y_train))
'''