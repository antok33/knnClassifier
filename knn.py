import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt

def distance(p1, p2):
    ''' 
    Returns the distance between 2 points.
    '''
    return np.sqrt(np.sum((np.power(p1 - p2, 2))))


def majority_vote(votes):
    '''
    Returns the most common element of the sequence.
    '''
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
            
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    
    return random.choice(winners)

def find_nearest_neighboors(p, points, k=5):
    '''
    Returns the k nearest neighboors (k default value is 5).
    '''
    distances = np.zeros(points.shape[0])
    # Loop over all points
    for i in range(len(distances)):
        # compute distance between target point and every other
        distances[i] = distance(p, points[i])
    # sort distances and return k nearest points to target point
    indexes = np.argsort(distances)
    return indexes[:k]


def knn_predict(p, points, classes, k=5):
    '''
    Predict the class for the point p.
    '''
    # find k nearest neighboors
    indexes = find_nearest_neighboors(p, points, k)
    # predict the class of p based on majority vote
    return majority_vote(classes[indexes])
    

def generate_data(n=400):
    '''
    Generate data from 2 different distributions (D_1 has mean=2 and std=2, D_2 has mean=4 and std=2).
    '''
    mean1 = 2; std1 = 2;
    mean2 = 4; std2 = 2;
    points = np.concatenate((ss.norm(mean1,std1).rvs((n,2)), ss.norm(mean2,std2).rvs((n,2))))
    classes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return points, classes

def accuracy(predicted_classes, test_classes):
    acc = np.mean(predicted_classes == test_classes)*100
    return acc    


def run_experinment():
    points, classes = generate_data(n=400) # n defines the number of instances for each class 
    threshold = int(points.shape[0] * 0.85) # threshold seperates the data to train and test set
    # Divide dataset to train and test set 
    train_points = points[:threshold]
    train_classes = classes[:threshold]
    test_points = points[threshold:]
    test_classes = classes[threshold:] 
    predicted_classes = [knn_predict(p, train_points, train_classes, k=3) for p in test_points]
    predicted_classes = np.array(predicted_classes)
    print "KNN accuracy on the generated instance: ", accuracy(predicted_classes, test_classes), "%"

run_experinment()
