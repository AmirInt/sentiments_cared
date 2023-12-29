import numpy as np
from sklearn.linear_model import SGDClassifier 
from typing import Callable




def get_classifier(
        datax: np.ndarray,
        datay: np.ndarray) -> SGDClassifier:

    classifier = SGDClassifier(loss="log", penalty=None)
    classifier.fit(datax, datay)
    return classifier


def test_classifier(
        classifier: SGDClassifier,
        trainx: np.ndarray,
        trainy: np.ndarray,
        testx: np.ndarray,
        testy: np.ndarray) -> None:

    pred_train = classifier.predict(trainx)
    pred_test = classifier.predict(testx)

    error_train = float(np.sum((pred_train > 0) != (trainy > 0))) / len(trainy)
    error_test = float(np.sum((pred_test > 0) != (testy > 0))) / len(testy)

    print("Train Error:", error_train)
    print("Test Error:", error_test)
    

def margin_counts(
        classifier: SGDClassifier,
        data: np.ndarray,
        gamma: float) -> float:

    # Compute probability of class 1 on each test point
    preds = classifier.predict_proba(data)[:, 1]
    
    # Find data points for which prediction is at least gamma away from 0.5
    margin_indicess = np.where((preds > (0.5 + gamma)) | (preds < (0.5 - gamma)))[0]

    return float(len(margin_indices))


def margin_errors(
        classifier: SGDClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        gamma: float) -> float:

    # Compute probability on each test point
    preds = classifier.predict_proba(data)[:, 1]

    # Find data points for which prediction is at least gamma away from 0.5
    margin_indices = np.where((preds > (0.5 + gamma)) | (preds < (0.5 - gamma)))[0]

    # Compute error on those data points
    num_errors = np.sum((preds[margin_indices] > 0.5) != (labels[margin_indices] > 0.0))
    return float(num_errors) / len(margin_indices)


def find_safe_margin(
        e: float,
        f: Callable,
        gammas: float) -> float:

    # f is the vectorised margin_errors. Get the margin errors
    errors = f(gammas)

    # Get all errors underneath the given error threshold
    confident_errors = errors < e

    # Find the margin index beyond whose margin all errors are less than the threshold
    safe_margin_idx = min([i for i in range(len(gammas)) if sum(confident_errors[i:]) == len(confident_errors[i:])])

    return gammas[safe_margin_idx]
