import numpy as np
import classification.logistic_regression_utils as lrutils
from classification.display_utils import plot_diagrams
from classification.config_utils import load_config
from classification.dataset_utils import Dataset




if __name__ == "__main__":
    try:
        config = load_config()

        dataset_config = config["dataset"]

        dataset = Dataset(
            dataset_config["path"],
            dataset_config["remove_punctuations"],
            dataset_config["remove_digits"],
            dataset_config["lower_case"],
            dataset_config["remove_stop_words"],
            dataset_config["stop_words"],
            dataset_config["test_count"],
            dataset_config["bag_of_words_max_features"])
        
        print("Dataset loaded...")

        classifier = lrutils.get_classifier(
            dataset.get_train_data(),
            dataset.get_train_labels())

        print("Classifier fit...")

        lrutils.test_classifier(
            classifier,
            dataset.get_train_data(),
            dataset.get_train_labels(),
            dataset.get_test_data(),
            dataset.get_test_labels())

        diagrams = []

        gammas = np.arange(0.0, 0.5, 0.01)

        # Get margin counts
        vect_margin_counts = np.vectorize(
            lambda g: lrutils.margin_counts(classifier, dataset.get_test_data(), g))
        
        diagrams.append(
            (gammas,
            vect_margin_counts(gammas) / len(dataset.get_test_data()),
            "Margin",
            "Fraction of Points Above Margin"))
        
        print("Margin counts calculated...")

        # Get margin errors
        vect_margin_errors = np.vectorize(
            lambda g: lrutils.margin_errors(classifier, dataset.get_test_data(), dataset.get_test_labels(), g))
        
        margin_errors = vect_margin_errors(gammas)

        diagrams.append(
            (gammas,
            margin_errors,
            "Margin",
            "Error Rate"))

        print("Margin errors calculated...")

        # Get safe margins
        errors = np.arange(np.max(margin_errors), np.min(margin_errors), -0.01)
        vect_find_safe_margin = np.vectorize(
            lambda e: lrutils.find_safe_margin(e, vect_margin_errors, gammas))
        
        safe_margins = vect_find_safe_margin(errors)

        diagrams.append(
            (errors,
            safe_margins,
            "Max Tolerable Error",
            "Safe Margin"))
        
        print("Safe margins calculated...")
        
        # Display all gathered data
        plot_diagrams(diagrams)

        # Display the most positive and negative words
        dataset.introduce_influencer_words(classifier.coef_[0, :], 10)

    except KeyboardInterrupt:
        print("User interrupted, exiting...")
        pass