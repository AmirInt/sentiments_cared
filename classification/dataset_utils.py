import os
import string
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer




class Dataset:
    def __init__(
            self,
            data_path: os.PathLike,
            remove_punctuations: bool,
            remove_digits: bool,
            lower_case: bool,
            remove_stop_words: bool,
            stop_words: list,
            test_count: int,
            max_features: int) -> None:

        load_dataset(data_path)
        
        if remove_punctuations:
            remove_puncs()
        
        if remove_digits:
            remove_numbers()

        if lower_case:
            to_lower_case()
        
        if remove_stop_words:
            remove_stops(stop_words)

        to_bag_of_words(max_features)

        split_train_test(test_count)


    def load_dataset(self, data_path: os.PathLike) -> None:
        with open(data_path) as file:
            content = file.readlines()
        
        # Remove the trailing white spaces
        content = [x.strip() for x in content]

        # Separate the sentences and labels
        self.__sentences = [x.split('\t')[0] for x in content]
        labels = [x.split('\t')[1] for x in content]
        
        # Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
        self.__y = np.array(labels, dtype='int')
        self.__y = 2 * self.__y - 1


    def remove_puncs(self) -> None:
        self.__sentences = [s.replace(punc, ' ') for punc in list(string.punctuations) for s in self.__sentences]

    
    def remove_numbers(self) -> None:
        self.__sentences = [s.replace(digit, ' ') for digit in str(range(10)) for s in self.__sentences]
    

    def to_lower_case(self) -> None:
        self.__sentences = [s.lower() for s in self.__sentences]
    
    
    def remove_stops(self, stop_words: list) -> None:
        self.__sentences = [s.split() for s in self.__sentences]
        self.__sentences = [" ".join(list(filter(lambda a: a not in stop_words, s))) for s in self.__sentences]
    
    
    def to_bag_of_words(self, max_features) -> None:
        self.__vectoriser = CountVectorizer(
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None,
            max_features=max_features)
        
        data_features = vectoriser.fit_transform(self.__sentences)

        self.__sentences = data_features.toarray()
    
    
    def split_train_test(self, test_count) -> None:
        np.random.seed(0)

        test_indices = np.append(
            np.random.choice(np.where(self.__y == -1)[0], test_count / 2, replace=False),
            np.random.choice(np.where(self.__y == 1)[0], test_count / 2, replace=False))
        
        train_indices = list(set(range(len(self.__y))) - set(test_indices))

        self.__train_data = self.__sentences[train_indices]
        self.__train_labels = self.__y[train_indices]

        self.__test_data = self.__sentences[test_indices]
        self.__test_labels = self.__y[test_indices]
    

    def introduce_influencer_words(self, w: np.ndarray, num: int) -> None:
        vocab = np.array(
            [z[0] for z in sorted(self.__vectoriser.vocabulary_.items(), key=lambda x: x[1])])
        
        indices = np.argsort(w)

        neg_indices = indices[0 : num]
        pos_indices = indices[-num + 1 : -1]

        print("Highly Negative Words:")
        print([str(x) for x in list(vocab[neg_inds])])
        print("\nHighly positive words:")
        print([str(x) for x in list(vocab[pos_inds])])


    def get_train_data(self) -> np.ndarray:
        return self.__train_data
    
    
    def get_train_labels(self) -> np.ndarray:
        return self.__train_labels
    
    
    def get_test_data(self) -> np.ndarray:
        return self.__test_data
    
    
    def get_test_labels(self) -> np.ndarray:
        return self.__test_labels