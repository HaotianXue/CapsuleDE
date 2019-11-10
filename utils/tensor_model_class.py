"""
This file is the interface class for model
You just simply pass your model and your data_loader and we will take care anything else
Including:
train (and also train_val split), test, recording the loss and validation results, plotting, saving checkpoints, etc

Note: it's an abstract class and you can extend it to a pytorch model or tensorflow/keras model, etc

Author: Haotian Xue
"""

from abc import abstractmethod


class TensorModel:

    """
    This class is an interface for all kinds of deep learning models.
    The design purpose for this is that we have different dl framework,
    so people may build a same model in different ways (some choose pytorch and others choose keras).
    They are in common in some properties and different in other aspects.
    So this interface class provides some basic common properties that you need after you
    handcraft a deep learning model and wants to train and test this model
    """

    def __init__(self, train_data_set, test_data_set, hyper_parameter, train_requirement):
        self.hyper_parameter = hyper_parameter
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set
        self.train_requirement = train_requirement

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def plot(self):
        pass


