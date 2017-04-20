from diktya.func_api_helpers import load_model
import numpy as np


class Matcher():
    """Wrapper class for tag_matcher model """
    def __init__(self, model_path=None):
        """Initialization for tag_matcher.

        Arguments:
            model_path (:obj:`string`): path to tag_matcher model.
        """
        # if model_path is None:
        #    model_path = os.join.path(os.path.dirname(__file__), '../model/tag_matcher.model')
        self.model = load_model(model_path)
        self.model._make_predict_function()

    @staticmethod
    def _int_to_bit_array(n, length):
        bits = [1 if digit == '1' else 0 for digit in bin(n)[2:]]
        if len(bits) < length:
            bits = [0]*(length-len(bits)) + bits
        return bits

    @staticmethod
    def _representation_to_bit_representation(representation):
        representation = np.array([
            Matcher._int_to_bit_array(n, 8)
            for n in representation]).flatten()
        return representation

    def match_representations(self, representations_a, representations_b):
        """Matches two lists of tag representations

        Arguments:
            representations_a(:obj:`list` of :obj:`list` of int)
            representations_b(:obj:`list` of :obj:`list` of int)

        Returns:
            :obj:`list` of float: similarity scores
        """
        assert len(representations_a) == len(representations_b), 'inputs do not have ' \
            'the same length'
        representations_a = np.array([Matcher._representation_to_bit_representation(representation)
                                      for representation in representations_a])
        representations_b = np.array([Matcher._representation_to_bit_representation(representation)
                                      for representation in representations_b])
        return self.model.predict([representations_a, representations_b]).flatten()
