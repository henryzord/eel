class Individual(object):
    def __init__(self):
        self._n_objectives = None
        self._model = None

    @property
    def n_objectives(self):
        return self._n_objectives

    def __eq__(self, other):
        pass

    def __ne__(self, o):
        pass

    def __ge__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def dominates(self, b):
        return Individual.a_dominates_b(self, b)

    @classmethod
    def a_dominates_b(cls, a, b):
        pass

    @classmethod
    def domination_matrix(cls, P, Q):
        pass