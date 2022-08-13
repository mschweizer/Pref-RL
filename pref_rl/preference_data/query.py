import uuid


class Query:
    def __init__(self):
        self.id = str(uuid.uuid4())  # TODO: Add env and experiment prefix


class ChoiceSetQuery(Query):
    def __init__(self, choice_set):
        assert len(choice_set) > 1, "A choice set must have at least two elements."
        super().__init__()
        self.choice_set = choice_set

    def __len__(self):
        return len(self.choice_set)

    def __getitem__(self, key):
        return self.choice_set[key]


class BinaryChoiceSetQuery(ChoiceSetQuery):
    def __init__(self, choice_set):
        assert len(choice_set) == 2, "The choice set of a binary choice must have exactly two alternatives."
        super(BinaryChoiceSetQuery, self).__init__(choice_set)
