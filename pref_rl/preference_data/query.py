import uuid


class Query:
    def __init__(self):
        """
        A (preference) query is sent to a user to elicit their preferences.
        """
        self.id = str(uuid.uuid4())  # TODO: Add env and experiment prefix


class ChoiceSetQuery(Query):
    def __init__(self, choice_set: tuple):
        """
        A choice set query is a special type of preference query that asks the user to make a choice among a (finite)
        set of alternatives.
        :param choice_set: The set of alternatives the user chooses from.
        """
        assert len(choice_set) > 1, "A choice set must have at least two elements."
        super().__init__()
        self.choice_set = choice_set

    def __len__(self):
        return len(self.choice_set)

    def __getitem__(self, key: int):
        return self.choice_set[key]


class BinaryChoiceSetQuery(ChoiceSetQuery):
    def __init__(self, choice_set: tuple):
        """
        A binary choice set query has exactly two alternatives in the choice set.
        :param choice_set: The set of alternatives the user chooses from.
        """
        assert len(choice_set) == 2, "The choice set of a binary choice must have exactly two alternatives."
        super(BinaryChoiceSetQuery, self).__init__(choice_set)
