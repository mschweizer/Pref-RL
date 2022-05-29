from ..preference_collector.binary_choice import BinaryChoice
from ..query_generator.query import BinaryChoiceQuery


# TODO: Find theoretically consistent name for classes Preference, Choice, and Query.
class Preference:
    def __init__(self, query, choice):
        self.query = query
        self.choice = choice

    def __repr__(self):
        return str(self.choice)


class BinaryChoiceSetPreference(Preference):
    def __init__(self, query, choice):
        assert isinstance(query, BinaryChoiceQuery)
        assert isinstance(choice, BinaryChoice)
        super(BinaryChoiceSetPreference, self).__init__(query, choice)
