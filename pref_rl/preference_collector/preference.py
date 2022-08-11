from ..preference_collector.binary_choice import BinaryChoice
from ..query_generation.query import BinaryChoiceQuery


# TODO: Find consistent name for classes Preference, Choice, and Query.
class Preference:
    def __init__(self, query, choice):
        self.query = query
        self.choice = choice

    def __repr__(self):
        return "Choice: " + str(self.choice)

    def __eq__(self, other):
        return self.choice == other.choice and self.query.id == other.query.id


class BinaryChoiceSetPreference(Preference):
    def __init__(self, query, choice):
        assert isinstance(query, BinaryChoiceQuery)
        assert isinstance(choice, BinaryChoice)
        super(BinaryChoiceSetPreference, self).__init__(query, choice)
