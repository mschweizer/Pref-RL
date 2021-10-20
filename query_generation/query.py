import uuid


class Query:
    def __init__(self):
        self.id = str(uuid.uuid4())  # TODO: Add env and experiment prefix


class ChoiceSetQuery(Query):
    def __init__(self, choice_set):
        super().__init__()
        self.choice_set = choice_set
