import logging
import random

PREFERENCE_QUERYSET_SIZE = 2


class RandomQueryGenerator:
    def __init__(self, segment_samples):
        self.segment_samples = segment_samples
        self.queries = []

    def try_save_query(self):
        try:
            self.save_query()
        except ValueError as e:
            log_msg = "Query generation failed. There aren't enough segment samples available ({}) to create " \
                      "a preference query of size {}. Original error message: " + str(e)
            logging.warning(
                log_msg.format(len(self.segment_samples), PREFERENCE_QUERYSET_SIZE))

    def save_query(self):
        query = self.generate_query()
        self.queries.append(query)

    def generate_query(self):
        return random.sample(self.segment_samples, PREFERENCE_QUERYSET_SIZE)

    def clear(self):
        self.queries.clear()
