class ExperienceBuffer:
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.experiences = []

    def append(self, experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.size:
            del self.experiences[0]


# TODO: what is the "correct" way to do this?
class LastNExperiences:
    def __init__(self, n, experience_buffer):
        self.n = n
        self.experience_buffer = experience_buffer

    def view(self):
        return self.experience_buffer.experiences[-self.n:]


class Experience:
    def __init__(self, observation, action=None, reward=None, done=None, info=None):
        self.action = action
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    def __eq__(self, other):
        return self.action is other.action and \
               self.observation is other.observation and \
               self.reward is other.reward and \
               self.done is other.done and \
               self.info is other.info
