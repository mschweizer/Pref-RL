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

    def __str__(self):
        if self.action is not None and self.reward is not None and self.done is not None and self.info is not None:
            string_rep = "Obs: {o}; Act: {a}; Rew: {r}; Done: {d}; ORew: {orew}".format(o=self.observation,
                                                                                        a=self.action,
                                                                                        r=self.reward,
                                                                                        d=self.done,
                                                                                        orew=self.info[
                                                                                            "original_reward"])
        else:
            string_rep = "Obs: {}".format(self.observation)

        return string_rep
