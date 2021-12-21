from agent_factory.agent_factory import PbRLAgentFactory
from agents.preference_based.pbrl_agent import PbRLAgent


class PbRLAgentAssembler:

    @staticmethod
    def assemble_agent(env, reward_model_name, agent_factory: PbRLAgentFactory, num_pretraining_epochs,
                       num_training_iteration_epochs, pb_step_freq) -> PbRLAgent:

        reward_model = agent_factory.create_reward_model(env, reward_model_name)
        policy_model = agent_factory.create_policy_model(env, reward_model)
        pretraining_query_generator = agent_factory.create_pretraining_query_generator()
        query_generator = agent_factory.create_query_generator()
        preference_collector = agent_factory.create_preference_collector()
        preference_querent = agent_factory.create_preference_querent()
        reward_model_trainer = agent_factory.create_reward_model_trainer(reward_model)
        query_schedule_cls = agent_factory.create_query_schedule_cls()

        return PbRLAgent(policy_model, pretraining_query_generator, query_generator, preference_querent,
                         preference_collector, reward_model_trainer, reward_model, query_schedule_cls,
                         pb_step_freq, num_pretraining_epochs, num_training_iteration_epochs)
