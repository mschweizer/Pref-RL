from pref_rl.agents.policy.model import PolicyModel


def test_saves_model_to_file(cartpole_env, tmpdir):
    policy_model = PolicyModel(cartpole_env, None, train_freq=10)
    model_name = "pbrl_agent_policy"
    policy_model.save(directory=str(tmpdir) + "/", model_name="pbrl_agent_policy")
    assert tmpdir.join("/" + model_name + ".zip").exists()
