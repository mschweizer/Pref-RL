from gym.wrappers import FrameStack


def wrap_env(env, frame_stack_depth=4):
    env = FrameStack(env, frame_stack_depth)
    return env
