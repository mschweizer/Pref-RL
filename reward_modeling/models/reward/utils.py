_model_registry = {}


def get_model_by_name(name):
    """
    Based on stablebaslines3
    """
    if name not in _model_registry:
        raise KeyError(
            f"Error: unknown model type {name},"
            f"the only registered model type are: {list(_model_registry.keys())}!"
        )
    return _model_registry[name]


def register_model(name, model):
    """
    Based on stablebaselines3
    """
    if name in _model_registry:
        # Check if the registered model is the same we try to register.
        # If not so, do not override and raise error.
        if _model_registry[name] != model:
            raise ValueError(f"Error: the name {name} is already registered for a different model, will not override.")
    _model_registry[name] = model
