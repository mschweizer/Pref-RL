_schedule_registry = {}


def get_schedule_by_name(name):
    """
    Based on stablebaslines3
    """
    if name not in _schedule_registry:
        raise KeyError(
            f"Error: unknown schedule type {name},"
            f"the only registered schedule types are: {list(_schedule_registry.keys())}!"
        )
    return _schedule_registry[name]


def register_schedule(name, schedule):
    """
    Based on stablebaselines3
    """
    if name in _schedule_registry:
        if _schedule_registry[name] != schedule:
            raise ValueError(f"Error: the name {name} is already registered for a different schedule, "
                             f"will not override.")
    _schedule_registry[name] = schedule
