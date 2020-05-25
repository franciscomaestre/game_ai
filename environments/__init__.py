

def make_train_env(env_params):
    if env_params["type"] == "super_mario":
        from environments.super_mario import make_train_env as _make_train_env
    if env_params["type"] == "atari":
        from environments.atari import make_train_env as _make_train_env            
    
    return _make_train_env(env_params = env_params)