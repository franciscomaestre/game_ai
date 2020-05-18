

def make_train_env(env_name, env_conf):
    if env_conf["type"] == "super_mario":
        from games.super_mario import make_train_env as _make_train_env
    if env_conf["type"] == "atari":
        from games.atari import make_train_env as _make_train_env            
    
    return _make_train_env(env_name = env_name, env_conf = env_conf)