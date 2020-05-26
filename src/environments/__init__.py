

def create_train_env(opt, video=False):
    if opt.game == "super_mario":
        from src.environments.super_mario_env import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)

    if opt.game == "invaders":
        from src.environments.invaders import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)