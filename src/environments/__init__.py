

def create_train_env(opt, video=False):
    if opt.game == "super_mario":
        from src.environments.games.super_mario import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)

    if opt.game == "invaders":
        from src.environments.games.invaders import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)

    if opt.game == "breakout":
        from src.environments.games.breakout import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)

    if opt.game == "pong":
        from src.environments.games.pong import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)

    if opt.game == "bipedal":
        from src.environments.games.bipedal import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)
    
    if opt.game == "lunarlander":
        from src.environments.games.lunarlander import create_train_env as _create_train_env
        if video:
            return _create_train_env(opt.world, opt.stage, opt.action_type, "{}/video_{}_{}_{}_{}.mp4".format(opt.output, opt.game, opt.world, opt.stage, opt.action_type))
        else:
            return _create_train_env(opt.world, opt.stage, opt.action_type)