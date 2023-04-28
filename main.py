from runner import Runner
from common.arguments import get_args
from common.utils import make_env, make_env_intersection
import numpy as np
import random
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    # env, args = make_env(args)
    env, args = make_env_intersection(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
