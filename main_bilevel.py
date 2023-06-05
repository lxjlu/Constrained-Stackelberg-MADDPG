from runner_bilevel import Runner_Bilevel, Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_highway_env
import numpy as np
import json


if __name__ == '__main__':
    # get the params
    args = get_args()

    # set train params
    # args.file_path = "./roundabout_env_result/exp3_base"
    # args.file_path = "./roundabout_env_result/exp5/seed_3"
    # args.file_path = "./intersection_env_result/exp2_base"
    # args.file_path = "./intersection_env_result/exp17/seed_4"
    # args.file_path = "./racetrack_env_result/exp10"
    # args.file_path = "./merge_env_result/exp3_base"
    # args.file_path = "./merge_env_result/exp6/seed_3"
    args.file_path = "./racetrack_env_result/exp11"
    args.save_dir = args.file_path
    with open(args.file_path+'/config.json','r') as f:
        vars(args).update(json.load(f))
    
    # set env
    env, args = make_highway_env(args)

    np.random.seed(args.seed)

    # choose action type and algorithm
    if args.action_type == "continuous":
        if args.version == "bilevel":
            runner = Runner_Bilevel(args, env)
        elif args.version == "c_bilevel":
            runner = Runner_C_Bilevel(args, env)
    elif args.action_type == "discrete":
        runner = Runner_Stochastic(args, env)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()

