from runner_bilevel import Runner_Bilevel, Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_highway_env
import json


if __name__ == '__main__':
    # get the params
    args = get_args()

    # set train params
    # args.file_path = "./roundabout_env_result/exp2"
    # args.file_path = "./intersection_env_result/exp3_base"
    # args.file_path = "./intersection_env_result/exp15"
    args.file_path = "./racetrack_env_result/exp1_base"
    # args.file_path = "./merge_env_result/exp5"
    with open(args.file_path+'/config.json','r') as f:
        vars(args).update(json.load(f))
    
    # set env
    env, args = make_highway_env(args)

    # choose action type and algorithm
    if args.action_type == "continuous":
        if args.version == "bilevel":
            runner = Runner_Bilevel(args, env)
        else:
            runner = Runner_C_Bilevel(args, env)
    elif args.action_type == "discrete":
        runner = Runner_Stochastic(args, env)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()

