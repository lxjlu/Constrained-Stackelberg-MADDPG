import numpy as np
import inspect
import functools
import gym
import json
import highway_env
highway_env.register_highway_envs()


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # env = MultiAgentEnv(world)
    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.high_action = 1
    args.low_action = -1
    return env, args

def make_env_intersection(args):
    env = gym.make('intersection-v1')

    env.configure({
    "controlled_vehicles": 2,
    "initial_vehicle_count" : 0,
    "spawn_probability": 0,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
        "type": "Kinematics",
        "flatten": True,
        "absolute": True,
        "see_behind": True,
        "normalize": False,
        "features": ['x', 'y', 'vx', 'vy'],
        "vehicles_count": 2
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
        "type": "ContinuousAction",  #throttle and steering angle 
        "longitudinal": True,
        "lateral": False
        }
    },
    "duration": 20,
    "arrived_reward": 10,
    "collision_reward": -10,
    "high_speed_reward": 0,
    "on_road_reward": 0,
    "offroad_terminal": True,
    "first_arrive_reward": 20,
    "second_arrive_reward": 10
    })

    env.reset()

    args.n_players = 2  # 包含敌人的所有玩家个数
    args.n_agents = 2  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [8, 9]  # 每一维代表该agent的obs维度
    # action_shape = []
    # for content in env.action_space:
    #     action_shape.append(content.n)
    # args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.action_shape = [1,1]
    args.terminal_shape = [1,1]
    args.high_action = 1
    args.low_action = -1
    return env, args

def make_highway_env(args):
    env = gym.make(args.scenario_name)

    with open(args.file_path+'/env_config.json','r') as f:
        env.configure(json.load(f))
    
    env.reset()

    args.n_players = 2  # agent number
    args.n_agents = 2  # agent number
    args.obs_shape = [8, 8]  # obs dim
    args.action_shape = [1,1] # act dim
    # args.action_shape = [2,2]
    args.action_dim = [5,5] # act num for discrete action
    args.terminal_shape = [1,1] # terminal dim
    args.high_action = 1  # act high for continuous action
    args.low_action = -1  # act low for continuous action

    return env, args
    
