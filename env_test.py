import gym
import time
import numpy as np
import highway_env
highway_env.register_highway_envs()
           
# env = gym.make("intersection-v0") # or any other environment  (e.g. roundabout-v0)
# env = gym.make("racetrack-v0")
env = gym.make("merge-v0")
env.configure({ 
  "manual_control": True,
  "real_time_rendering": True,
  "screen_width": 1000,
  "screen_height": 1000,
  "duration": 20,
  "observation": {
      "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "flatten": True,
            "absolute": True,
            "see_behind": True,
            "normalize": False,
            "features": ["x", "y", "vx", "vy"],
            "vehicles_count": 2
            }
  },
  "action": {
    "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction"
        }
  }
})
# env.configure({ 
#   "duration": 30,
#   "screen_width": 2000,
#   "screen_height": 2000,
#   "observation": {
#       "type": "MultiAgentObservation",
#         "observation_config": {
#             "type": "Kinematics",
#             "flatten": True,
#             "absolute": True,
#             "see_behind": True,
#             "normalize": False,
#             "features": ["x", "y", "vx", "vy"],
#             "vehicles_count": 2
#             }
#   },
#   "action": {
#     "type": "MultiAgentAction",
#         "action_config": {
#             "type": "DiscreteMetaAction",  
#         }
#   }
# })
# env.reset()
# # print(env.action_space)
# # print(env.observation_space)
# done = False
# while not done:
#     act = env.action_space.sample() # with manual control, these actions are ignored
#     obs, reward, done, _, _ = env.step(act) # with manual control, these actions are ignored
#     done = np.all(done)
#     env.render()
#     time.sleep(1)
# env.close()

# env = gym.make("roundabout-v0")
# env.configure({
#     "manual_control": True,
#     "real_time_rendering": True,
#     "screen_width": 2000,
#     "screen_height": 2000,
#     # "action": {
#     #     "type": "DiscreteMetaAction"
#     # }
#     # "observation": {
#     #   "type": "MultiAgentObservation",
#     #     "observation_config": {
#     #       "type": "Kinematics",
#     #       "flatten": True,
#     #       "absolute": True,
#     #       "see_behind": True,
#     #       "normalize": False,
#     #       "features": ['x', 'y', 'vx', 'vy'],
#     #       "vehicles_count": 2
#     #     }
#     # },
#     "action": {
#       "type": "MultiAgentAction",
#         "action_config": {
#           "type": "DiscreteMetaAction",
#           "target_speeds": [0, 4, 8, 12, 16]
#         }
#     },
# })

env.reset()
done = False
while not done:
    act = env.action_space.sample()

    obs, reward, done, _, _ = env.step(act) 

    # print(env.controlled_vehicles[0].target_speeds)
    # print(env.controlled_vehicles[1].target_speeds)
    # print(".......")
    done = np.all(done)
    env.render()
    
    time.sleep(1)