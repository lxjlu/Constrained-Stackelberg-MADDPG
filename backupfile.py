# one time target calculation
class Runner_Stochastic:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir
        self.reward_record = [[] for i in range(args.n_agents)]
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        self.n_action = 5
        self.enable_cost = args.enable_cost
        self.cost_threshold = 3
        self._init_agents()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        self.leader_agent = Leader_Stochastic(self.args, 0)
        self.follower_agent = Follower_Stochastic(self.args, 1)

    def run(self):
        returns = []
        total_reward = [0, 0]
        done = [False*self.args.n_agents]
        info = None
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step==0 or np.all(done):
                # save episode rewards
                for i in range(self.args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                # save first arrive record
                if info is not None:
                    self.crash_record.append(info["crash"])
                    self.leader_arrive_record.append(info["leader_arrived"])
                    self.follower_arrive_record.append(info["follower_arrived"])
                # reset episode total reward
                total_reward = [0, 0]
                # reset
                s, info = self.env.reset()
                # reshape observation
                s = np.array(s).reshape((2, 8))
            with torch.no_grad():
                # choose actions
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon, self.cost_threshold)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon, self.cost_threshold)
            u = [leader_action, follower_action]
            actions = (leader_action, follower_action)
            # step simulation
            s_next, r, done, truncated_n, info = self.env.step(actions)
            # reshape observation
            s_next = np.array(s_next).reshape((2, 8))
            # cost
            c = info["cost"]
            # add target actions(estimate nest actions)
            u_next = self.target_action(s_next[:self.args.n_agents], done[:self.args.n_agents])
            # store transitions
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], u_next, done[:self.args.n_agents], c=c)
            # observe update
            s = s_next
            # accumulate episode reward
            for i in range(self.args.n_agents):
                total_reward[i]+=r[i]
            # train
            if self.buffer.current_size >= self.args.sample_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.leader_agent.train(transitions)
                self.follower_agent.train(transitions)
            # plot reward
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                # plt.figure()
                # plt.plot(range(len(returns)), returns)
                # plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                # plt.ylabel('average returns')
                # plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            # np.save(self.save_path + '/returns.pkl', returns)
        # analyze data
        for i in range(self.args.n_agents):
            self.analysis(i)
        # save data
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
        np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
        np.save(self.save_path + '/crash_record.npy', self.crash_record)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            step = 0
            s, info = self.env.reset()
            s = np.array(s).reshape((2, 8))
            rewards = [0, 0]
            for time_step in range(self.args.evaluate_episode_len):
                step+=1
                with torch.no_grad():
                    leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon, self.cost_threshold)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon, self.cost_threshold)
                actions = (leader_action, follower_action)
                # print(actions)
                s_next, r, done, truncated_n, info = self.env.step(actions)
                s_next = np.array(s_next).reshape((2, 8))
                rewards[0] += r[0]
                rewards[1] += r[1]
                self.env.render()
                s = s_next
                # print(self.env.controlled_vehicles[0].speed)
                # print(self.env.controlled_vehicles[1].speed)
                
                # print(".......")
                # time.sleep(1)
                if np.all(done):
                    # print(info)
                    break
            returns.append(rewards)
            print('Returns is', rewards)
            # print(step)
        return np.sum(returns, axis=0) / self.args.evaluate_episodes

    def analysis(self, agent_id):
        plt.figure()
        plt.plot(self.reward_record[agent_id])
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.savefig(self.save_path + '/reward_agent{}.png'.format(agent_id), format='png')
    
    def target_action(self, o_next, t):
        # next_obs_leader = torch.tensor(o_next[0])
        # next_obs_follower = torch.tensor(o_next[1])
        next_leader_act = self.leader_agent.select_action(o_next[0], self.noise, self.epsilon, self.cost_threshold)
        next_follower_act = self.follower_agent.select_action(o_next[1], next_leader_act, self.noise, self.epsilon, self.cost_threshold)
        # max_leader_q = float("-inf")
        # max_follower_q = float("-inf")
        # next_leader_act = 0
        # next_follower_act = 0
        # # leader decision
        # for leader_act in torch.arange(self.n_action):
        #     for follower_act in torch.arange(self.n_action):
        #         u = [F.one_hot(leader_act, num_classes=self.n_action), F.one_hot(follower_act, num_classes=self.n_action)]
        #         temp = self.leader_agent.critic_network(next_obs_leader, u, dim=0)
        #         if temp>=max_leader_q:
        #             max_leader_q = temp
        #             next_leader_act = leader_act
        # # follower decision
        # for follower_act in torch.arange(self.n_action):
        #     u = [F.one_hot(next_leader_act, num_classes=self.n_action), F.one_hot(follower_act, num_classes=self.n_action)]
        #     temp = self.follower_agent.critic_network(next_obs_follower, u, dim=0)
        #     if temp>=max_follower_q:
        #         max_follower_q = temp
        #         next_follower_act = follower_act
        return [next_leader_act, next_follower_act]
