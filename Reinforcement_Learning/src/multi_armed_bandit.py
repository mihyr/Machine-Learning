from os import access
import numpy as np


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed. Use an epsilon-greedy policy for action selection.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        available_actions = env.action_space.n
        available_space = env.observation_space.n
        state_action_values = np.zeros((available_space,available_actions))
        rewards = np.zeros((100))
        n = np.zeros(available_actions)
        q = np.zeros(available_actions)
        
        reward_list = np.array([])
        counter = 0
        trigger = 0

        while trigger<steps:
          trigger +=1
          initialize = env.reset()
          reward_iter = 0

          for j in range(100):
            rand = np.random.random()

            if rand < self.epsilon:
              action = env.action_space.sample()
              
            elif rand >= self.epsilon:
              action = np.argmax(state_action_values[initialize,:])
            
            obs, reward, done, inf = env.step(action)
            reward_iter+=reward
            if done:
              env.reset()
              break
            
          np.append(reward_list,reward_iter)

          if np.size(reward_list)== np.floor(steps/100):
            if trigger% np.floor(steps / 100)==0:
              rewards[counter] = np.sum(reward_list)/len(reward_list)
              reward_list = np.array([])
              counter += 1

          n[action] += 1
          rqa = reward - q[action]
          q[action] += 1/n[action]*rqa

        state_action_values = np.tile(q, (available_space, 1))
        return state_action_values, rewards
  

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        states = []
        actions = []
        rewards = []
        sav =state_action_values
        initialize = env.reset()
        trigger = True

        while trigger:
          get_max = np.argmax(sav[initialize,:])
          obs, reward, done, inf = env.step(get_max)

          initialize = obs
          states.append(obs)
          actions.append(get_max)
          rewards.append(reward)

          #print(obs, reward, done, inf)
          if done != True:
            pass
          else:
            trigger = False
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            env.reset()
            return states, actions, rewards
