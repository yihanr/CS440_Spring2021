import gym
import numpy as np
import torch
from torch import nn

import utils
from policies import QPolicy


def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features= statesize, out_features=64), 
        torch.nn.ReLU(), 
        torch.nn.Linear(in_features = 64, out_features = actionsize))
    return model
    pass


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        if model is not None:
            self.model = model
        else:
            self.model = np.zeros(self.buckets + (actionsize,))
        self.statesize = statesize
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        self.model.train()
        state = torch.from_numpy(state).type(torch.FloatTensor)
        #next_state = torch.from_numpy(next_state).type(torch.FloatTensor)

        #print("--------------------")
        #print(self.model(state))
        #print("--------------------")
        self.optimizer.zero_grad()

        original_qval = self.model(state)[action]

        if done == True:
            temp = np.array(reward)
            temp = torch.from_numpy(temp).type(torch.FloatTensor)
            target = temp
        else:
            qvals = self.qvals(next_state)
            qvals = torch.from_numpy(qvals).type(torch.FloatTensor)
            target = reward + self.gamma * torch.max(qvals)

        # Q(s,a)←Q(s,a)+α⋅(target−Q(s,a))
        self.model(state)[action] = original_qval + self.lr * (target - original_qval)

        loss = self.loss_fn(target, original_qval)
        #loss = (original_qval - target) * (original_qval - target)

        loss.backward()
        self.optimizer.step()

        return float(loss)


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'dqn.model')
