import gym
import torch
import numpy as np

from DDPG.DPG_Actor_NN import Actor_NN
from DDPG.DPG_Critic_NN import Critic_NN
from DDPG.Van_Memory_Buffer import V_Memory_B
from ModifiedMountainCar_env import ModifiedInvMCar

n_episodes = 50
batch_size = 64
buffer_size = 60000
start_update = 1
max_t_steps = 999
discount= 0.85
ln_rate_c = 5e-4#0.005
ln_rate_a = 1e-3 #0.001
ep_print = 1
decay_upd = 0.45#0.999#0.9999
action_std = 0.1
theta = 0.1
mu = 0
max_action=1
epsilion =1
eps_decay = 0.9


torch.manual_seed(0)
env = gym.make("MountainCarContinuous-v0")
env = ModifiedInvMCar(env,easier=False)


state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]


rpl_buffer = V_Memory_B(batch_size = batch_size,size = buffer_size)

#Initialise actor
agent = Actor_NN(Input_size=state_size,Hidden_size = 64,ln_rate = ln_rate_a,max_action=max_action).double()
agent.apply(agent.weights)
target_agent = Actor_NN(Input_size=state_size,Hidden_size = 64,ln_rate = 0,max_action=max_action).double()

target_agent.load_state_dict(agent.state_dict())
target_agent.freeze_params()

# Initialise two critic NN one to be the fixed target and the other to be trained for stability
critic_nn = Critic_NN(Input_size = state_size + action_size,Hidden_size = 64,ln_rate = ln_rate_c).double()
critic_nn.apply(critic_nn.weights)
critic_target = Critic_NN(Input_size = state_size + action_size,Hidden_size = 64,ln_rate =0 ).double()


# Make sure two critic NN have the same initial parameters
critic_target.load_state_dict(critic_nn.state_dict())

#Freeze the critic target NN parameter
critic_target.freeze_params()


#Initialise the replay buffer
total_acc = []
ep_rwd = []
critic_losses = []
training_acc = []
det_actions = []

# Delete
Q_target = 0
Q_estimate = 0

for ep in range(1,n_episodes):

    c_state = env.reset()
    dn = False
    c_noise =1

    while not dn:

        with torch.no_grad():

            det_action = agent(torch.from_numpy(c_state))

            # Use OI noise
            stochasticity = theta * (mu - c_noise) + action_std * torch.randn(1)
            c_noise += stochasticity
            #stochasticity = torch.randn(1) * action_std #0.1
            action = torch.clamp(det_action + epsilion * c_noise,-1,1)
            det_actions.append(det_action)

        n_s,rwd,dn,_ = env.step(action.numpy())


        ep_rwd.append(rwd)

        #rwd += 13 * abs(n_s[1])
        rpl_buffer.store_transition(c_state,action,rwd,n_s,dn)

        c_state = n_s


        # Check if it's time to update
        if  ep > start_update: #t%25 == 0 and

            # Randomly sample batch of transitions from buffer
            b_spl_c_state, b_spl_a, b_spl_rwd,b_spl_n_state, b_spl_done = rpl_buffer.sample_transition()


            # convert everything to tensor
            b_spl_c_state = torch.from_numpy(np.array(b_spl_c_state))
            b_spl_rwd = torch.from_numpy(np.array(b_spl_rwd))
            b_spl_n_state = torch.from_numpy(np.array(b_spl_n_state))
            b_spl_done = torch.from_numpy(np.array(b_spl_done))


            # Create input for target critic, based on next state and the optimal action there
            trg_crit_inpt = torch.cat([b_spl_n_state, target_agent(b_spl_n_state)],dim=1)#


            # Compute Q target value
            Q_target = b_spl_rwd + discount * (~b_spl_done) * critic_target(trg_crit_inpt).squeeze() # squeeze so that all dim in equation match for element-wise operations



            # Compute Q estimate
            b_spl_a = torch.stack(b_spl_a) # need to increase dim to have same size as states
            critic_nn_inpt = torch.cat([b_spl_c_state,b_spl_a],dim=1)


            Q_estimate = critic_nn(critic_nn_inpt).squeeze() # squeeze to have same dim as Q_target

            # Update critic
            critic_loss = critic_nn.update(Q_target, Q_estimate)
            critic_losses.append(critic_loss.detach())


            # Update actor
            actor_loss_inpt = torch.cat([b_spl_c_state,agent(b_spl_c_state)],dim=1)
            actor_loss = - critic_nn(actor_loss_inpt)
            agent.update(actor_loss)


            # Update target NN through polyak average
            target_agent.soft_update(agent, decay_upd)
            critic_target.soft_update(critic_nn,decay_upd)


    total_acc.append(sum(ep_rwd)/max_t_steps) # store rwd
    ep_rwd = []




    if ep % ep_print == 0: #and ep > 100:

        avr_acc = sum(total_acc) / ep_print
        det_actions = np.abs(det_actions)

        print("ep: ", ep)
        print("Aver accuracy: ",avr_acc)
        print("Critic loss: ", sum(critic_losses)/ (ep_print*max_t_steps))
        print("Action mean: ",np.mean(det_actions),'Action std: ',np.std(det_actions))

        training_acc.append(avr_acc)
        total_acc = []
        critic_losses = []
        det_actions = []
        epsilion *= eps_decay


        # Stop at a good, but not perfect policy
        # if avr_acc > -2:
        #     break

#torch.save(training_acc,"/Users/michelegaribbo/Desktop/Results/DDPG_accuracy.pt")
torch.save(agent.state_dict(),"/Users/px19783/PycharmProjects/ChaoticEnv/MountainCar/Results/ContMCarPolicy.pt")

















