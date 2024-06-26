from dynamics import QuadrotorEnv, render
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys, os
from gym import spaces

def test(args):
    env = QuadrotorEnv(args)
    cwd = os.getcwd()

    action_space_tf = spaces.Box(low=np.array([-0.3, -0.3, -25.0]), high=np.array([0.3, 0.3, 0.0]), shape=(3,))
    action_space_tr = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))

    agent_tf = SAC(14, action_space_tf, args)
    agent_tr = SAC(18, action_space_tr, args)

    path_tf = os.path.join(cwd, 'src/RL_gazebo_drone/scripts/checkpoints/takeoff_0316_700')
    path_tr = os.path.join(cwd, 'src/RL_gazebo_drone/scripts/checkpoints/tracking_NED_15m_50hz_01')

    agent_tf.load_model(path_tf)
    agent_tr.load_model(path_tr)

    state = np.zeros((14,))
    done = False
    states = []
    actions = []
    angles = []
    uni_states = []
    while not done:
        s = state[:3].copy() # save the position
        s[2] = -s[2]
        states.append(s)

        state[10:] = [0, 0, 0, 0] # for takeoff policy use, set uni pos vel to 0
        state_tr = np.concatenate([state, np.zeros(4,)])
        uni_fur_pos, _ = env.compute_uni_future_traj(4)
        rel_pos = (uni_fur_pos - state[:2].reshape(-1, 1)).flatten('F')
        state_tr[10:] = rel_pos

        
        
        uni_states.append(env.get_unicycle_state(env.steps))
        action = agent_tf.select_action(state, eval=True)

        if env.steps > 300:
            action = agent_tr.select_action(state_tr)
            
        next_state, reward, done, _ = env.step(action)
        state = next_state.copy()
        a = action.copy()
        a[2] = -a[2]
        actions.append(a)
        q = np.array(state[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])
        state = next_state
        env.steps += 1

    actions = np.array(actions)
    fig = plt.figure()
    plt.plot(actions[:, 0], label='ux', color='darkviolet')
    plt.plot(actions[:, 1], label='uy', color='darkorange')
    plt.plot(actions[:, 2], label='uz', color='dodgerblue')
    plt.legend()
    plt.show()
        
    

    states = np.array(states)

    fig = plt.figure()
    plt.plot(states[:, 0], label='x', color='darkviolet')
    plt.plot(states[:, 1], label='y', color='darkorange')
    plt.plot(states[:, 2], label='z', color='dodgerblue')
    plt.legend()
    plt.show()
    angles = np.array(angles)
    uni_states = np.array(uni_states)
    render(states, angles, uni_states, actions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--num_episodes', type=int, nargs='?', default=800, help='total number of episode')
    parser.add_argument('--updates_per_step', type=int, nargs='?', default=1, help='total number of updates per step')
    parser.add_argument('--batch_size', type=int, nargs='?', default=256, help='batch size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--hidden_size', type=int, nargs='?', default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('seed', type=int, nargs='?', default=12345, help='random seed')
    parser.add_argument('--gamma', type=float, nargs='?', default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, nargs='?', default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, nargs='?', default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--target_update_interval', type=int, nargs='?', default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, nargs='?', default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--lr', type=float, nargs='?', default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lam_a', type=float, nargs='?', default=10.0, metavar='G', help='action temporal penalty coefficient (set to 0 to disable smoothness penalty)')
    parser.add_argument('--policy', default="Gaussian", type=str,  nargs='?', help='Policy Type: Gaussian | Deterministic')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='tracking', type=str, help='')
    parser.add_argument('--load_model', default=False, type=bool, help='load trained model for train function')

    parser.add_argument('--load_model_path', default='/home/adrian/RL_ws/src/RL_gazebo_drone/scripts/checkpoints/tracking_NED_15m_50hz_01', type=str, help='path to trained model (caution: do not use it for model saving)')
    
    # parser.add_argument('--load_model_path', default='checkpoints/sac_checkpoint_Quadrotor_episode2000_mode_tracking', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')
    

    args = parser.parse_args()
    test(args)