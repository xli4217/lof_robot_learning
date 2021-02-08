import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# def load_dataset(exp_name, exp_num):
#     directory = Path(__file__).parent.parent / 'dataset' / exp_name
#     # if directory doesn't exist, create it
#     Path(directory).mkdir(parents=True, exist_ok=True)
#     file_name = 'results_' + str(exp_num) + '.npz'
#     path_name = directory / file_name
    
#     data = np.load(path_name, allow_pickle=True)

#     return data

def load_dataset(exp_name, method_name, task_name, test_num):
    directory = Path(__file__).parent / 'dataset' / exp_name / method_name / task_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(test_num) + '.npz'
    path_name = directory / file_name
    
    data = np.load(path_name, allow_pickle=True)

    return data['arr_0'][()]

task_reward_bounds = [(-300, 0), (-300, 0), (-300, 0)]
task_names = ['or', 'if', 'sequential', 'composite']
# task_names = ['lunchbox', 'lunchbox2', 'lunchbox3']
method_names = ['lof', 'fsa', 'greedy']
method_plot_names = ['LOF-VI', 'LOF-QL', 'Greedy']
method_colors = ['b', 'r', 'g']
# method_names = ['greedy']
# method_plot_names = ['Greedy']
# method_colors = ['g']

h_env_horizon = 50 # number of training steps per epoch
h_epochs = [i for i in range(0, 900, 50)] + [899]
h_steps = np.array(h_epochs) * h_env_horizon
rm_epochs = [i for i in range(0, 990, 10)] + [999]
rm_env_horizon = 800
rm_steps = np.array(rm_epochs) * rm_env_horizon

num_exp = 2 # number of separate training runs

def get_plot_data_for_task(task_num, task_name, num_exp=2):
    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []
    method_steps = []

    for method_name in method_names:
        first_data = load_dataset('composability', method_name, task_name, 0)
        steps = first_data['steps']
        method_steps.append(steps)
        num_data = len(steps)
        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        std_rewards = [0]*num_data

        rewards = []

        for i in range(num_exp):
            results = load_dataset('composability', method_name, task_name, i)
            rewards.append(results['reward'])
            for j, reward in enumerate(results['reward']):
                # for each experiment, average reward over the tasks
                ave_rewards[j] += reward / (num_exp)
                if reward > max_rewards[j]:
                    max_rewards[j] = reward
                if reward < min_rewards[j]:
                    min_rewards[j] = reward

        std_rewards = np.std(np.array(rewards))

        method_max_rewards.append(np.array(ave_rewards) + std_rewards)
        method_min_rewards.append(np.array(ave_rewards) - std_rewards)
        # method_max_rewards.append(max_rewards)
        # method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards, method_steps


def get_plot_data_over_tasks():
    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []
    method_steps = []

    for method_name in method_names:
        first_data = load_dataset('composability', method_name, task_names[0], 0)
        steps = first_data['steps']
        num_data = len(steps)
        method_steps.append(steps)
        
        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        std_rewards = [0]*num_data

        all_rewards = []


        for i, task_name in enumerate(task_names):
            rewards = []
            for j in range(num_exp):
            # task_ave_rewards_list = [[0 for nt in range(num_trials)] for ei in range(len(epochs))]
            # for each experiment, average reward over the tasks
            # task_ave_rewards = [0]*num_data

                results = load_dataset('composability', method_name, task_name, j)
                rewards.append(results['reward'])
            all_rewards.append(rewards)

        all_rewards = np.array(all_rewards)
        task_ave_rewards = np.average(all_rewards, axis=0)
        ave_rewards = np.average(task_ave_rewards, axis=0)
        std_rewards = np.std(task_ave_rewards, axis=0)
        print('f')
            # reward = results['reward'][k]
            # task_ave_rewards[epoch_i] += reward/(len(task_names)*num_trials)
            # task_ave_rewards_list[epoch_i][k] += reward/len(task_names)

            # for ei in range(len(epochs)):
            #     ave_reward = task_ave_rewards_list[ei][k]
            #     if ave_reward > max_rewards[ei]:
            #         max_rewards[ei] = ave_reward
            #     if ave_reward < min_rewards[ei]:
            #         min_rewards[ei] = ave_reward
                    
                
            # for n, ave_reward in enumerate(task_ave_rewards):
            #     ave_rewards[n] += ave_reward / (num_exp)

            # for epoch_i, epoch in enumerate(epochs):
            #     std_rewards[epoch_i] = np.std(task_ave_rewards_list[epoch_i])

        method_max_rewards.append(ave_rewards + std_rewards)
        method_min_rewards.append(ave_rewards - std_rewards)
        # method_max_rewards.append(max_rewards)
        # method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards, method_steps

def plot_data_over_tasks():
    method_ave_rewards, method_min_rewards, \
        method_max_rewards, method_steps = get_plot_data_over_tasks()

    for i, (method_name, method_color, ave_reward, min_reward, max_reward, steps) in enumerate(zip(
            method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards, method_steps)):
        plt.plot(steps, ave_reward, color=method_color, label=method_name)    
        plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

    plt.ylim(-30, -10)
    # plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    # plt.title('Reward Averaged over Tasks')
    plt.xlabel('Number of metapolicy retraining steps', fontsize=21)
    plt.ylabel('Average reward', fontsize=21)
    # plt.xlim((0, rm_steps[-1]))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    directory = Path(__file__).parent / 'dataset' / 'composability'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = 'results_averaged_over_tasks.png'
    path_name = directory / file_name
    legend_path = directory / 'legend.png'

    plt.savefig(path_name, bbox_inches='tight')

    legend = plt.legend(ncol=5, bbox_to_anchor=(0.5, -0.2), framealpha=1, frameon=True)

    export_legend(legend, filename=legend_path)

def plot_data_per_task():
    for i, task_name in enumerate(task_names):
        fig = plt.figure()
        method_ave_rewards, method_min_rewards, \
        method_max_rewards, method_steps = get_plot_data_for_task(i, task_name)

        for i, (method_name, method_color, ave_reward, min_reward, max_reward, steps) in enumerate(zip(
                method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards, method_steps)):
            plt.plot(steps, ave_reward, color=method_color, label=method_name)    
            plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

        plt.ylim(-40, -10)
        # plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
        plt.tight_layout()
        # plt.title("Reward for {} task".format(task_name))
        plt.xlabel('Number of metapolicy retraining steps', fontsize=21)
        plt.ylabel('Average reward', fontsize=21)
        # plt.xlim((0, rm_steps[-1]))
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        directory = Path(__file__).parent / 'dataset' / 'composability'
        # if directory doesn't exist, create it
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = 'results_{}.png'.format(task_name)
        path_name = directory / file_name

        plt.savefig(path_name, bbox_inches='tight')

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == '__main__':
    plot_data_over_tasks()
    plot_data_per_task()