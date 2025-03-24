import pandas as pd
import wandb
import yaml
import os

import numpy as np
import seaborn as sns
import shutil
# from wandb import Api
from wandb.sdk.wandb_run import Run
from algorithm.utils.logging import sha256, squash_info
from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from marl_eval.utils.diagnose_data_errors import DiagnoseData
from matplotlib import pyplot as plt
api = wandb.Api()
entity, project = "vinokur-eg-", "Wildfire"

# filter by algorithm, run 
# Two forms - summary table with each run summary: 
# Averaged over 
FIELDS = ['ID', 
          'group', 
          'name',
          'seed', 
          'partial_obs', 
          'batch_size',
# Second - at each equal step in history collect into a separate table these metrics
# HISTORY = [
        
        #    'loss',
        #    'q_taken_mean', 
        #    'td_error', 
           '_total_timesteps',
          '_runtime'
                    'train/mean_burnt trees',
          'train/mean_reward',
          'train/std_of_mean_reward',
          'train/burnt trees', #avg over 5 runs
          'train/mean_ep_length',
        'train/burnt trees', 
          'eval/burnt trees', #avg over 5 runs
        
        'eval/std_of_mean_reward',
        'eval/mean_ep_length',
          ]

HISTORY = [
          '_step',

        # 'pg_loss',
        # 'agent_0_0_critic_loss'
        'eval/mean_reward', #avg over 5 runs
        # 'eval/burnt trees',
        # 'train/burnt trees', 
        # 'train/mean_burnt trees'
        'eval/mean_burnt trees',
        'eval/mean_ep_length',

]

AGENTS = [5, 10, 15, 20]

def _load_data_from_wandb(ids=None, metric=None, date_after="2025-02-28##"):
    """Helper function for pulling results data from logs

    Args:
        folder (str):
        metric (str):
        step (int):
        step_metric (str):

    Returns:
        list of performance values
    """
    # The given folder will contain several sub-folders with random hashes like "1a8fdsk3"
    # Within each sub-folder is the data we need
    runs = api.runs(entity + "/" + project, order='-created_at', filters={"$and": [{"created_at": {
                                                                                        "$gt": date_after
                                                                                        }
                                                                                        }]})
    run: Run
    data = []
    fields = FIELDS
    fields.extend(HISTORY)


    
    for run in runs:
        if run.state == 'failed':
            continue
        # run_data = {field: run.summary.get(field, None) for field in FIELDS}
        summary = dict(run.summary._json_dict)

        

        if not 'eval/mean_reward' in run.summary and not 'train/mean_reward' in run.summary:
            continue
    
        # if run.summary.get('eval/mean_ep_length') <= 1:
        #     continue
        history_data: pd.Series = run.history(pandas=True) 
                                            #    index=['_step'], 
                                            #    columns=HISTORY[1:]
        if history_data.get('eval/burnt trees') is None and history_data.get('train/burnt trees'):
            continue
        history_data.filter(items=HISTORY)

        history_data.dropna()
        # if num_steps != expected_steps:
        #     # Skip runs with differing number of steps.
        #     continue

        # Prepare per-step data and aggregate absolute metrics.
        steps_dict = {}
        
        print("whole history: ",history_data.keys())
        print(len(history_data.index))
        # history_data = history_data.interpolate(method='index',) #'cubicspline'
        # run_data.update({field: history_data.get(field, None) for field in FIELDS})
        print(run.name)
        run_data = {}
        run_data.update({field: summary.get(field, None) for field in FIELDS if summary.get(field, None) is not None})
        run_data.update({
            'ID': run.id,
            'group': run.group,
            'name': run.name,
            'algo': run.config.get('name', None),
            'team size': run.config.get('agents', -5),
            'eval team size': run.config.get('agents_inference', -5),
            'seed': run.config.get('seed', None),
            'eval_seed': run.config.get('eval_seed', None),
            'job_type': run.job_type,
        })
        run_data.update({field: run.config.get(field, None) for field in fields if run.config.get(field, None) is not None})
        # for _, row in history_data.iterrows():
        #     row.filter(FIELDS)
        #     run_data.update(row.to_dict())
        # print(history_data.tail(1).to_dict())
        # history_data.dropna()
        history_data.filter(fields)
        # best_hist_dict = history_data.add_prefix('95% ',axis=1).quantile(0.95)
        # mean_hist_dict = history_data.add_prefix('Mean ',axis=1).mean()
        # std_hist_dict = history_data.add_prefix('Std ',axis=1).std()
        hist_dict = history_data.tail(1)
        hist_dict.dropna()
        hist_dict = history_data.to_dict()
        # hist_dict = {field: hist_dict.get(field)[history_data.last_valid_index()] for field in fields if hist_dict.get(field) is not None}
        # best_hist_dict = best_hist_dict
        print(hist_dict)
        # exit()
        run_data.update(hist_dict)
        # run_data.update(best_hist_dict.)
        
        

        # if 'video' in run.config:
        #     print("Video found:", run.config['video']['path'])
        #     artifact = run.use_artifact(run.config['video']['path'])
        #     artifact_dir = artifact.download()
        #     print("loading video to :", artifact_dir)

        # if 'config.yaml' in run.config:
        #     artifact = run.use_artifact(run.config['config.yaml'])
        #     artifact_dir = artifact.download()
        #     print("loading yaml to :", artifact_dir)
        #     with open(os.path.join(artifact_dir, 'config.yaml')) as f:
        #         config_data = yaml.safe_load(f)
        #         run_data.update(config_data)
        
        data.append(run_data)
        # last_step_metrics = run.history(keys=['_runtime', '_timestamp', '_step'], pandas=True).iloc[-1].to_dict()
        system_metrics:pd.DataFrame = run.history(
                                    keys=[
                                        "system.proc.memory.rssMB", 
                                        "system.disk.\.usageGB", 
                                        "system.proc.cpu.threads",
                                        ],
                                        stream='system', pandas=False)
        # if len(system_metrics) != 0:

        #     print("System: ", system_metrics.keys())
        # # run_data.update(last_step_metrics)
        run_data.update(system_metrics)


    runs_df = pd.DataFrame(
        data
    )

    print(runs_df.shape)
    print(runs_df)
    # runs_df = runs_df.set_index(keys=['id'])

    

    runs_df.to_csv(f"{project}_wandb.csv",index=False)
    runs_df.to_excel(f"{project}_wandb.xlsx", 'sheet_name=Format ',index=False)
    runs_df.to_json(f'{project}_wandb.json', orient='records', indent=2, )
   

# Convert to a expertiment dict for each eval step 

def load_marl_eval_history_data(equal_steps=None, output_filename='marl_eval_raw.json', algos=[], job_type='test', date_before="2026-01-01", date_after="2025-03-12##"):
    """
    Use wandb API to load run history data for runs that have an equal number of eval steps.
    Formats the data into a nested dictionary with this structure:
      {
        "env": {              # equals project name
          "task": {           # equals run.group (or a default if missing)
             "algo": {        # equals run.config.get('algo') (or 'algo_unknown')
                "run_id": {
                    "STEP_1": { "step_count": 10006, "return": [1,2,3,4], "win_rate": [0.8] },
                    "STEP_2": { ... },
                    "STEP_3": { ... },
                    "absolute_metrics": { "return": [...], "win_rate": [...] }
                },
                ...
             }
          }
        }
      }

    Args:
        equal_steps (int): Only process runs that have exactly this number of history steps.
                           If None, the first valid run defines the expected number.
        output_filename (str): JSON output filename.
    """
    import json
    data = {}
    alg = [{"config.name": str(name)} for name in algos]
    runs = api.runs(entity + "/" + project, order='-created_at',
                    filters={
                        "$and": [
                            # 
                            {"created_at": {"$lt": "{}".format(date_before),"$gt": "{}".format(date_after)}},
                            # {"job_type": job_type}
                            # {"summary_metrics.eval.mean_reward": {"$ne": None} },
                                 ],
                        "$or": 
                        # [
                        #     {"config.name": "vdn"},
                        #     {"config.name": "iql"},
                        #     {"config.name": "qmix"},
                        # ],
                        alg,
                        },
                    )
    expected_steps = equal_steps
    samples = expected_steps if expected_steps is not None else 25 
    # Initial list
    allowed_steps = list()
    steps_len = []
    for run in runs:
        
        print(run.id)
        print(run.name)
        print(run.job_type)
        if job_type is not None and run.job_type != job_type:
            continue
        # print(run._metadata['created_at'])
        print(run.config.get("agent_view_size"))
        if not 'eval/mean_reward' in run.summary:
            continue
        
        # if run.summary.get('eval/mean_ep_length', 0) <= 1:
        #     continue
        history_data: pd.Series = run.history(keys=HISTORY,samples=samples, pandas=True) 
                                            #    index=['_step'], 
                                            #    columns=HISTORY[1:]
        print(history_data.head(2))
        history_data.filter(items=HISTORY)
        print(history_data.head(2))
        # history_data.dropna()

        if not any(history_data.notna()):
            continue
        # if history_data.get("eval/mean_ep_length", [1])[0] == 1:
        #     continue
        print(history_data.values)
        num_steps = len(history_data)
        steps_len.append(history_data.size)
        # If we haven't set expected_steps, use the first valid run as our baseline.
        if expected_steps is None:
            expected_steps = num_steps

        # if num_steps != expected_steps:
        #     # Skip runs with differing number of steps.
        #     continue

        # Prepare per-step data and aggregate absolute metrics.
        steps_dict = {}
        
        print("whole history: ",history_data.keys())
        # history_data = history_data.interpolate(method='index',) #'cubicspline'
        # for i, row in history_data.enumerate()
        # if len(history_data.index) <= 1:
        #     print(len(history_data.index))
        #     continue
        for i, row in history_data.iterrows():


            step_label = f"STEP_{i+1}"
            # print(row.shape)
            # Assume keys "step_count", "return", and "win_rate" exist in history.
            row_items = {field: row.get(field, None) for field in HISTORY if row.get(field, None) is not None}
            # ret = row.get('return', None)
            # win_rate = row.get('win_rate', None)
            # Ensure these are lists.
            steps_dict[step_label] = {
            }
            print("Extracted: ", row_items.values())
            for k, v in row_items.items():
            # Nest a scalar key into a list
                print(type(v))
                if pd.notna(v):
                    if k in ['eval/mean_reward','train/mean_reward']:
                        k = 'mean_norm_return'

                    if k in [
                             'eval/burnt trees',
                             'train/burnt trees', 
                             'eval/mean_burnt trees', 
                             'train/mean_burnt trees'
                             ]:
                        
                        k = 'win_rate'
                        v = 1. - v
                    if k != '_runtime' and i not in allowed_steps:
                        allowed_steps.append(i)
                    steps_dict[step_label][k] = v
                    # row_items[k] = [v]
                

        # Determine keys for env, task (group) and algo.
        env_key = project  # project name as environment.
        # task_key = run.group if run.group is not None else "default"
        # TODO implement get_task_group method
        group_dict = {}
        group_dict.update(run.config)
        group_dict = {
            'ID': run.id,
            'group': run.group,
            'name': str(run.name),
            'algo': run.config.get('name', None),
            'team size': run.config.get('agents', -5),
            'eval team size': run.config.get('agents_inference', -5),
            'seed': run.config.get('seed', None),
            'eval_seed': run.config.get('eval_seed', None),
            'job_type': run.job_type,
        }
        # group_dict.update(run.config._json_config)
        task_key = get_run_group(group_dict)
        algo_key = run.config.get('name', 'algo_unknown')
        run_id = str(run.name)

        # Nest the output data.
        data.setdefault(env_key, {})\
            .setdefault(task_key, {})\
            .setdefault(algo_key, {})[run_id] = {}
        data[env_key][task_key][algo_key][run_id].update(steps_dict)
        # data[env_key][task_key][algo_key][run_id]["absolute_metrics"] = abs_metrics


    print("Existing steps: ", np.unique(allowed_steps))
    print("min Num of steps", min(steps_len))


    # Dump the nested dictionary to JSON.
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)


def get_run_group(run_dict):
    seed = str(run_dict.get('eval_seed', -1))
    obs = str(run_dict.get('agent_view_size', 13))
    job_type = str(run_dict['job_type'])
    agents = str(run_dict.get("team size", '-1'))
    world = str(run_dict.get('world_size', '-1'))
    # _, n_agents, size, reward, suffix = 
    
    name = run_dict["name"].split('_')
    if agents == '-1':
        agents = name[1]
    if world == '-1':
        world = '13'
    print(run_dict["name"].split('_'))
    return "_".join([seed, agents, world, job_type])


def plot_history_data(data, metrics, tasks=None, savedir='figures', savename='metrics'):
    """
    Plot history data for different metrics across different runs.
    Optionally separate graphs for different tasks.
    """
    figures = []
    if not os.path.exists(savedir):
        os.mkdir("figures")      
    plt.grid(alpha=0.3)    
    for env, env_data in data.items():
        for task, task_data in env_data.items():
            if tasks and task not in tasks:
                continue
            fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))
            for i, ax in enumerate(axes):
                
                for algo, algo_data in task_data.items():
                    for run_id, run_data in algo_data.items():
                        # steps = sorted(run_data.keys())
                        # values = [step.get(metric, np.nan) for step in run_data.keys()]
                        values = [run_data[step].get(metrics[i], np.nan) for step in run_data.keys()]
                        steps = [run_data[step]['_step'] for step in run_data.keys()]
                        # at least 3 data points
                        if values.count(np.nan) < len(values) - 3:
                            ax.plot(steps, values, label=f'{algo} - {run_id}')
                            ax.set_xlabel('Steps')
                            ax.set_ylabel(metrics[i])
                            ax.set_title(f'{metrics[i]} for Task: {task}')
            savepath = os.path.join(savedir, savename + task + ".png")
            print(f"Saving to {savepath}")
            fig.legend(bbox_to_anchor=(1.05, 1),
                        loc='lower left', borderaxespad=0.)
            figures.append(fig)
            fig.savefig(savepath, bbox_inches="tight")
    
    for fig in figures:
        fig.show()
    # df = pd.concat(hist_list, ignore_index=True)
    # df = df.query("`val/loss` != 'NaN'")


def _load_data_from_subfolder(folder, metric, step=None, step_metric=None):
    """Helper function for pulling results data from logs

    Args:
        folder (str):
        metric (str):
        step (int):
        step_metric (str):

    Returns:
        list of performance values
    """
    # The given folder will contain several sub-folders with random hashes like "1a8fdsk3"
    # Within each sub-folder is the data we need
    results = []

    for subfolder in os.listdir(folder):
        data = pd.read_csv(f'{os.path.join(folder, subfolder, "results.csv")}')

        if step is not None and step_metric is not None:
            data = [data[data[step_metric] == step][metric].tolist()[0]]

        else:
            data = data[metric].tolist()

        results.append(data)

    return results

# apply seaborne theme globally
sns.set_theme()
from matplotlib import rc
FONTSIZE = 20 # 16
TICK_FONTSIZE = 16
rc('font', size=FONTSIZE)
rc('axes', titlesize=FONTSIZE, labelsize=FONTSIZE)
rc('xtick', labelsize=TICK_FONTSIZE)
rc('ytick', labelsize=TICK_FONTSIZE)
rc('legend', fontsize=FONTSIZE)


def glob_re(pattern, strings):
    '''Given a list of strings, returns those that contain the regex pattern'''
    return filter(re.compile(pattern).search, strings)

def compute_figure_size(num_subfigs): 
    if num_subfigs == 1: 
        figsize = (6.5, 6.0)
    elif num_subfigs ==4: 
        figsize = (5.0*num_subfigs, 5.2)
    else:
        figsize = (5.0*num_subfigs, 6.0)
    return figsize       

def generate_summary(eval_paths,
                     remove_duplicates=False):
    '''Assumption: eval path is a json file'''
    eval_path_res = {
        "test_return_mean": {},
        "test_return_std": {},
        "test_ep_length_mean": {},
        "seed_paths": {} # record seeds involved in computation
    }

    for p in eval_paths: 
        seedi, seedj, n = get_seed_pair(p)
        seed_pair_str = f"seedi={seedi}_seedj={seedj}"
        eval_name = f"{seed_pair_str}_n-{n}"
        for k, stats_dict in eval_path_res.items():
            if seed_pair_str not in stats_dict:
                stats_dict[seed_pair_str] = []
                            
        if eval_name in eval_path_res["seed_paths"][seed_pair_str]:
            if remove_duplicates:
                shutil.rmtree(os.path.dirname(os.path.dirname(p)))
                print(f"Warning: removing duplicate eval {p}")
            else:
                print("Warning: duplicate eval detected: ", p)
            continue
        else:
            eval_path_res["seed_paths"][seed_pair_str].append(eval_name)
            
        with open(p) as f:
            eval_info = json.load(f)
            mean = eval_info["test_return_mean"][0]["value"]
            std = eval_info["test_return_std"][0]["value"]
            ep_len=  eval_info["test_ep_length_mean"][0]

            eval_path_res["test_return_mean"][seed_pair_str].append(mean)
            eval_path_res["test_return_std"][seed_pair_str].append(std)
            eval_path_res["test_ep_length_mean"][seed_pair_str].append(ep_len)
    # seed_pair_means = [np.mean(eval_path_res["test_return_mean"][seed_pair_str]) for seed_pair_str in eval_path_res["test_return_mean"]]
    # seed_pair_std_errors = np.std(seed_pair_means) / np.sqrt(n_trials)

    # compute standard error of the return mean from the std dev
    # unpack all return means into a single list
    returns = []
    for seed_pair_str in eval_path_res["test_return_mean"]:
        returns.extend(eval_path_res["test_return_mean"][seed_pair_str])
    n_samples = len(returns)
    std_errors = np.std(returns) / np.sqrt(n_samples)

    summary = {
        "mean": np.mean(returns),
        "ci": 1.96 * std_errors
    }
    # return summary stats and raw eval path results
    return summary, returns

def check_seed_pair_equal(eval_name):
    '''Given an eval name, check if the seeds of algo1 and algo2 are the same'''
    # last seed is the eval seed
    seed_pair = re.findall(r"seed=(\d+)", eval_name)[:2]
    if seed_pair[0] == seed_pair[1]:
        return True
    return False

def get_seed_pair(eval_name):
    '''Given an eval name, find seeds in the eval name'''
    # last seed is the eval seed
    seed_pair = re.findall(r"seed=(\d+)", eval_name)[:2]
    n = re.findall(r"n-(\d+)", eval_name)[0]
    return seed_pair[0], seed_pair[1], n


def plot_single_exp(ax, exp_dfs, baselines=None,
                    stat_name=None, 
                    xlabel=None, ylabel=None,
                    yaxis_lims=None, xaxis_lims=None, 
                    plot_title=None):
    baseline_palette = list(sns.color_palette("Oranges", as_cmap=False, n_colors=7).as_hex())
    open_palette = list(sns.color_palette("mako", as_cmap=False, n_colors=12).as_hex())
    default_palette = sns.color_palette("colorblind", as_cmap=True)

    for i, (exp_name, exp_df) in enumerate(exp_dfs.items()):
        if "baseline" in exp_name: 
            color = baseline_palette.pop()
            baseline_palette.pop()
        elif "open" in exp_name: 
            color = open_palette[i*2 + 3]
        else: 
            color = default_palette[i]
        g = sns.lineplot(data=exp_df, 
                     x="ts", y="test_battle_won_mean" if stat_name is None else stat_name,
                     errorbar=('ci', 95), # "sd", 
                     ax=ax, label=exp_name, 
                     color=color,
                     legend=False
                    )
    g.set(xlabel=xlabel, ylabel=ylabel, title=plot_title, ylim=yaxis_lims, xlim=xaxis_lims)
    if baselines is not None:
        color_list = ["darkslategray", "slateblue", "darkslateblue", "indigo", "darkviolet"]
        # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        linestyle_list = [
             ('densely dashed',        (0, (5, 1))),
             ('loosely dotted',        (0, (1, 10))),
             ('dotted',                (0, (1, 1))),
             ('loosely dashed',        (0, (5, 10))),
             ('long dash with offset', (5, (10, 3))),
              ]
        for i, (baseline_nm, baseline_data) in enumerate(baselines.items()):
            bs_mean, bs_std = baseline_data
            ax.axhline(y=bs_mean, xmin=0.01, xmax=0.99, 
                       color=color_list[0], # figure_colors[i+1], 
                       linestyle= linestyle_list.pop(-1)[1], 
                       label=baseline_nm
                       )

def plot_learning_curves(exp_dfs:dict, 
                     savename:str, 
                     plot_title:str=None, 
                     stat_name:str=None,
                     legend=True,
                     legend_cols=4,
                     legend_loc=(1.0, -0.25),
                     baselines:dict=None, 
                     xaxis_lims=None,
                     yaxis_lims=None,
                     save=False, 
                     savedir="figures/"):
    '''
    baselines: a dict with format {exp_name: [mean, std]}. will be plotted as horizontal line
    '''
    _, axis = plt.subplots(1, 1, figsize=(7, 5)) # figsize argument

    plot_single_exp(axis, exp_dfs, baselines,
                    stat_name=stat_name, 
                    xlabel="Timesteps", 
                    ylabel=stat_name.replace("_", " ").title() if stat_name is not None else "Mean Test Return",
                    yaxis_lims=yaxis_lims, xaxis_lims=xaxis_lims, 
                    plot_title=plot_title,
                    )

    if legend:
        plt.legend(bbox_to_anchor=legend_loc, 
                   borderaxespad=0.,
                   ncol=legend_cols,
                  )
    else:
        leg = axis.get_legend()
        leg.remove()
    
    if save:
        if not os.path.exists(savedir):
            os.mkdir("figures")            
        savepath = os.path.join(savedir, savename + ".pdf")
        
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_learning_curves_all(exp_dict:dict, 
                     savename:str, 
                     plot_suptitle:str=None, 
                     show_subplot_task_name=True,
                     stat_name:str=None,
                     legend=True,
                     legend_cols=4,
                     legend_loc=(1.0, -0.25),
                     baselines:dict=None, 
                     xaxis_lims=None,
                     yaxis_lims=None,
                     save=False, 
                     savedir="figures/"):
    '''
    exp_dict: a dict with format {task: {exp_name: exp_df}}
    baselines: a dict with format {task: {exp_name: [mean, std]}}. 
    will be plotted as horizontal line
    '''
    ntasks =len(exp_dict)
    fig, axes = plt.subplots(1, ntasks, 
                             figsize=(6*ntasks, 4.3), 
                            squeeze=True) # figsize argument
    for i, (task, exp_df) in enumerate(exp_dict.items()):
        task = task.split("/")[0]
        ylabel = stat_name.replace("_", " ").title() if stat_name is not None else "Mean Test Return"
        plot_single_exp(axes[i] if ntasks > 1 else axes, 
                        exp_df, 
                        baselines[task],
                        stat_name=stat_name, 
                        xlabel="Timesteps", 
                        ylabel=ylabel if i == 0 else None,
                        yaxis_lims=yaxis_lims, xaxis_lims=xaxis_lims, 
                        plot_title=task if show_subplot_task_name else None)

    if legend:
        plt.legend(borderaxespad=0., ncol=legend_cols)
    if plot_suptitle:
        plt.suptitle(plot_suptitle)

    if save:
        if not os.path.exists(savedir):
            os.mkdir("figures")            
        savepath = os.path.join(savedir, savename + ".pdf")
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()



# if __name__ == '__main__':
#     _load_data_from_wandb()

# Example usage:

files = ['vdn_qmix_iql_train.json']

algos = ['vdn', 'iql', 'qmix', 'vdn_ns', 'iql_ns', 'qmix_ns']

steps = 20

if __name__ == '__main__': 
    
    
    # _load_data_from_wandb(date_after="2025-02-28##")
    # Uncomment to regenerate data
    # Date after is a MongoDB regex
    # load_marl_eval_history_data(equal_steps=steps, output_filename=files[0], job_type="train", algos=algos, date_before="2025-04-24##",  date_after="2025-03-16##")


    # # Load and process experiment outputs
    raw_dict = load_and_merge_json_dicts(files)

    # metrics = ['eval/mean_reward', 'eval/mean_burnt trees']
    metrics = ['mean_norm_return','win_rate']
    tasks = ['3456_10_13_train', '2004_10_13_train', '8007_10_13_train']
    plot_history_data(raw_dict, metrics, tasks)
    
    # exit()
    processed_data = Plotting.process_data(raw_dict)
    # print(processed_data)
    # plot_learning_curves_all(processed_data, savename="_".join(algos))
    doctor = DiagnoseData(raw_data=processed_data)
    print(doctor.check_data())

    # print(doctor.check_runs(num_runs=[3]))
    # print(doctor.check_algo(['vdn', 'qmix', 'iql']))
    # print(doctor.check_metric(['eval/mean_reward']))
    # exit()
    
    Plotting.task_sample_efficiency_curves(
        processed_data=processed_data, env=project, task=tasks[0]
    )
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name=project)



    # Plotting
    Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )


    Plotting.probability_of_improvement(
        environment_comparison_matrix,
        algorithms_to_compare=[["iql", "vdn"]],
    )
    plt.show()
