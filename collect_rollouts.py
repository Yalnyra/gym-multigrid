import pandas as pd
import wandb
import yaml
import os

import numpy as np
# from wandb import Api
from wandb.sdk.wandb_run import Run
from algorithm.utils.logging import sha256, squash_info

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
        'train/mean_reward', 'train/mean_burnt trees', 'eval/burnt trees', 'pg_loss', 'agent_0_0_critic_loss',
        #    'loss',
          'eval/mean_reward', #avg over 5 runs
          'eval/std_of_mean_reward',
          'eval/burnt trees', #avg over 5 runs
          'eval/mean_burnt trees',
          'eval/mean_ep_length',
        #    'q_taken_mean', 
        #    'td_error', 
           '_total_timesteps',
           'pg_loss',
          '_runtime']
# Export video

# Convert to a expertiment dict for each eval step (Optuonal)

def _load_data_from_wandb(ids=None, metric=None):
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
    runs = api.runs(entity + "/" + project, order='-created_at', filters={"$and": [{"created_at": {"$gt": "2025-03-10##"}}]})
    run: Run
    data = []
    for run in runs:
        if run.state != 'failed':
            # run_data = {field: run.summary.get(field, None) for field in FIELDS}
            summary = dict(run.summary._json_dict)
            if len(summary) <= 1:
                continue
            history_data: pd.Series = run.history(pandas=True)
            print("Did we find it: ", history_data.get('eval/burnt trees'))
            
            if history_data.get('eval/burnt trees') is None:
                continue
            # if len(history_data) == 0:
            #     continue
            # run_data.update({field: history_data.get(field, None) for field in FIELDS})
            print(run.name)
            run_data = {}
            run_data.update({field: summary.get(field, None) for field in FIELDS if summary.get(field, None) is not None})
            run_data.update({
                'ID': run.id,
                'group': run.group,
                'name': run.name,
                'seed': run.config.get('seed', None),
                'eval_seed': run.config.get('eval_seed', None),
                'job_type': run.job_type,
            })
            run_data.update({field: run.config.get(field, None) for field in FIELDS if run.config.get(field, None) is not None})
            # for _, row in history_data.iterrows():
            #     row.filter(FIELDS)
            #     run_data.update(row.to_dict())
            # print(history_data.tail(1).to_dict())
            hist_dict = history_data.tail(1)
            hist_dict.dropna()
            hist_dict = history_data.to_dict()
            hist_dict = {field: hist_dict.get(field)[history_data.last_valid_index()] for field in FIELDS if hist_dict.get(field) is not None}
            print(hist_dict)
            # exit()
            run_data.update(hist_dict)
            
            

            if 'video' in run.config:
                print("Video found:", run.config['video']['path'])
                artifact = run.use_artifact(run.config['video']['path'])
                artifact_dir = artifact.download()
                print("loading video to :", artifact_dir)

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
                                          stream='system', pandas=True)
            if len(system_metrics) != 0:

                print("System: ", system_metrics.keys())
            # run_data.update(last_step_metrics)
                run_data.update(system_metrics)
        # .summary contains output keys/values for
        # metrics such as accuracy.
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        # config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    runs_df = pd.DataFrame(
        data
    )

    print(runs_df.shape)
    print(runs_df)
    # runs_df = runs_df.set_index(keys=['id'])

    

    runs_df.to_csv(f"{project}_wandb.csv",index=False)
    runs_df.to_excel(f"{project}_wandb.xlsx", 'sheet_name=Format ',index=False)
    runs_df.to_json(f'{project}_wandb.json', orient='records')
   
   
   # for subfolder in folder:
    #     # data = pd.read_csv(f'{os.path.join(folder, subfolder.summa, "results.csv")}')


    #     if step is not None and step_metric is not None:
    #         data = [data[data[step_metric] == step][metric].tolist()[0]]

    #     else:
    #         data = data[metric].tolist()

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


def make_agg_metrics_intervals(folders, algos, metric, step=None, step_metric=None):
    """Pulls results for the 'Aggregate metrics with 95% Stratified Bootstrap CIs' plot
    Can also be used for "Performance Profiles" plot

    Below is an example usage for this function:
        make_agg_metrics_intervals(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
            step=[240, 240, 500, 500],
            step_metric=['environment_steps', 'environment_steps', 'updates', 'updates']
        )

    Shape of the output data is {'algo_1': (n_runs x n_envs), ..., 'algo_j': (n_runs x n_envs}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dict of performance matrices
    """
    # For the interval estimates plot, we need performance at a specific point during training/evaluation
    if step is None:
        raise ValueError('For interval plots, a specific step must be specified')
    if step_metric is None:
        raise ValueError('For interval plots, a specific step_metric must be specified')

    # Process for reading in the data
    results = {}

    for i in range(len(folders)):
        data = _load_data_from_subfolder(os.path.join(folders[i], algos[i]), metric[i], step[i], step_metric[i])

        if algos[i] not in results.keys():
            results[algos[i]] = []

        results[algos[i]].append(data)

    # Now we need to transpose the pulled results into results matrices. For specific shape, see function docstring
    results_T = {}

    for algo in results.keys():
        pulled_results = results[algo]
        results_T[algo] = np.array(pulled_results).T[0]

    return results_T


def make_agg_metrics_pxy(folders, algos, metric, step=None, step_metric=None):
    """Pulls results for the 'Probability of Improvement' plot

    Below is an example usage for this function:
        make_agg_metrics_pxy(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
            step=[240, 240, 500, 500],
            step_metric=['environment_steps', 'environment_steps', 'updates', 'updates']
        )

    Shape of the output data is {'algo_1,algo_2': ((n_runs x n_envs), (n_runs x n_envs)), ...}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dicts of comparative performance matrices
    """
    # First pulling the metrics as we would for other single-value plots
    agg_metrics = make_agg_metrics_intervals(folders=folders, algos=algos, metric=metric,
                                             step=step, step_metric=step_metric)

    # Now building out the combinatorics dict
    results = {}

    for i in range(len(algos)):
        for j in range(len(algos)):
            if i == j:
                continue
            results[f'{algos[i]},{algos[j]}'] = (agg_metrics[algos[i]], agg_metrics[algos[j]])

    return results


def make_agg_metrics_efficiency(folders, algos, metric):
    """Pulls results for the 'Aggregate metrics with 95% Stratified Bootstrap CIs' plot
    Can also be used for "Performance Profiles" plot

    Below is an example usage for this function:
        make_agg_metrics_efficiency(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
        )

    Shape of the output data is {'algo_1': (n_runs x n_envs x n_steps), ...,}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dict of performance matrices
    """
    step = [None for _ in range(len(algos))]
    step_metric = [None for _ in range(len(algos))]

    # Process for reading in the data
    results = {}

    for i in range(len(folders)):
        data = _load_data_from_subfolder(os.path.join(folders[i], algos[i]), metric[i], step[i], step_metric[i])

        if algos[i] not in results.keys():
            results[algos[i]] = []

        results[algos[i]].append(data)

    results_T = {}

    for algo in results.keys():
        pulled_results = results[algo]

        n_envs = len(pulled_results)
        n_runs = len(pulled_results[0])
        n_steps = len(pulled_results[0][0])


        results_T[algo] = np.array(pulled_results).reshape((n_runs, n_envs, n_steps))

    return results_T



if __name__ == '__main__':
    _load_data_from_wandb()
