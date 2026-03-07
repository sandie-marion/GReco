from training import lr_CIFAR10_Purchase100, lr_MNIST
from utility import set_seed
from models import get_model
from defenses import Aggregator
from attacks import Attack
from worker_datasets import worker_distributions, load_data, heterogeneous_distributions
from training import stochastic_heavy_ball
from workers import Workers
import os
import glob
import gc
import torch
from multiprocessing import Pool
import itertools
from os.path import join 
import argparse 
import json


def get_attack_parameters(kwargs):
    if kwargs['attack_name'] == 'ALIE':
        return {'n_workers': kwargs['n_workers'], 'f': kwargs['n_byzantine_workers']}
    elif kwargs['attack_name'] == 'FOE':
        return {'epsilon': 0.1}
    elif kwargs['attack_name'] == 'Mimic':
        return {'worker_id_to_duplicate':0}
    elif kwargs['attack_name'] == 'NNP':
        return {'n': kwargs['n_workers'], 'f': kwargs['n_byzantine_workers']}
    elif kwargs['attack_name'] == 'LF' :
        return {'n_classes' : kwargs['n_classes'], 
                'shift' : 2, 
                'reg_param' : kwargs['reg_param'], 
                'clip_param' : kwargs['clip_param'], 
                'beta' : kwargs['beta']}
    elif kwargs['attack_name'] == 'PLF' : 
        return {'true_label' : 0, 
                'false_label' : 2, 
                'reg_param' : kwargs['reg_param'], 
                'clip_param' : kwargs['clip_param'], 
                'beta' : kwargs['beta']
                }
    else:
        return {}

def get_aggregator_parameters(kwargs):
    if kwargs['aggregator_name'] == 'CwTM':
        return {'q': kwargs['n_byzantine_workers']}
    elif kwargs['aggregator_name'] == 'RFA':
        return {'T': 10, 'nu': 0.1}
    elif kwargs['aggregator_name'] == 'Krum':
        return {'f':kwargs['n_byzantine_workers']}
    elif kwargs['aggregator_name'] == 'GAS' : 
        return {'f':kwargs['n_byzantine_workers'], 
                'gas_p' : 1000, 
                'base_agg' : 'krum'}
    elif kwargs['aggregator_name'] == 'Bulyan' : 
        return {'f':kwargs['n_byzantine_workers']}
    elif kwargs['aggregator_name'] == 'HullGuard':
        return {'C': kwargs['n_classes'], 'h': kwargs['n_honest_workers'], 'f': kwargs['n_byzantine_workers']}
    else:
        return {}
        
def get_pre_aggregator_parameters(kwargs):
    if kwargs['pre_aggregator_name'] == 'NNM':
        return {'f':kwargs['n_byzantine_workers']}
    
    elif kwargs['pre_aggregator_name'] == 'BKT':
        s = int(kwargs['n_workers']/(2*kwargs['n_byzantine_workers'])) if kwargs['n_byzantine_workers']>0 else None    
        return {'s': s}
    
    elif kwargs['pre_aggregator_name'] == 'FoundFL':
        m = 1
        return {'m': m}
    
    elif kwargs['pre_aggregator_name'] == 'ARC' :
        return {'f' : kwargs['n_byzantine_workers']}
    
    else:
        return {}

def get_criterion_parameters(kwargs):
    if kwargs['criterion_name'] == 'FedLC':
        return {'tau': 1}
    
    elif kwargs['criterion_name'] == 'DMFL':
        return {'T': 1}

    else:
        return {}  




def get_id(kwargs):
    
    exclude = {
        'n_classes', 'attack_parameters', 'aggregator_parameters',
        'pre_aggregator_parameters', 'criterion_parameters',
        'n_step', 'reg_param', 'clip_param', 'lr', 'experiment_folder', 'n_experiments',
        'heterogeneous_distribution', 'seed', 'training_parameters', 'device', 'n_honest_workers', 'gpu_list'
    }
    
    experiment_id = '_'.join(str(v) for k, v in kwargs.items() if k not in exclude)
    experiment_id += '_'

    return experiment_id
    
def get_completed_experiments(experiment_folder: str) -> list:
    """
    Retrieves the list of already computed experiments from stored statistics files.

    Parameters:
    experiment_folder (str): The path to the folder containing experiment results.

    Returns:
    List[int]: A list of experiment indices that have been computed.
    """
    files = sorted(glob.glob(os.path.join(experiment_folder, "*_statistics.pt")))
    
    return [os.path.basename(file).replace('statistics.pt', '') for file in files]

def split_list(lst: list, s: int) -> list:
    """
    Splits a list into sublists of size `s`.
    For instance, [0, 1, 2, 3, 4] with s=2 returns [[0, 1], [2, 3], [4]]
    """
    return [lst[i:i + s] for i in range(0, len(lst), s)]

def save(data: dict, name: str, experiment_id: int, experiment_folder:str) -> None:
    """
    Saves the dictionary to a file in the experiment_folder directory.

    Creates the experiment_folder directory if it doesn't exist and saves the provided
    data to a file named based on the experiment_id and name.
    """
    os.makedirs(experiment_folder, exist_ok=True)
    run_name = str(experiment_id) + name
    torch_path = join(experiment_folder, run_name + ".pt")
    
    torch.save(data, torch_path)



def run(kwargs: dict) -> None:
    """
    Runs an experiment with the given configuration.
    """
    # Extract parameters
    experiment_id = kwargs['experiment_id']
    n_experiments = kwargs['n_experiments']
    
    seed = kwargs['seed']
    device = kwargs['device']
    
    dataset_name = kwargs['dataset_name']
    n_classes = kwargs['n_classes']
    alpha = kwargs['alpha']
    n_workers = kwargs['n_workers']
    n_honest_workers = kwargs['n_honest_workers']
    n_byzantine_workers = kwargs['n_byzantine_workers']
    
    aggregator_name = kwargs['aggregator_name']
    aggregator_parameters = kwargs['aggregator_parameters']
    pre_aggregator_name = kwargs['pre_aggregator_name']
    pre_aggregator_parameters = kwargs['pre_aggregator_parameters']

    attack_name = kwargs['attack_name']
    attack_parameters = kwargs['attack_parameters']
    
    criterion_name = kwargs['criterion_name']
    criterion_parameters = kwargs['criterion_parameters']
    batch_size = kwargs['batch_size']
    beta = kwargs['beta']
    
    n_step = kwargs['n_step']
    reg_param = kwargs['reg_param']
    clip_param = kwargs['clip_param']
    lr = kwargs['lr']
    experiment_folder = kwargs['experiment_folder']


    # Log start
    print('Experiment', experiment_id, '/', n_experiments, 'starts.')
    set_seed(seed)
    
    # Initialize the model
    model = get_model(dataset_name, device)

    # Aggregator
    aggregator = Aggregator(aggregator_name, aggregator_parameters, pre_aggregator_name, pre_aggregator_parameters) 

    # Attack
    if attack_name=='PoisonedFL':
        attack = Attack(attack_name = attack_name, **{'net':model})   
    elif attack_name=='NNP':
        attack = Attack(attack_name = attack_name, **{ 'f':n_byzantine_workers, 'n':n_workers, 'robust_aggregator':aggregator, 'net':model})   
    else:
        attack = Attack(attack_name = attack_name, **attack_parameters)   
    
    if kwargs['heterogeneous_distribution'] : 
        print("distribution dirichlet per class")
        honest_distributions, Byzantine_distribution = heterogeneous_distributions(n_honest_workers, n_byzantine_workers, alpha, n_classes, dataset_name)
    else :
    # Local label distributions
        print("distribution dirichlet per client")
        honest_distributions, Byzantine_distribution = worker_distributions(n_honest_workers, n_byzantine_workers, alpha, n_classes, dataset_name)
    
    kwargs['distributions honest workers'] = honest_distributions  # Store distributions for reproducibility
    kwargs['distribution byzantine workers'] = Byzantine_distribution  # Store distributions for reproducibility

    class_dist = torch.sum(honest_distributions, 0) 
    tot = torch.sum(class_dist) 

    prop = class_dist / tot 
    prop = prop.tolist() 

    # Data loaders for workers
    honest_worker_loaders, test_loader = load_data(honest_distributions, batch_size, dataset_name, kwargs['heterogeneous_distribution'])
    byzantine_worker_loaders, _ = load_data(Byzantine_distribution, batch_size, dataset_name, kwargs['heterogeneous_distribution'])

    worker_loaders = honest_worker_loaders + byzantine_worker_loaders # Concatenate the two lists of loaders

    # Update criterion_parameters for workers
    criterion_parameters['device'] = device
    criterion_parameters['local_distributions'] = honest_distributions
    criterion_parameters['Byzantine_local_distribution'] = Byzantine_distribution
    
    # Initialize workers
    workers = Workers(n_honest_workers, n_byzantine_workers, worker_loaders, criterion_name, criterion_parameters, model)
    
    # Save experiment parameters
    kwargs_to_save = {k: v for k, v in kwargs.items() if k != 'lr'} # Exclude 'lr' because it is a function and should not be saved
    save(data = kwargs_to_save, name = 'kwargs', experiment_id = experiment_id, experiment_folder = experiment_folder)

    # Run the federated learning experiment
    print('Training', experiment_id, '/', n_experiments, ' starts.')
    stochastic_heavy_ball(model, workers, aggregator, attack, test_loader, prop, kwargs)

    # Log end
    print('Experiment ', experiment_id, 'ends.')


def load_config(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    return ( load_config(args.config) )


def multiple_exp () : 
    # variable_parameters = {
    #                         'attack_name': ['ALIE','FOE','Mimic','SF','PoisonedFL','MinMax','MinSum','NNP'],
    #                         'aggregator_name': ['CWMed','CwTM','RFA','Krum','Mean'],
    #                         'pre_aggregator_name': ['None','NNM','BKT','FoundFL'],
    #                         'criterion_name': ['CrossEntropy','FedLC','DMFL'],
    #                         'dataset_name': ['Purchase100', 'MNIST', 'CIFAR10', 'Fashion_MNIST'],
    #                     }

    variable_parameters = parse_args() 
    gpu_list = variable_parameters["gpu_list"]
    gpu_selection = 0

    variable_parameters.pop("gpu_list") 
    

    constant_parameters = {
                    'n_workers': 60,
                    'batch_size': 32,
                    'reg_param':1e-4,
                    'clip_param': 5,
                    'beta': 0.9,
                    'seed': 1,
                    'experiment_folder':'test_hullguard',
                    'heterogeneous_distribution' : 0,
                }
    
    experiment_folder = constant_parameters['experiment_folder']
    already_done = get_completed_experiments(experiment_folder)
    
    combinations = []
    
    keys = list(variable_parameters)
    for values in itertools.product(*map(variable_parameters.get, keys)):
        combinations.append(dict(zip(keys, values)))
    

    n_experiments = len(combinations)

    experiments = []

    for idx, parameters in enumerate(combinations) : 
        all_parameters = parameters | constant_parameters
        all_parameters['n_honest_workers'] = all_parameters['n_workers'] - all_parameters['n_byzantine_workers']
        infered_parameters = {}

        if (parameters['dataset_name'] == 'Purchase100') :
            n_classes = 100 
        elif (parameters['dataset_name'] == 'EMNIST') :
            n_classes = 62
        else : 
            n_classes = 10 

        all_parameters['n_classes'] = n_classes

        if torch.cuda.is_available(): 
            n_gpu = gpu_list[gpu_selection%len(gpu_list)]
            device = torch.device(f"cuda:{n_gpu}")
            all_parameters['device'] = device
            
            gpu_selection += 1
        else : 
            all_parameters['device'] = 'cpu'

        attack_parameters = get_attack_parameters(all_parameters)
        infered_parameters['attack_parameters'] = attack_parameters

        aggregator_parameters = get_aggregator_parameters(all_parameters)
        infered_parameters['aggregator_parameters'] = aggregator_parameters

        pre_aggregator_parameters = get_pre_aggregator_parameters(all_parameters)
        infered_parameters['pre_aggregator_parameters'] = pre_aggregator_parameters

        criterion_parameters = get_criterion_parameters(all_parameters)
        infered_parameters['criterion_parameters'] = criterion_parameters

        experiment_id = get_id(all_parameters)
        infered_parameters['experiment_id'] = experiment_id
        infered_parameters['n_experiments'] = n_experiments
        
        if all_parameters['dataset_name'] == 'MNIST' or all_parameters['dataset_name'] == 'EMNIST' or all_parameters['dataset_name'] == 'Fashion_MNIST' :
            infered_parameters['n_step'] =  501
            infered_parameters['lr'] = lr_MNIST
        else:
            infered_parameters['n_step'] =  1501 
            infered_parameters['lr'] = lr_CIFAR10_Purchase100

        if experiment_id not in already_done : 
            kwargs = all_parameters | infered_parameters
            experiments.append(kwargs)

    print("NB EXP :", len(experiments))
    how_many_in_parallel = 2
    mini_batch_of_combinations = split_list(experiments, how_many_in_parallel)


    torch.multiprocessing.set_start_method('spawn')

    for combination_batch in mini_batch_of_combinations:
        pool = Pool()
        pool.map(run, combination_batch)
        pool.close()
        pool.join()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__" :     
    multiple_exp()