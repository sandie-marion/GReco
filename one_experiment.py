from training import lr_CIFAR10_Purchase100, lr_MNIST
from utility import set_seed
from models import get_model
from defenses import Aggregator
from attacks import Attack
from worker_datasets import worker_distributions, load_data
from training import stochastic_heavy_ball
from workers import Workers
import os
import glob
import gc
import torch
from multiprocessing import Pool
import itertools
from os.path import join 



def get_attack_parameters(kwargs):
    if kwargs['attack_name'] == 'ALIE':
        return {'n_workers': kwargs['n_workers'], 'f': kwargs['n_byzantine_workers']}
    elif kwargs['attack_name'] == 'FOE':
        return {'epsilon': 0.1}
    elif kwargs['attack_name'] == 'Mimic':
        return {'worker_id_to_duplicate':0}
    elif kwargs['attack_name'] == 'NNP':
        return {'n': kwargs['n_workers'], 'f': kwargs['n_byzantine_workers']}
    else:
        return {}

def get_aggregator_parameters(kwargs):
    if kwargs['aggregator_name'] == 'CwTM':
        return {'q': kwargs['n_byzantine_workers']}
    elif kwargs['aggregator_name'] == 'RFA':
        return {'T': 10, 'nu': 0.1}
    elif kwargs['aggregator_name'] == 'Krum':
        return {'f':kwargs['n_byzantine_workers']}
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

    training_function = kwargs['training_function']
    

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
        
    # Local label distributions
    honest_distributions, Byzantine_distribution = worker_distributions(n_honest_workers, n_byzantine_workers, alpha, n_classes, dataset_name)
    
    kwargs['distributions honest workers'] = honest_distributions  # Store distributions for reproducibility
    kwargs['distribution byzantine workers'] = Byzantine_distribution  # Store distributions for reproducibility

    # Data loaders for workers
    honest_worker_loaders, test_loader = load_data(honest_distributions, batch_size, dataset_name)
    byzantine_worker_loaders, _ = load_data(Byzantine_distribution, batch_size, dataset_name)
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
    training_function(model, workers, aggregator, attack, test_loader, kwargs)

    # Log end
    print('Experiment ', experiment_id, 'ends.')

def one_exp () : 
    kwargs = {
                        'n_workers': 10,
                        'batch_size': 32,
                        'reg_param':1e-4,
                        'clip_param': 5,
                        'how_many_in_parallel': 60,
                        'beta': 0.9,
                        'lr' : lr_MNIST,

                        'n_step' : 100,
                        'seed': 1,
                        'alpha': 1,
                        'n_byzantine_workers' : 4,
                        'attack_name': 'ALIE',
                        'aggregator_name': 'CwTM',
                        'pre_aggregator_name': 'ARC',
                        'criterion_name': 'CrossEntropy',
                        'training_function': stochastic_heavy_ball,

                        'attack_parameters' : {'n_workers': 10, 'f': 4},
                        'aggregator_parameters' : {'q' : 4},
                        'pre_aggregator_parameters' : {'f':4},
                        'criterion_parameters' : {},

                        'dataset_name': 'MNIST', # 'CIFAR10', 'Fashion_MNIST', 'MNIST', 'Purchase100'
                        'experiment_folder':'test',
                        'experiment_id': 'ARC_',
                        'n_experiments' : 1,
                        'device' : 'cpu',
                        'n_classes' : 10, 
                        'n_honest_workers' : 6,
                    }

    run (kwargs)  

    kwargs = {
                        'n_workers': 10,
                        'batch_size': 32,
                        'reg_param':1e-4,
                        'clip_param': 5,
                        'how_many_in_parallel': 60,
                        'beta': 0.9,
                        'lr' : lr_MNIST,

                        'n_step' : 100,
                        'seed': 1,
                        'alpha': 1,
                        'n_byzantine_workers' : 4,
                        'attack_name': 'ALIE',
                        'aggregator_name': 'CwTM',
                        'pre_aggregator_name': 'NNM',
                        'criterion_name': 'CrossEntropy',
                        'training_function': stochastic_heavy_ball,

                        'attack_parameters' : {'n_workers': 10, 'f': 4},
                        'aggregator_parameters' : {'q' : 4},
                        'pre_aggregator_parameters' : {'f':4},
                        'criterion_parameters' : {},

                        'dataset_name': 'MNIST', # 'CIFAR10', 'Fashion_MNIST', 'MNIST', 'Purchase100'
                        'experiment_folder':'test',
                        'experiment_id': 'NNM_',
                        'n_experiments' : 1,
                        'device' : 'cpu',
                        'n_classes' : 10, 
                        'n_honest_workers' : 6,
                    }


    run (kwargs)  


    


def multiple_exp () : 
    variable_parameters = {
                            'attack_name': ['ALIE','FOE','Mimic','SF','PoisonedFL','MinMax','MinSum','NNP'],
                            'aggregator_name': ['CWMed','CwTM','RFA','Krum','Mean'],
                            'pre_aggregator_name': ['None','NNM','BKT','FoundFL'],
                            'criterion_name': ['CrossEntropy','FedLC','DMFL'],
                            'training_name': ["stochastic_heavy_ball", "gradient_approx_training"],
                            'dataset_name': ['Purchase100', 'MNIST', 'CIFAR10', 'Fashion_MNIST'],
                        }
    

    constant_parameters = {
                    'n_workers': 17,
                    'n_byzantine_workers' : 2,
                    'batch_size': 32,
                    'reg_param':1e-4,
                    'clip_param': 5,
                    'beta': 0.9,

                    'seed': 1,
                    'alpha': 1,

                    'experiment_folder':'test',
                    'experiment_id': 1,
                    'n_experiments' : 1,
                    'n_classes' : 10, 
                }
    
    combinations = []
    
    keys = list(variable_parameters)
    for values in itertools.product(*map(variable_parameters.get, keys)):
        combinations.append(dict(zip(keys, values)))


    n_experiments = len(combinations)

    experiments = []

    gpu_selection = 0

    for parameters in combinations : 
        all_parameters = parameters | constant_parameters
        infered_parameters = {}
        infered_parameters['n_classes'] = 100 if (parameters['dataset_name'] == 'Purchase100') else 10

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

        infered_parameters['n_honest_workers'] = all_parameters['n_workers'] - all_parameters['n_byzantine_workers']
        
        if all_parameters['dataset_name'] == 'MNIST' or all_parameters['dataset_name'] == 'Fashion_MNIST' :
            infered_parameters['n_step'] =  801 
            infered_parameters['lr'] = lr_MNIST
        else:
            infered_parameters['n_step'] =  2001 
            infered_parameters['lr'] = lr_CIFAR10_Purchase100

        infered_parameters["training_function"] = stochastic_heavy_ball if all_parameters["training_name"] == "stochastic_heavy_ball" else gradient_approx_training

        if torch.cuda.is_available(): 
            n_gpu = gpu_selection % torch.cuda.device_count()
            device = torch.device(f"cuda:{n_gpu}")
            infered_parameters['device'] = device
            
            gpu_selection += 1
        else : 
            infered_parameters['device'] = 'cpu'

        kwargs = all_parameters | infered_parameters

        experiments.append(kwargs)
    

    n_parallel = 10
    mini_batch_of_combinations = split_list(experiments[10], n_parallel)

    torch.multiprocessing.set_start_method('spawn')

    for combination_batch in mini_batch_of_combinations:
        pool = Pool()
        pool.map(run, combination_batch)
        pool.close()
        pool.join()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__" :     
    one_exp() 