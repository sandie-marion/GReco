from os.path import join, exists 
import matplotlib.pyplot as plt 
from os import mkdir 
import itertools
import torch 

from time import sleep 




def get_file_name(kwargs):
    
    exclude = {
        'n_classes', 'attack_parameters', 'aggregator_parameters',
        'pre_aggregator_parameters', 'criterion_parameters',
        'n_step', 'reg_param', 'clip_param', 'lr', 'experiment_folder', 'n_experiments',
        'heterogeneous_distribution', 'seed', 'training_parameters'
    }
    experiment_id = f"{kwargs["attack_name"]}_" 
    experiment_id += f"{kwargs["aggregator_name"]}_" 
    experiment_id += f"{kwargs["pre_aggregator_name"]}_" 
    experiment_id += f"{kwargs["criterion_name"]}_" 
    experiment_id += f"{kwargs["dataset_name"]}_" 
    experiment_id += f"{kwargs["n_byzantine_workers"]}_" 
    experiment_id += f"{kwargs["alpha"]}_" 
    experiment_id += f"{kwargs["n_workers"]}_" 
    experiment_id += f"{kwargs["batch_size"]}_" 
    experiment_id += f"{kwargs["beta"]}" 
    experiment_id += '_statistics.pt'

    return experiment_id


def get_save_file_name(kwargs):
    
    exclude = {
        'n_classes', 'attack_parameters', 'aggregator_parameters',
        'pre_aggregator_parameters', 'criterion_parameters',
        'n_step', 'reg_param', 'clip_param', 'lr', 'experiment_folder', 'n_experiments',
        'heterogeneous_distribution', 'seed', 'training_parameters'
    }
    
    experiment_id = '_'.join(str(v) for k, v in kwargs.items() if k not in exclude)
    experiment_id += '.png'

    return experiment_id


def get_all_combinations (dictionary) : 
    combinations = []
    keys = list(dictionary)
    for values in itertools.product(*map(dictionary.get, keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations 



def load_data (path, feature_name) : 
    try : 
        data = torch.load(path) 
        return data[feature_name] 
    except : 
        print(f"file {path} not found.")
        return []


def plot (files, var_values, feature, save_path) : 
    n_colors = len(var_values)
    colors = ['#87CEEB', '#B47ECF',  '#ECD266', '#F4A460', '#A8D8B9'] 
    data_list = []
    for path in files : 
        data = load_data(path, feature) 
        if len(data) > 0 : 
            data_list.append(data) 
    
    plt.figure(figsize=(6, 4))

    for idx, (data, legend, color) in enumerate(zip(data_list, var_values, colors[:n_colors])) : 
        print(legend) 
        plt.plot(data, color=color, label=legend) 
        plt.legend() 
        plt.xlabel("Evaluation Step") 
        plt.ylabel(feature)
    
    plt.savefig(save_path)
    plt.close() 


experiment_folder = 'test_ARC_NNM'

constant_parameters = {
                'n_workers': 20,
                'batch_size': 32,
                'beta': 0.9,
                'attack_name': 'ALIE',
                'aggregator_name': 'CwTM',
                'alpha' : 0.3,
                'criterion_name': 'CrossEntropy',
            }


variable_parameters = {
                            'pre_aggregator_name': ['NNM','ARC'],
                            'dataset_name': ['CIFAR10','MNIST'],
                            'n_byzantine_workers' : [4, 8],
                        }


combinations = []

variable_keys = variable_parameters.keys() 

feature = "Accuracy"

for variable_name in variable_keys : 
    folder = join(experiment_folder, variable_name) 
    if not exists (folder) : 
        mkdir(folder) 
    var_values = variable_parameters[variable_name]
    var_param = {i : variable_parameters[i] for i in variable_keys if i != variable_name}
    combinations = get_all_combinations(var_param) 
    for combi in combinations : 
        combi = combi | constant_parameters 
        save_name = get_save_file_name(combi) 
        files = [] 
        for var_value in var_values : 
            combi = combi | {variable_name : var_value}
            file_name = get_file_name(combi) 
            files.append(join(experiment_folder, file_name))
            plot(files, var_values, feature, join(folder, save_name))
         

