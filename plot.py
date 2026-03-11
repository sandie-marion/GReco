from os.path import join, exists 
import matplotlib.pyplot as plt 
from os import mkdir 
import itertools
import torch 

from time import sleep 




def get_file_name(kwargs):
    experiment_id = f"{kwargs["attack_name"]}_" 
    experiment_id += f"{kwargs["aggregator_name"]}_" 
    experiment_id += f"{kwargs["pre_aggregator_name"]}_" 
    if "hullguard_attack_param" in kwargs.keys() :
        experiment_id += f"{kwargs["hullguard_attack_param"]}_" 
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
        feature = data[feature_name]
        percent = [i*100 for i in feature]
        return percent
    except : 
        print(f"file {path} not found.")
        return []


def plot (files, var_values, feature, save_path) :
    n_colors = len(var_values)
    colors = ['#87CEEB', '#B47ECF',  '#ECD266', '#A8D8B9', '#F4A460'] 
    data_list = []
    for path in files : 
        data = load_data(path, feature) 
        if len(data) > 0 : 
            data_list.append(data) 
    
    plt.figure(figsize=(6, 4))

    for idx, (data, legend, color) in enumerate(zip(data_list, var_values, colors[:n_colors])) : 
        plt.plot(data, color=color, label=legend) 
        plt.legend() 
        plt.xlabel("Evaluation Step") 
        plt.ylabel(feature)
        plt.grid(True)
    
    plt.savefig(save_path)
    plt.close() 


def run () : 
    experiment_folder = 'test_seed1'

    constant_parameters = {
                    'n_workers': 60,
                    'batch_size': 32,
                    'beta': 0.9,
                    'criterion_name': 'CrossEntropy', 
                }


    variable_parameters = {
                        "aggregator_name": ["CWMed","CwTM","RFA", "GAS", "HullGuard"],
                        "attack_name": ["ALIE","FOE","Mimic","MinSum","LF"],
                        "pre_aggregator_name": ["BKT", "None"],
                        "n_byzantine_workers" : [8, 14, 20, 26],
                        "alpha" : [0.1, 1, 10], 
                        'dataset_name': ["CIFAR10", "MNIST", "Fashion_MNIST"]
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

                if combi["aggregator_name"] == "HullGuard" :  
                        combi["hullguard_attack_param"] = "dirac" 
                        combi["pre_aggregator_name"] = "None"
                        file_name = get_file_name(combi) 
                        files.append(join(experiment_folder, file_name))
                else : 
                    file_name = get_file_name(combi) 
                    files.append(join(experiment_folder, file_name))
            plot(files, var_values, feature, join(folder, save_name)) 
    
            

if __name__ == "__main__" : 
    run() 