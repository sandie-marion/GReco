import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utility import Statistics, save 

from gradients import model_parameters_format, gradient_dissimilarity
from time import time 

from collections import Counter 

# Descent algorithm
################################################################################################
def stochastic_heavy_ball(model, workers, aggregator, attack, test_loader, kwargs):
    """
    The stochastic heavy ball algorithm is described in detail in Fixing by Mixing: A Recipe for Optimal Byzantine ML under Heterogeneity.
    """
 
    n_honest_workers = kwargs['n_honest_workers']
    f = kwargs['n_byzantine_workers']
    beta = kwargs['beta']
    device = kwargs['device']
    experiment_id = kwargs['experiment_id']
    n_classes = kwargs['n_classes']
    n_step = kwargs['n_step']
    lr = kwargs['lr']
    reg_param = kwargs['reg_param']
    clip_param = kwargs['clip_param']
    experiment_folder = kwargs['experiment_folder']

    n_workers = n_honest_workers + f 

    statistics_to_save = Statistics()
    step = 0

    worker_iters = [iter(loader) for loader in workers.loaders()]

    for step in range (0, n_step) :
        start = time() 
        
        for worker_id in range (0, n_workers) : 

            model.train() 
            running_loss = 0.0

            try : 
                batch = next(worker_iters[worker_id])
            except StopIteration :
                worker_iters[worker_id] = iter(workers.loaders()[worker_id])
                batch = next(worker_iters[worker_id])
            
            inputs, labels = batch 
            minibatch_distrib = get_distribution(labels, n_classes)
                    
            # if an honest worker
            if workers[worker_id].honest:
                                
                model.zero_grad()
                
                inputs, labels = inputs.to(device), labels.to(device)
         
                outputs = model(inputs)

                reg = regularization(model, reg_param)
                
                loss = workers[worker_id].compute_loss(outputs, labels) + reg
                
                loss.backward()

                clip_grad_norm(model, clip_param)
                
                running_loss += loss.item()/n_honest_workers
        
                workers[worker_id].compute_momentum(model, beta)
            
            # if a Byzantine worker
            else:
                worker = workers[worker_id]
                row_honest_gradients = workers.get_momentums(only_honest = True, row = True)
                row_bad_gradient = attack(step, model, worker, inputs, labels, row_honest_gradients)
                bad_gradient = model_parameters_format(row_bad_gradient, model)
                workers[worker_id].momentum = bad_gradient
                
        # Update model
        with torch.no_grad():
            row_momentums = workers.get_momentums(only_honest = False, row = True)

            row_aggregated_momentum = aggregator(row_momentums)
            unrow_aggregated_momentum = model_parameters_format(row_aggregated_momentum, model)
            for param_idx, param in enumerate(model.parameters()):
                param -= lr(step) * unrow_aggregated_momentum[param_idx]
            
            end = time() 

            if step % 5 == 0:
                # Compute remaining statistics to save
                accuracy = evaluate_model(model, test_loader, device)

                row_honest_momentums = workers.get_momentums(only_honest = True, row = True)
                grad_dissimilarity = gradient_dissimilarity(row_honest_momentums)

                t_epoch = end - start 

                # Stock statistics
                statistics_to_save.append('Steps', step)
                statistics_to_save.append('RunningLoss', running_loss)
                statistics_to_save.append('GradientDissimilarity_Momentums', grad_dissimilarity)
                statistics_to_save.append('Accuracy', accuracy)
                statistics_to_save.append('Time', t_epoch)

                clear_output(wait=True)
                print(f"Experiment {kwargs['experiment_id']} Progress {step}/{kwargs['n_step']}")
                #plt.plot(statistics_to_save['Accuracy'])
                #plt.title(int(np.mean(statistics_to_save['Accuracy'])))
                #plt.show()

            running_loss = 0.0 


    # Save statistics
    save(data = statistics_to_save.data, name = 'statistics', experiment_id = kwargs['experiment_id'], experiment_folder = kwargs['experiment_folder'])
    del statistics_to_save
    del inputs, labels, outputs
    
# About Training
################################################################################################
def regularization(model: Module, l2_lambda: float) -> torch.Tensor:
    """
    Compute L2 regularization term for all model parameters.
    """
    l2_reg = sum(param.norm(2) ** 2 for param in model.parameters())
    return l2_lambda * l2_reg


def clip_grad_norm(model: Module, max_norm: float) -> None:
    """
    Clips gradients to prevent exploding gradients during training.
    """
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def lr_CIFAR10_Purchase100(step):
    """
    Learning rate for CIFAR10 ans Purchase100.
    """
    if step < 1500:
        return 0.25
    return 0.025


def lr_MNIST(step):
    """
    Learning rate for MNIST and FashionMNIST dataset.
    """
    lr = 0.75
    lr /= 1+int(step/50)
    return lr

# Evaluation
################################################################################################
def evaluate_model(model: Module, test_loader, device: torch.device) -> float:
    """Return model accuracy (%) on test data."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def get_distribution (labels : list, n_labels : int) -> torch.Tensor : 
    nb_samples = len(labels) 
    label_list = [int(l) for l in labels]
    sample_per_class = Counter(label_list)
    distribution = [0 for i in range(0,n_labels)]
    for label in sample_per_class :
        distribution[label] = sample_per_class[label] / nb_samples

    return torch.tensor(distribution)  
