import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utility import Statistics, save 
from training import * 
from copy import deepcopy
import torch.nn.functional as F

from gradients import model_parameters_format, gradient_dissimilarity

# Descent algorithm
################################################################################################
def FedReDefense_training (model, workers, aggregator, attack, test_loader, kwargs):
    """
    The stochastic heavy ball algorithm is described in detail in Fixing by Mixing: A Recipe for Optimal Byzantine ML under Heterogeneity.
    """
 
    n_honest_workers = kwargs['n_honest_workers']
    f = kwargs['n_byzantine_workers']
    beta = kwargs['beta']
    device = kwargs['device']
    experiment_id = kwargs['experiment_id']
    n_step = kwargs['n_step']
    lr = kwargs['lr']
    reg_param = kwargs['reg_param']
    clip_param = kwargs['clip_param']
    experiment_folder = kwargs['experiment_folder']
    train_param = kwargs['training_parameters']

    channel = train_param['channel']
    h = train_param['h']
    w = train_param['w']
    n_labels = train_param['n_labels']
    n_iter = train_param['n_iter']
    n_steps = train_param['n_steps']
    batch_size = train_param['batch_size']
    img_per_class = train_param['image_per_class']


    statistics_to_save = Statistics()
    step = 0

    #initialize all LKD elements for each worker 
    LKD_list = [LKD(model, 5e-2, channel, h, w, n_labels, img_per_class) for n in range (n_honest_workers+f)]
    workers_to_consider = {}
    for i in range (0, n_honest_workers+f) : 
        workers_to_consider[i] = 1


    while(step < n_step):
        
        # go through worker batches
        for batch_idx, batches in enumerate(zip(*workers.loaders())):
            
            if step < n_step :

                model.train()
                
                running_loss = 0.0

                losses = [] 
                is_byz = []
                
                # for each batch of workers do
                for worker_id, (inputs, labels) in enumerate(batches):
                    if not workers_to_consider[worker_id] :
                        continue 
                    
                    # if an honest worker
                    if workers[worker_id].honest:
                        is_byz.append(0) 
                                        
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
                        is_byz.append(1) 
                        row_honest_gradients = workers.get_momentums(only_honest = True, row = True)
                        row_bad_gradient = attack(step, model, row_honest_gradients)
                        bad_gradient = model_parameters_format(row_bad_gradient, model)
                        workers[worker_id].momentum = bad_gradient
                    
                    LKD_list[worker_id].update_target_model(workers[worker_id].flatten_momentum(), lr(step))
                    loss = LKD_list[worker_id].local_knowledge_distillation(n_iter, n_steps, batch_size)
                    losses.append(loss) 

                #all workers went through their round 
                
                acc, recall, fpr, fnr, label_pred = threshold_detection(losses, is_byz)


                    #HERE calculate distilled local knowledge
                    #train model on distilled local knowledge 
                    #compare synthetic model and real model, l2 norm gives the supposed update ~= gradient 
                    #reconstruction error is the diff between synthetic gradient and real one 
                    #mark workers below threshold 
                    #do not use them ever again in aggregation 
                    #threshold_detection (losses, real_labels) 
                    
                        
                # Update model
                with torch.no_grad():
                    row_momentums = workers.get_momentums(only_honest = False, row = True)

                    selected_row_momentums = [] 
                    for i in range (0, n_honest_workers+f) : 
                        if workers_to_consider[i] : 
                            selected_row_momentums.append(row_momentums[i]) 

                    row_aggregated_momentum = aggregator(row_momentums)
                    
                    unrow_aggregated_momentum = model_parameters_format(row_aggregated_momentum, model)
                    
                    for param_idx, param in enumerate(model.parameters()):
                        param -= lr(step) * unrow_aggregated_momentum[param_idx]
                    
                    if step % 5 == 0:
                        # Compute remaining statistics to save
                        accuracy = evaluate_model(model, test_loader, device)

                        row_honest_momentums = workers.get_momentums(only_honest = True, row = True)
                        grad_dissimilarity = gradient_dissimilarity(row_honest_momentums)
        
                        # Stock statistics
                        statistics_to_save.append('Steps', step)
                        statistics_to_save.append('RunningLoss', running_loss)
                        statistics_to_save.append('GradientDissimilarity_Momentums', grad_dissimilarity)
                        statistics_to_save.append('Accuracy', accuracy)

                        clear_output(wait=True)
                        print(f"Experiment {kwargs['experiment_id']} Progress {step}/{kwargs['n_step']}")
                        #plt.plot(statistics_to_save['Accuracy'])
                        #plt.title(int(np.mean(statistics_to_save['Accuracy'])))
                        #plt.show()
    
                    # Update util variable 
                    step += 1
                    running_loss = 0.0


    # Save statistics
    save(data = statistics_to_save.data, name = 'statistics', experiment_id = kwargs['experiment_id'], experiment_folder = kwargs['experiment_folder'])
    del statistics_to_save
    del inputs, labels, outputs
    



class LKD () : 
    def __init__ (self, base_model, lr, channel, h, w, n_labels, img_per_class) :
        print("initializing LKD")
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 

        self.base_model = base_model 
        self.target_model = base_model 

        label_syn = torch.zeros(img_per_class*n_labels, n_labels)
        self.label_syn = label_syn.detach().to(self.device).requires_grad_(True)

        image_syn = torch.randn(size=(img_per_class*n_labels, channel, h, w), dtype=torch.float)
        self.image_syn = image_syn.detach().to(self.device).requires_grad_(True)

        syn_lr = torch.tensor(lr).to(self.device)
        self.syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)


    def initialize_optimizers (self) : 

        self.optimizer_img = torch.optim.SGD([self.image_syn], lr=1e-1, momentum=0.5)
        self.optimizer_label = torch.optim.SGD([self.label_syn], lr=5e-2, momentum=0.5)
        self.optimizer_lr = torch.optim.SGD([self.syn_lr], lr=5e-5, momentum=0.5)
    

    def update_target_model (self, momentum, lr) : 
        with torch.no_grad() : 
            for param_idx, param in enumerate(self.target_model.parameters()):
                param -= lr* momentum[param_idx]

    
    def local_knowledge_distillation (self, n_iter, n_steps, batch_size) :
        print("start of local knowledge distillation") 

        self.initialize_optimizers() 

        num_params = sum([np.prod(p.size()) for p in (self.base_model.parameters())])

        true_iter = -1
        for i in range (n_iter) :

            student_model = deepcopy(self.base_model)
            student_model.train() 
            
            start_trajectory = [self.base_model.state_dict().copy()[name].cpu().clone() for name in self.base_model.state_dict()]
            end_trajectory = [self.target_model.state_dict().copy()[name].cpu().clone() for name in self.target_model.state_dict()]


            #flatten models parameters 
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in end_trajectory], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in start_trajectory], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in start_trajectory], 0)

            syn_images = self.image_syn
            y_hat = self.label_syn
            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(n_steps):
                if len(indices_chunks) == 0:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, batch_size))

                these_indices = indices_chunks.pop()
                x = syn_images[these_indices]
                print(x.size()) 
                this_y = y_hat[these_indices]
                forward_params = student_params[-1]
                x = student_model(x)
                ce_loss = kd_loss(x, this_y)
                
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True, allow_unused=True)[0]
                
                student_params.append(student_params[-1] - self.syn_lr * grad)


            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += F.mse_loss(student_params[-1], target_params, reduction="sum") + 1e-9
            param_dist += F.mse_loss(starting_params, target_params, reduction="sum") + 1e-9

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)


            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            if grand_loss.detach().cpu() < 0.6: #early convergence ???
                true_iter = i + 1
                break

            self.optimizer_img.zero_grad()
            self.optimizer_label.zero_grad()
            self.optimizer_lr.zero_grad()

            grand_loss.backward()

            self.optimizer_img.step()
            self.optimizer_lr.step()
            self.optimizer_label.step() 

        if true_iter != -1:
            iters = true_iter

        return grand_loss.item()




def kd_loss(output, y):
    soft_label = F.softmax(y, dim=1)
    # soft_label = y
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(- soft_label * logsoftmax(output))




def threshold_detection(loss, real_label, threshold=0.6):
    loss = np.array(loss)
    # import pdb; pdb.set_trace()
    if np.isnan(loss).any() == True:
        label_pred =np.where(np.isnan(loss), 1, 0).squeeze()
    else:
        label_pred = loss > threshold
    # import pdb; pdb.set_trace()
    real_label = np.array(real_label)
    if np.mean(loss[label_pred == 0]) > np.mean(loss[label_pred == 1]):
        #1 is the label of malicious clients
        label_pred = 1 - label_pred
    nobyz = sum(real_label)
    acc = len(label_pred[label_pred == real_label])/loss.shape[0]
    recall = np.sum(label_pred[real_label==1]==1)/nobyz
    fpr = np.sum(label_pred[real_label==0]==1)/(loss.shape[0]-nobyz)
    fnr = np.sum(label_pred[real_label==1]==0)/nobyz
    return acc, recall, fpr, fnr, label_pred
