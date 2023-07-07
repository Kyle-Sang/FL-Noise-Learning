# -*- coding: utf-8 -*-
"""FadAvg

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZwRozzWme6BlfDc1HVgcFjSdTyjcHn68
"""

# Commented out IPython magic to ensure Python compatibility.
#   %load_ext tensorboard
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from argparse import ArgumentParser
from torchvision import transforms as tt
from PIL import Image
import numpy as np
import torchvision.transforms.functional as T
import glob, random

# from torchsummary import summary

# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_cifar10():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
  data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))
  
def load_image(infilename, resize=32) :
    img = T.resize(Image.open(infilename), size=resize)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return np.transpose(data)
  
def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  client_tmp=[]
  sum_=0
  #### creating random values for each client ####
  for i in range(nclients-1):
    tmp=random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)
  sum_+=random.randint(10,100)
  client_tmp= np.array(client_tmp)
  #### using those random values as weights ####
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()
  to_ret = list(clients_dist)
  to_ret.append(num)
  return to_ret

def mixup_data(x, data, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0.:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1.
    # batch_size = x.size()[0]
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)
    low = 0.1
    k = 3

    dist = np.random.beta(alpha, alpha)
    main_weight = .5 + (dist * .25)
    a = np.random.rand(k)
    a = (a/a.sum()*(1-low*k))
    weights = a+low
    weights = weights * (1-main_weight)
    mixed_x = ((x * main_weight) + np.sum([arr * weights[i] for i, arr in enumerate(np.random.choice(data, k))], axis=0)).astype(np.uint8)
    # y_a, y_b = y, y[index]
    return mixed_x

def AddNoise(tensor, mean, std, gaussian_application, natural_image_application):
    choice = np.random.rand()
    if choice < gaussian_application:
        tensor = torch.from_numpy(tensor)
        return ((tensor + torch.randn(tensor.size()) * std + mean).numpy()).astype(np.uint8)
    elif choice < (gaussian_application + natural_image_application):
        #add try except
        return natural_images[np.random.randint(1000)]
    else:
        return tensor
    
def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True, proportion=1, supplement=False, params=[0., 0., 0., 0.]):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''
  print(supplement)
  #### constants ####
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1
  # gauss = lambda x, y, z : x + torch.randn(x.size()) * z + y
  # params = [mean, std, mixup proportion, natural image proportion]
  noise_data = AddNoise(data, params[0], params[1], params[2], params[3])

  ### client distribution ####
  data_per_client = clients_rand(int(len(data) / proportion), n_clients)
  data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)

  noise_idcs = copy.deepcopy(data_idcs)
  # split data among clients
  clients_split = []
  c = 0
  # We want all clients to have this much data for each class
  max_data = np.max(data_per_client_per_class)
  print(f"Data Per Client: {data_per_client_per_class}, and goal amount of data each client has per class: {max_data}")
  for i in range(n_clients):
    client_idcs = []
    share_idcs = []

    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    cl = []
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]

      budget -= take

      cl.append(c)
      if take < max_data and supplement:
        fill = min((max_data-take), len(noise_idcs[c]))
        share_idcs += random.choices(noise_idcs[c], k=fill)
      c = (c + 1) % n_labels

    # add noise examples
    if supplement:
        class_list = [x for x in range (0, n_labels)]
        noise_classes = [x for x in class_list if x not in cl]
        for class_ in noise_classes:
            take = min(max_data, len(noise_idcs[class_]))
            share_idcs += random.choices(noise_idcs[class_], k=take)

    print(f"Untouched: {len(client_idcs)} Noise: {len(share_idcs)}, Classes: {cl}")
    print(f"Labels: {np.asarray(np.unique(labels[(client_idcs + share_idcs)], return_counts=True)).T}")
    if supplement:
        print('here!')
        clients_split += [(np.concatenate((data[client_idcs], noise_data[share_idcs])), labels[(client_idcs + share_idcs)])]
    else:
       clients_split += [(data[client_idcs], labels[client_idcs])]

  def print_split(clients_split):
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()

    if verbose:
      print_split(clients_split)

  clients_split = np.array(clients_split)

  return clients_split

def shuffle_list(data):
  '''
  This function returns the shuffled data
  '''
  for i in range(len(data)):
    tmp_len= len(data[i][0])
    index = [i for i in range(tmp_len)]
    random.shuffle(index)
    data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
  return data

def shuffle_list_data(x, y):
  '''
  This function is a helper function, shuffles an
  array while maintaining the mapping between x and y
  '''
  inds = list(range(len(x)))
  random.shuffle(inds)


class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms 

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]
          

def get_default_data_transforms(train=True, verbose=True):
  transforms_train = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  }
  transforms_eval = {    
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar10'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar10'], transforms_eval['cifar10'])


def get_data_loaders(nclients,batch_size,classes_pc=10, real_wd =False ,verbose=True, proportion=1, params=[0., 0., 0., 0.], supplement=False):
  
  x_train, y_train, x_test, y_test = get_cifar10()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  
  if real_wd:
     print('help')
  else:  
    split = split_image_data(x_train, y_train, n_clients=nclients, classes_per_client=classes_pc, verbose=verbose, proportion=proportion, params=params, supplement=supplement)
  
  split_tmp = split #shuffle_list(split)
  
  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batch_size, shuffle=True, drop_last=True) for x, y in split_tmp]
  
  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False)

  return client_loaders, test_loader

"""## Partitioning the Data (IID and non-IID)"""


def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


def non_iid_partition(dataset, n_nets, alpha):
    """
        :param dataset: dataset name
        :param n_nets: number of clients
        :param alpha: beta parameter of the Dirichlet distribution
        :return: dictionary containing the indexes for each client
    """
    y_train = np.array(dataset.targets)
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])

    # net_dataidx_map is a dictionary of length #of clients: {key: int, value: [list of indexes mapping the data among the workers}
    # traindata_cls_counts is a dictionary of length #of clients, basically assesses how the different labels are distributed among
    # the client, counting the total number of examples per class in each client.
    return net_dataidx_map


"""## Federated Averaging

### Local Training (Client Update)

Local training for the model on client side
"""


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, sch_flag):
        self.train_loader = dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sch_flag = sch_flag

    def train(self, model):

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # if self.sch_flag == True:
        #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue;

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                # if self.sch_flag == True:
                #  scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            # self.learning_rate = optimizer.param_groups[0]['lr']

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss


"""### Server Side Training

Following Algorithm 1 from the paper
"""


def training(model, rounds, batch_size, lr, ds, C, K, E, plt_title, plt_color, cifar_data_test,
             test_batch_size, criterion, num_classes, classes_test, sch_flag):
    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
      - model:           PyTorch model to train
      - rounds:          Number of communication rounds for the client update
      - batch_size:      Batch size for client update training
      - lr:              Learning rate used for client update training
      - ds:              Dataset used for training
      - data_dict:       Type of data partition used for training (IID or non-IID)
      - C:               Fraction of clients randomly chosen to perform computation on each round
      - K:               Total number of clients
      - E:               Number of training passes each client makes over its local dataset per round
      - tb_writer_name:  Directory name to save the tensorboard logs
    Returns:
      - model:           Trained model on the server
    """

    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_accuracy = 0
    # measure time
    start = time.time()

    for curr_round in range(1, rounds + 1):
        w, local_loss = [], []
        # Retrieve the number of clients participating in the current training
        m = max(int(C * K), 1)
        # Sample a subset of K clients according with the value defined before
        S_t = np.random.choice(range(K), m, replace=False)
        # For the selected clients start a local training
        for k in S_t:
            # Compute a local update
            local_update = ClientUpdate(dataset=ds[k], batchSize=batch_size, learning_rate=lr, epochs=E,
                                        sch_flag=sch_flag)
            # Update means retrieve the values of the network weights
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
        # lr = 0.999*lr
        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        if curr_round == 200:
            lr = lr / 2
            E = E - 1

        if curr_round == 300:
            lr = lr / 2
            E = E - 2

        if curr_round == 400:
            lr = lr / 5
            E = E - 3

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        # print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)), lr)
        train_loss.append(loss_avg)

        t_accuracy, t_loss = testing(model, cifar_data_test, test_batch_size, criterion, num_classes, classes_test)
        test_accuracy.append(t_accuracy)
        test_loss.append(t_loss)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy
        # torch.save(model.state_dict(), plt_title)
        print(curr_round, loss_avg, t_loss, test_accuracy[0], best_accuracy)
        # print('best_accuracy:', best_accuracy, '---Round:', curr_round, '---lr', lr, '----localEpocs--', E)

    end = time.time()
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()
    x_axis = np.arange(1, rounds + 1)
    y_axis1 = np.array(train_loss)
    y_axis2 = np.array(test_accuracy)
    y_axis3 = np.array(test_loss)

    ax.plot(x_axis, y_axis1, 'tab:' + 'green', label='train_loss')
    ax.plot(x_axis, y_axis2, 'tab:' + 'blue', label='test_accuracy')
    ax.plot(x_axis, y_axis3, 'tab:' + 'red', label='test_loss')
    ax.legend(loc='upper left')
    ax.set(xlabel='Number of Rounds', ylabel='Train Loss',
           title=plt_title)
    ax.grid()
    # fig.savefig(plt_title+'.jpg', format='jpg')
    print("Training Done!")
    print("Total time taken to Train: {}".format(end - start))

    return model


"""## ResNet50 Model (W & W/O GN)

> Indented block


"""


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


"""## Testing Loop"""


def testing(model, dataset, bs, criterion, num_classes, classes):
    # test loss
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())

        # test accuracy for each object class
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    return 100. * np.sum(correct_class) / np.sum(total_class), test_loss

def loaders(nclients,batch_size,classes_pc=10, real_wd =False ,verbose=True, proportion=1, params=[0., 0., 0., 0.], supplement=False, count=3):
    try:
        train_loader, test_loader = get_data_loaders(nclients, batch_size, classes_pc=classes_pc, real_wd=real_wd, verbose=verbose, proportion=proportion, supplement=supplement, params=params)
    except:
        if count > 0:
            train_loader, test_loader = loaders(nclients,batch_size,classes_pc=10, real_wd =False ,verbose=True, proportion=proportion, supplement=supplement, params=params, count=count-1)
        else:
           print('FAILED TO CREATE DATALOADERS')
           return None, None
    finally:
       return train_loader, test_loader

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--norm', default="bn")
    parser.add_argument('--partition', default="noniid")
    parser.add_argument('--client_number', default=10)
    parser.add_argument('--alpha_partition', default=0.5)
    parser.add_argument('--commrounds', type=int, default=100)
    parser.add_argument('--clientfr', type=float, default=0.1)
    parser.add_argument('--numclient', type=int, default=10)
    parser.add_argument('--clientepochs', type=int, default=20)
    parser.add_argument('--clientbs', type=int, default=64)
    parser.add_argument('--clientlr', type=float, default=0.001)
    parser.add_argument('--sch_flag', default=False)
    parser.add_argument('--proportion', type=int, default=1)
    parser.add_argument('--classespc', type=int, default=2)

    args = parser.parse_args()

    # images = glob.glob("./noisy_data/small_scale/dead_leaves-mixed/train/0000000000/*.jpg")
    # natural_images = np.asarray(list(map(load_image, images)))
    supplement = True
    train_loader, test_loader = loaders(args.client_number, args.clientbs, classes_pc=args.classespc, real_wd=False, verbose=True,
                                        proportion=args.proportion, params=[0., 50., 0.5, 0.], supplement=supplement)
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = tt.Compose([tt.ToTensor(),
                                         tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                         tt.RandomHorizontalFlip(p=0.5),
                                         tt.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize(*stats)])

    cifar_data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
    cifar_data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar_test)

    classes = np.array(list(cifar_data_train.class_to_idx.values()))
    classes_test = np.array(list(cifar_data_test.class_to_idx.values()))
    num_classes = len(classes_test)

    criterion = nn.CrossEntropyLoss()

    # Hyperparameters_List (H) = [rounds, client_fraction, number_of_clients, number_of_training_rounds_local, local_batch_size, lr_client]
    H = [args.commrounds, args.clientfr, args.numclient, args.clientepochs, args.clientbs, args.clientlr]

    # if args.norm == 'gn':
    #     cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
    #                                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
    # else:
    cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                    width_per_group=64, replace_stride_with_dilation=None)

    cifar_cnn.cuda()

    plot_str = args.partition + '_' + args.norm + '_' + 'comm_rounds_' + str(args.commrounds) + '_clientfr_' + str(
        args.clientfr) + '_numclients_' + str(args.numclient) + '_clientepochs_' + str(
        args.clientepochs) + '_clientbs_' + str(args.clientbs) + '_clientLR_' + str(args.clientlr) + '_proportion_' + str(
        args.proportion) + '_classespc_' + str(args.classespc) + '_supplement_' + str(supplement)
    print(plot_str)

    trained_model = training(cifar_cnn, H[0], H[4], H[5], train_loader, H[1], H[2], H[3], plot_str,
                                "green", cifar_data_test, 128, criterion, num_classes, classes_test, args.sch_flag)

# add time, and choose to supplement option, and supplement parameters