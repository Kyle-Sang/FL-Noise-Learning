# -*- coding: utf-8 -*-

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
import torchvision.models as models
from tqdm import tqdm
from torchvision.models import ResNet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from argparse import ArgumentParser
from torchvision import transforms as tt
import csv
import glob, random

# from resnet import ResNet

# from torchsummary import summary

# set manual seed for reproducibility
# seed = 42

# # general reproducibility
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from PIL import Image
import numpy as np
import torchvision.transforms.functional as T

def load_image(infilename, resize=32) :
    img = T.resize(Image.open(infilename).convert('L'), size=resize)
    w, h = img.size
    img = img.crop((2, 2, w-2, h-2))
    img.load()
    data = np.asarray(img, dtype="uint8")
    # data = np.expand_dims(data, axis=2)
    
    return np.transpose(data)

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
    k = len(data)

    dist = np.random.beta(alpha, alpha)
    main_weight = .5 + (dist * .25)
    a = np.random.rand(k)
    a = (a/a.sum()*(1-low*k))
    weights = a+low
    weights = weights * (1-main_weight)
    mixed_x = ((x * main_weight) + np.sum([arr * weights[i] for i, arr in enumerate(data)], axis=0)).astype(np.uint8)
    # y_a, y_b = y, y[index]
    return mixed_x

def AddNoise(tensor, dataset, mean, std, gaussian_application, natural_image_application, simple=False, mix_num=3):
    choice = np.random.rand()
    if choice < gaussian_application:
        if not simple:
            tensor = mixup_data(tensor, dataset[np.random.choice(len(dataset), size=mix_num)])
        tensor = torch.from_numpy(tensor)
        return ((tensor + torch.tensor(np.random.laplace(mean, std, size=tensor.size()))).numpy()).astype(np.uint8)
    elif choice < (gaussian_application + natural_image_application):
        return natural_images[np.random.randint(60000)]
    else:
        return tensor


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


def non_iid_partition(dataset, n_nets, alpha, mixup_prop, natural_prop, real_prop, supplement = True):
    """
        :param dataset: dataset name
        :param n_nets: number of clients
        :param alpha: beta parameter of the Dirichlet distribution
        :return: dictionary containing the indexes for each client
    """
    y_train = np.array(dataset.targets)
    min_size = 0
    K = len(dataset.classes)
    N = y_train.shape[0]
    net_dataidx_map = {}
    num = []

    pbar = tqdm(desc='partitioning', total = 10)
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
            pbar.update(min_size)
            num.append(len(idx_batch))


    if supplement:
        for i, client in enumerate(idx_batch):
            classes, counts = np.unique(y_train[client], return_counts=True)
            # How many samples each class should have
            goal_count = np.max(counts)
            counts_dict = dict(zip(classes, counts))

            # Add supplemental data for each class so that a clients classes all have balanced data
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                add = goal_count
                if k in counts_dict:
                    add -= counts_dict[k]
                # Add indexes of untouched real images
                supplements = np.random.choice(idx_k, round(add*real_prop))
                idx_batch[i] += list(supplements)
                # Add indexes of mixup images
                supplements = np.random.choice(idx_k, round(add*mixup_prop)) + len(dataset.data)
                idx_batch[i] += list(supplements)
                # Add indexes of natural images
                supplements = np.random.choice(idx_k, round(add*natural_prop)) + (2 * len(dataset.data))
                idx_batch[i] += list(supplements)

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
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, sch_flag):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
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

def training(model, rounds, batch_size, lr, ds, data_dict, C, K, E, plt_title, plt_color, cifar_data_test,
             test_batch_size, criterion, num_classes, classes_test, sch_flag, filename):
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
        for k in tqdm(S_t):
            # Compute a local update
            local_update = ClientUpdate(dataset=ds, batchSize=batch_size, learning_rate=lr, epochs=E, idxs=data_dict[k],
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
        if filename is not None:
            with open(filename, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow([curr_round, loss_avg, t_loss, t_accuracy, best_accuracy])
        print(f"Current Round: {curr_round}, Average Loss: {loss_avg}, Test Loss: {t_loss}, Test Accuracy: {t_accuracy}, Best: {best_accuracy}")
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
    print(f"Best Accuracy: {best_accuracy}")
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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--norm', default="bn")
    parser.add_argument('--partition', default="noniid")
    parser.add_argument('--client_number', default=100)
    parser.add_argument('--alpha_partition', default=0.001)
    parser.add_argument('--commrounds', type=int, default=200)
    parser.add_argument('--clientfr', type=float, default=1.0)
    parser.add_argument('--numclient', type=int, default=10)
    parser.add_argument('--clientepochs', type=int, default=1)
    parser.add_argument('--clientbs', type=int, default=128)
    parser.add_argument('--clientlr', type=float, default=0.001)
    parser.add_argument('--sch_flag', default=False)
    parser.add_argument('--mixup_prop', type=float, default=0.0)
    parser.add_argument('--natural_img_prop', type=float, default=0.)
    parser.add_argument('--real_prop', type=float, default=0.)
    parser.add_argument('--mix_num', type=int, default=3)
    parser.add_argument('--laplace_scale', type=float, default=50.)
    parser.add_argument('--no_supplement', action="store_false")
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--natural_image_path', default="./stylegan-oriented/train")

    args = parser.parse_args()
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    cifar_data_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_cifar_train)
    cifar_data_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_cifar_test)
    cifar_data_train.data = cifar_data_train.data.numpy()
    # cifar_data_test.data = cifar_data_test.data.numpy()

    classes = np.array(list(cifar_data_train.class_to_idx.values()))
    classes_test = np.array(list(cifar_data_test.class_to_idx.values()))
    num_classes = len(classes_test)

    criterion = nn.CrossEntropyLoss()

    path = args.natural_image_path + "/**/*.jpg"
    images = glob.glob(path)
    print(len(images))
    natural_images = np.asarray(list(map(load_image, images[:len(cifar_data_train)])))
    print("done")
    print(len(natural_images))

    # Hyperparameters_List (H) = [rounds, client_fraction, number_of_clients, number_of_training_rounds_local, local_batch_size, lr_client]
    H = [args.commrounds, args.clientfr, args.numclient, args.clientepochs, args.clientbs, args.clientlr]

    if args.partition == 'noniid':
        # (dataset, clients, total_shards, shards_size, num_shards_per_client):
        # alpha for the Dirichlet distribution
        data_dict = non_iid_partition(cifar_data_train, args.numclient, float(args.alpha_partition),
                                       args.mixup_prop, args.natural_img_prop, args.real_prop, supplement=args.no_supplement)
    else:
        data_dict = iid_partition(cifar_data_train, 100)  # Uncomment for idd_partition

    if args.norm == 'gn':
        # cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
        #                           width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
        cifar_cnn = models.resnet18() #ResNet9(3,10)
    else:
        # cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
        #                           width_per_group=64, replace_stride_with_dilation=None)
        cifar_cnn = models.resnet18() # ResNet9(3,10)
    cifar_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    cifar_cnn.cuda()

    #print(cifar_cnn)
    #MNIST Compatibility changes
    
    #cifar_cnn.fc = nn.Linear(512, 10, bias=True)

    # Add noise portion of dataset
    # IMPORTANT: this part must go after the non_iid_partition is made
    mixup_copy = copy.deepcopy(cifar_data_train.data)

    # please ensure proportions add up to one
    mixup_dataset = AddNoise(mixup_copy, cifar_data_train.data, 0., args.laplace_scale, 1., 0., mix_num=args.mix_num)
    if len(natural_images) != 0:
        rng = np.random.default_rng()
        natural_dataset = rng.choice(natural_images, len(cifar_data_train.data))
    else:
        natural_dataset = copy.deepcopy(cifar_data_train.data)
        print("NO NATURAL IMAGES FOUND!")
    # natural_dataset = np.swapaxes(natural_dataset, 1, 3)
    # print(data_dict)
    # print(natural_dataset.shape)
    #print(type(cifar_data_train.targets))
    
    cifar_data_train.data = np.concatenate((cifar_data_train.data, mixup_dataset, natural_dataset))
    cifar_data_train.targets = np.concatenate((cifar_data_train.targets, cifar_data_train.targets, cifar_data_train.targets))

    # img = Image.fromarray(cifar_data_train.data[80000], 'L')
    # img.save('test_img_noise.jpg')

    plot_str = 'MNIST_' + args.partition + '_' + args.norm + '_' + 'comm_rounds_' + str(args.commrounds) + '_clientfr_' + str(
        args.clientfr) + '_numclients_' + str(args.numclient) + '_clientepochs_' + str(
        args.clientepochs) + '_clientbs_' + str(args.clientbs) + '_clientLR_' + str(args.clientlr)
    print(plot_str)
    plot_str2 = f"Supplement: {args.no_supplement}, Mixup: {args.mixup_prop} (Mixup_k:{args.mix_num}, Laplacian Scale: {args.laplace_scale}), Natural: {args.natural_img_prop}, Real: {args.real_prop}"
    print(plot_str2)

    cifar_data_train.data = torch.from_numpy(cifar_data_train.data)

    # Check client distributions
    # y_train = cifar_data_train.targets
    # a, t = np.unique(y_train[data_dict[2]], return_counts=True)
    # print(f"{a} \n{t}")
    filename=None
    if args.save:
        num = 1
        filename = f"{plot_str}||{plot_str2}||{num}"
        while os.path.isfile(filename):
            print('Name is taken...trying again...')
            num += 1
            filename = f"{plot_str}||{plot_str2}||{num}"
    
    
    trained_model = training(cifar_cnn, H[0], H[4], H[5], cifar_data_train, data_dict, H[1], H[2], H[3], plot_str, "green", cifar_data_test, 128, criterion, num_classes, classes_test, args.sch_flag, filename)