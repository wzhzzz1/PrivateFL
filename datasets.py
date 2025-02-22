import random
from collections import defaultdict
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CIFAR100, EMNIST
from dataloader import *
np.random.seed(2022)

def get_datasets(data_name, dataroot, preprocess = None):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='cifar10':
        normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#这是归一化操作。它将输入图像的每个通道的像素值从范围 [0, 1] 缩放到范围 [-1, 1]，使图像的均值为0，标准差为1。这通常是为了更好地适应神经网络的训练，以提高训练的稳定性。
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(120), normalization]) if preprocess==None else preprocess#组合变换，这个就很基本了，转化张量，重定义尺寸，标准化

        data_obj = CIFAR10
    elif data_name =='cifar100':
        normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), normalization]) if preprocess==None else preprocess

        data_obj = CIFAR100
    elif data_name == 'mnist':
        normalization = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([transforms.ToTensor(), normalization])
        data_obj = MNIST
    elif data_name == 'fashionmnist':
        normalization = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([transforms.ToTensor(),  normalization])
        data_obj = FashionMNIST
    elif data_name == 'emnist':
        normalization = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([transforms.ToTensor(),  normalization])
        data_obj = EMNIST
    elif data_name == 'purchase':
        transform = transforms.Compose([transforms.ToTensor()])
        data_obj = Purchase
    elif data_name == 'chmnist':
        normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150,150)),normalization])
        data_obj = CHMNIST
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100', 'fashionmnist', 'emnist, 'purchase', 'chmnist']")


    if data_name == 'emnist':
        train_set = data_obj(
            dataroot,
            train=True,
            transform=transform,
            split='digits',
            download=True
        )

        test_set = data_obj(
            dataroot,
            train=False,
            split='digits',
            transform=transform
        )

    else:
        train_set = data_obj(
            dataroot,
            train=True,
            transform=transform,
            download=True
        )

        test_set = data_obj(
            dataroot,
            train=False,
            transform=transform
        )

    return train_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    #classes 包含了数据集中的唯一类别，其中每个元素代表一个不同的类别。
    #num_samples 包含了每个唯一类别出现的次数，即每个类别的样本数量。
    classes, num_samples = np.unique(data_labels_list, return_counts=True)  #return_counts=True 参数被设置为 True，这表示要返回唯一值的同时，还要返回它们在原始数组中出现的次数。
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)  #_是占位符，表示不打算用第三个返回值，这样又不会报错

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # print(num_classes)
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed" #它检查每个客户端获得的类别数量是否能够整除总的类别数量。如果条件不满足，即无法等分类别给每个客户端，就会引发 AssertionError
    count_per_class = (classes_per_user * num_users) // num_classes #计算每个类别应该被分配给多少个客户端
    class_dict = {}
    for i in range(num_classes):
        probs=np.array([1]*count_per_class)  #这个数组 probs 表示了当前类别被分配给不同客户端的概率。因为每个客户端应该平均分配类别，所以每个类别的概率都初始化为1
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]   #其中包含了每个类别的剩余分配数量
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]  #max_class_counts 是一个包含整数值的一维 NumPy 数组，每个整数值表示一个类别的索引，这些类别在剩余分配数量中具有最大值。这些索引可以用于后续的操作，比如随机选择一个类别分配给当前客户端，以确保数据的均匀分布。
            max_class_counts = np.setdiff1d(max_class_counts, np.array(c)) #排除已经分配给当前客户端的类别，以确保不会重复分配。np.setdiff1d 函数用于找到两个数组的差集，从 max_class_counts 中去掉已经分配的类别。
            c.append(np.random.choice(max_class_counts))  #从剩余最多的类的索引随机选一个
            class_dict[c[-1]]['count'] -= 1  #将被分配的类别的剩余分配数量减去1，以反映已经分配给一个客户端
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}
    #这行代码创建了一个字典 data_class_idx，其中包含了每个类别的数据索引。
    '''
    data_class_idx = {
        0: array([1, 3, 7, ...]),    # 类别0的样本索引
        1: array([2, 5, 10, ...]),   # 类别1的样本索引
        ...                       # 其他类别的样本索引
    }
    '''
    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)


    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)  #计算应该分配给当前客户端的类别 c 的样本数量，根据分配概率和总样本数量来确定。
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])  #将类别 c 的前 end_idx 个样本索引添加到当前客户端的 user_data_idx 列表中。
            data_class_idx[c] = data_class_idx[c][end_idx:]   #从 data_class_idx 中移除已经分配的样本索引，以确保不会重复分配。
        if len(user_data_idx[usr_i])%2 == 1: user_data_idx[usr_i] = user_data_idx[usr_i][:-1]

    return user_data_idx


def gen_classes_id(num_users=10, num_classes_per_user=2, classes=10):
    class_partitions = defaultdict(list)
    class_counts = [list(range(classes)) for _ in range(num_classes_per_user)]
    user_data_classes = []
    for user in range(num_users):
        classes_user = np.random.choice(class_counts[0], size=1)
        class_counts[0].remove(classes_user[0])
        tmp = class_counts[1].copy()
        if classes_user[0] in tmp:tmp.remove(classes_user[0])
        if tmp is None:
            tmp=[user_data_classes[-1][0]]
            user_data_classes[-1][0] = classes_user[0]
        classes_user = np.append(classes_user, np.random.choice(tmp, size=1))
        class_counts[1].remove(classes_user[1])
        user_data_classes.append(classes_user)
    for c in user_data_classes:
        class_partitions['class'].append(c)
        class_partitions['prob'].append([0.5, 0.5])
    return class_partitions


def gen_classes(num_users=10, num_classes_per_user=6, classes=10):
    class_partitions = defaultdict(list)
    class_counts = [list(range(classes)) for _ in range(num_classes_per_user)]
    user_data_classes = []
    for user in range(num_users):
        user_data_classes.append(np.array([*range(user, user+num_classes_per_user)])%10)
    for c in user_data_classes:
        class_partitions['class'].append(c)
        class_partitions['prob'].append([1/num_classes_per_user]*num_classes_per_user)
    return class_partitions


def gen_random_loaders(data_name, data_path, num_users, bz, num_classes_per_user, num_classes, preprocess=None):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}  #1：参数大小 2：是否随机重排 3：通常用于加速数据加载 4：设置为 0，表示在加载数据时不使用多个工作进程
    dataloaders = []
    datasets = get_datasets(data_name, data_path, preprocess=preprocess)  #datasets是包含训练集和测试集的元组
    cls_partitions = None
    distribution = np.zeros((num_users, num_classes))  #这里distribution的行是用户的数量，列的标签的数量
    for i, d in enumerate(datasets):  #i是索引，d是数据集对象
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, num_classes_per_user)
            '''
            class_partitions = {
                'class': [c_1, c_2, ..., c_num_users],  # 每个 c_i 是一个列表，包含被分配给第 i 个客户端的类别
                'prob': [p_1, p_2, ..., p_num_users]     # 每个 p_i 是一个列表，包含了第 i 个客户端类别的分配概率
            }
            '''
            print(cls_partitions)
            for index in range(num_users):
                distribution[index][cls_partitions['class'][index]] = cls_partitions['prob'][index]  #此时每行包括了两个不为0的元素，元素的位置就是分配给客户端的类别，值就是相应的概率

            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions) #返回值是每个客户端被分配的样本索引

        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))  #接下来，将每个客户端被分配的样本索引列表 usr_subset_idx 转化为 PyTorch 数据子集（Subset）对象。这将为每个客户端创建一个独立的数据子集，包含了他们分配到的样本。
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets))) #最后，将每个数据子集（Subset）使用 PyTorch 的 DataLoader 类转化为数据加载器。数据加载器用于按批次加载数据并进行训练

    return dataloaders