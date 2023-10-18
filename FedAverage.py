from modelUtil import *
from datasets import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import LDPServer, CDPServer
from datetime import date
import argparse
import time

start_time = time.time()

def parse_arguments():
    '''
    data:数据集，默认mnist
    nclient:客户端的数量，默认100
    nclass:数据集标签类别数量，默认10
    ncpc:分给每个客户端类别的数量，默认2
    model:选用的模型，默认mnist_fully_connected_IN
    mode:DP的模式，分为CPD和LDP，默认LDP
    round:训练的轮次，默认150
    eps:隐私预算的大小，默认8
    physical_bs:这个有点奇怪，不是很理解，看解释是为了解决泊松分布造成的采样偏差问腿，实际跑起来设置到64就很容易显存爆满
    sr:每轮差分隐私机制的采样率，默认1.0
    lr:学习率，默认0.1
    flr:学习率，默认0.1
    E：文件前缀
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','fashionmnist','emnist','purchase','chmnist'])
    parser.add_argument('--nclient', type=int, default= 100)
    parser.add_argument('--nclass', type=int, help= 'the number of class for this dataset', default= 10)
    parser.add_argument('--ncpc', type=int, help= 'the number of class assigned to each client', default=2)
    parser.add_argument('--model', type=str, default='mnist_fully_connected_IN', choices = ['mnist_fully_connected_IN', 'resnet18_IN', 'alexnet_IN', 'purchase_fully_connected_IN', 'mnist_fully_connected', 'resnet18', 'alexnet', 'purchase_fully_connected'])
    parser.add_argument('--mode', type=str, default= 'LDP')
    parser.add_argument('--round',  type = int, default= 150)
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--physical_bs', type = int, default=3, help= 'the max_physical_batch_size of Opacus LDP, decrease if cuda out of memory')
    parser.add_argument('--sr',  type=float, default=1.0,
                        help='sample rate in each round')
    parser.add_argument('--lr',  type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--flr',  type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--E',  type=int, default=1,
                        help='the index of experiment in AE')
    args = parser.parse_args()
    return args

args = parse_arguments()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  #判断GPU是否可用
today = date.today().isoformat()   #返回一个表示当前本地日期的date对象，并将格式改为'YYYY-MM-DD'的字符串
DATA_NAME = args.data   #数据集名称
NUM_CLIENTS = args.nclient  #客户端的数量
NUM_CLASSES = args.nclass  #数据集标签的数量
NUM_CLASES_PER_CLIENT= args.ncpc   #分给每个客户端标签的数量
MODEL = args.model   #训练使用的模型
MODE = args.mode   #判断是CDP还是LDP
EPOCHS = 1  #本地训练轮次
ROUNDS = args.round   #全局训练轮次
BATCH_SIZE = 64  #本地每次训练的批量大小
LEARNING_RATE_DIS = args.lr  #学习率
LEARNING_RATE_F = args.flr  #不知道干嘛的
mp_bs = args.physical_bs  #不知道干嘛的
target_epsilon = args.epsilon  #隐私预算
target_delta = 1e-3  #差分隐私高斯机制的delta
sample_rate=args.sr  #采样率

os.makedirs(f'log/E{args.E}', exist_ok=True)   #判断EX文件存不存在，不存在就创建
user_param = {'disc_lr': LEARNING_RATE_DIS, 'epochs': EPOCHS}   #参数字典，包含了学习率以及本地训练轮次
server_param = {}
if MODE == "LDP":
    user_obj = LDPUser
    server_obj = LDPServer
    user_param['rounds'] = ROUNDS
    user_param['target_epsilon'] = target_epsilon
    user_param['target_delta'] = target_delta
    user_param['sr'] = sample_rate
    user_param['mp_bs'] = mp_bs
elif MODE == "CDP":
    user_obj = CDPUser
    server_obj = CDPServer
    user_param['flr'] = LEARNING_RATE_F
    server_param['noise_multiplier'] = opacus.accountants.utils.get_noise_multiplier(target_epsilon=target_epsilon,
                                                                                 target_delta=target_delta, 
                                                                                 sample_rate=sample_rate, steps=ROUNDS)
    print(f"noise_multipier: {server_param['noise_multiplier']}")
    server_param['sample_clients'] = sample_rate*NUM_CLIENTS
else:
    raise ValueError("Choose mode from [CDP, LDP]")

if DATA_NAME == 'purchase':
    root = 'data/purchase/dataset_purchase'
elif DATA_NAME == 'chmnist':
    root = 'data/CHMNIST'
else: root = '~/torch_data'

#数据集名字，目录，客户端数量，批次大小，每个客户端分配的标签数量，标签总数量
train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, root, NUM_CLIENTS,
                                                         BATCH_SIZE, NUM_CLASES_PER_CLIENT, NUM_CLASSES)

print(user_param)
users = [user_obj(i, device, MODEL, None, NUM_CLASSES, train_dataloaders[i], **user_param) for i in range(NUM_CLIENTS)]   #在函数调用中，**用于解包字典，将字典中的每个键值对作为关键字参数传递给函数。
server = server_obj(device, MODEL, None, NUM_CLASSES, **server_param)
for i in range(NUM_CLIENTS):
    users[i].set_model_state_dict(server.get_model_state_dict())
best_acc = 0
for round in range(ROUNDS):
    random_index = np.random.choice(NUM_CLIENTS, int(sample_rate*NUM_CLIENTS), replace=False)
    for index in random_index:users[index].train()
    if MODE == "LDP":
        weights_agg = agg_weights([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(weights_agg)
    else:
        server.agg_updates([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(server.get_model_state_dict())
    print(f"Round: {round+1}")
    acc = evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    if acc > best_acc:
        best_acc = acc
    if MODE == "LDP":
        eps = max([user.epsilon for user in users])
        print(f"Epsilon: {eps}")
        if eps > target_epsilon:
            break

end_time = time.time()
print("Use time: {:.2f}h".format((end_time - start_time)/3600.0))
print(f'Best accuracy: {best_acc}')
results_df = pd.DataFrame(columns=["data","num_client","ncpc","mode","model","epsilon","accuracy"])
results_df = results_df._append(
    {"data": DATA_NAME, "num_client": NUM_CLIENTS,
     "ncpc": NUM_CLASES_PER_CLIENT, "mode":MODE,
     "model": MODEL, "epsilon": target_epsilon, "accuracy": best_acc},
    ignore_index=True)
results_df.to_csv(f'log/E{args.E}/{DATA_NAME}_{NUM_CLIENTS}_{NUM_CLASES_PER_CLIENT}_{MODE}_{MODEL}_{target_epsilon}.csv', index=False)
