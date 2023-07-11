import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms,models,utils
from tqdm.notebook import tqdm
# from tqdm import tqdm_notebook as tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms,utils

train_path = 'D:/AIdata/dog vs cat/train'
test_path = 'D:/AIdata/dog vs cat/test1'
data_root = 'D:/AIdata/dog vs cat/'
csv_path = './submission_valnet.csv'
tensorboard_path='C:/Users/BraveY/Documents/BraveY/AI-with-code/dog-vs-cat/tensortboard'
model_save_path = 'C:/Users/BraveY/Documents/BraveY/AI-with-code/dog-vs-cat/modelDict/dogs-vs-cats-notebook.pth'

class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog' :
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # split 的是str类型要转换为int
        label = torch.as_tensor(label, dtype=torch.int64) # 必须使用long 类型数据，否则后面训练会报错 expect long
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
    def __len__(self) -> int:
        return len(self.path_list)

train_ds = MyDataset(train_path)
test_ds = MyDataset(test_path,train=False)
for i, item in enumerate(tqdm(train_ds)):
#     pass
    print(item)
    break

full_ds = train_ds
train_size = int(0.8 * len(full_ds))
validate_size = len(full_ds) - train_size
new_train_ds, validate_ds = torch.utils.data.random_split(full_ds,[train_size, validate_size])#数据集划分

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,
                                            shuffle=True, pin_memory=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                            shuffle=True, pin_memory=True, num_workers=0)
## numworkers设置不为0 会报错 Broken pipe Error 网上说是win10上的pytorch bug

new_train_loader = torch.utils.data.DataLoader(new_train_ds, batch_size=32,
                                            shuffle=True, pin_memory=True, num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=32,
                                            shuffle=True, pin_memory=True, num_workers=0)

for i, item in enumerate(train_loader):
#     pass
    print(item[0].shape)
    break

img_PIL_Tensor = train_ds[1][0]
new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
plt.imshow(new_img_PIL)
plt.show()
# print(new_img_PIL.show())

import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2, 2)  # 2*2的池化 池化后size 减半
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)  # 两个池化，所以是224/2/2=56
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    #         self.dp = nn.Dropout(p=0.5)
    def forward(self, x):
        #         print("input:", x)
        x = self.pool(F.relu(self.conv1(x)))
        #         print("first conv:", x)
        x = self.pool(F.relu(self.conv2(x)))
        #         print("second conv:", x)

        x = x.view(-1, 16 * 56 * 56)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

net = MyCNN()
# net = resnet50


criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()  #二分类交叉熵损失函数
# criterion = nn.BCEWithLogitsLoss() #二分类交叉熵损失函数 带log loss
# criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#也可以选择Adam优化方法
# optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


## topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensortboard/')

def train( epoch, train_loader, device, model, criterion, optimizer,tensorboard_path):
    model = model.to(device)
    for e in range(epoch):
        model.train()
    	top1 = AvgrageMeter()
        train_loss = 0.0
        train_loader = tqdm(train_loader)  #转换成tqdm类型 以方便增加日志的输出
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', e + 1, epoch, 'lr:', 0.001))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            # topk 准确率计算
            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()

    print('Finished Training')

def validate(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            #         inputs,labels = data[0],data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc

def submission(csv_path,test_loader, device, model):
    result_list = []
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            softmax_func = nn.Softmax(dim=1)  # dim=1表示行的和为1
            soft_output = softmax_func(outputs)
            predicted = soft_output[:, 1]
            for i in range(len(predicted)):
                result_list.append({
                    "id": labels[i].item(),
                    "label": predicted[i].item()
                })
    # 从list转成 dataframe 然后保存为csv文件
    columns = result_list[0].keys()
    result_dict = {col: [anno[col] for anno in result_list] for col in columns}
    result_df = pd.DataFrame(result_dict)
    result_df = result_df.sort_values("id")
    result_df.to_csv(csv_path, index=None)

net = MyCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()  #二分类交叉熵损失函数
# criterion = nn.BCEWithLogitsLoss() #二分类交叉熵损失函数 带log loss
# criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#也可以选择Adam优化方法
# optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)

# train( 1, train_loader, device,net, criterion, optimizer,tensorboard_path) # 完整的训练数据集
train( 1, new_train_loader, device,net, criterion, optimizer,tensorboard_path) # 划分80%后的训练数据集

torch.save(net.state_dict(), model_save_path)
val_net = MyCNN()
val_net.load_state_dict(torch.load('./dogs-vs-cats_12epoch_valnet.pth'))

validate(validate_loader,device,val_net,criterion)

submission('./test.csv',test_loader, device, val_net)




