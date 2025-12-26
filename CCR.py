import torch
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 1. 加载数据并做处理
# 1.1 从文件读取数据
train_data = pd.read_csv("../data/fashion-mnist_train.csv")
test_data = pd.read_csv("../data/fashion-mnist_test.csv")
# 1.2 分离特征和标签
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values
# 1.3 转换为二维图像的tensor形式
x_train = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
x_test = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# plt.imshow(x_train[12345, 0], cmap='gray')
# plt.show()
# print(y_train[12345])

# 1.4 构建数据集
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# 2. 搭建模型
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(in_features=400, out_features=120),
    # nn.Sigmoid(),
    nn.ReLU(),

    nn.Linear(in_features=120, out_features=84),
    # nn.Sigmoid(),
    nn.ReLU(),

    nn.Linear(in_features=84, out_features=10),
)

# 测试每层前向传播输出的形状
# x = torch.rand((10, 1, 28, 28))
# for layer in model:
#     x = layer(x)
#     print(f"Layer: {layer.__class__.__name__:<12}  output shape: {x.shape}")

# 3. 初始化相关操作
# 3.1 参数初始化
def init_weights(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
model.apply(init_weights)

# 3.2 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3.3 设置超参数
lr = 0.01
batch_size = 256
epochs = 200

# 3.4 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 3.5 损失函数和优化器
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 4. 模型训练
for epoch in range(epochs):
    # 4.1 训练过程
    model.train()
    train_loss = 0
    train_acc_num = 0
    # 按小批次一次遍历训练集
    for i, (X, target) in enumerate(train_loader):
        X, target = X.to(device), target.to(device)
        # 前向传播
        output = model(X)
        # 计算损失
        loss = loss_func(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 累加训练损失
        train_loss += loss.item() * X.shape[0]
        # 累加预测正确的数量
        y_pred = output.argmax(dim=1)
        train_acc_num += y_pred.eq(target).sum().item()
        # 进度条
        print(f"\rEpoch: {epoch + 1:0>3} [{'=' * int((i+1) / len(train_loader) * 50):<50}]", end="")

    # 本轮训练完毕，计算平均损失和准确率
    this_train_loss = train_loss / len(train_dataset)
    this_train_acc = train_acc_num / len(train_dataset)

    # 4.2 验证
    model.eval()
    val_loss = 0
    val_acc_num = 0
    with torch.no_grad():
        for X, target in test_loader:
            X, target = X.to(device), target.to(device)
            # 前向传播
            output = model(X)
            # 计算损失
            loss = loss_func(output, target)
            # 累加验证损失
            val_loss += loss.item() * X.shape[0]
            # 累加预测正确的数量
            y_pred = output.argmax(dim=1)
            val_acc_num += y_pred.eq(target).sum().item()
    this_val_loss = val_loss / len(test_dataset)
    this_val_acc = val_acc_num / len(test_dataset)

    print(f"Epoch: {epoch + 1}, train loss: {this_train_loss:.4f}, train acc: {this_train_acc:.4f}, "
          f"val loss: {this_val_loss:.4f}, val acc: {this_val_acc:.4f}")

# 5. 测试（预测）
x_new = x_test[666]
plt.imshow(x_new[0], cmap='gray')
plt.show()
print(y_test[666])  # 正确分类标签

# 用模型预测分类
output = model(x_new.unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)
print(y_pred)
