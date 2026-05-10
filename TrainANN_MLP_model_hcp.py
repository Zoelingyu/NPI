import os
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flat_without_diagnal(matrix):

    "Flatten the matrix without including the diagnal"

    n = matrix.shape[0]
    flattened = []
    for i in range(n):
        for j in list(range(i)) + list(range(i + 1, n)):
            flattened.append(matrix[i][j])
    return np.array(flattened)

def multi2one(time_series, steps):
    "Split the data into several input-output pairs"

    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps + i].flatten()
        target_Y[i] = time_series[steps + i].flatten()
    return np.array(input_X), np.array(target_Y)

class ANN_MLP(nn.Module):
    "Use MLP as a surrogate brain"

    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        ).to(device)

    def forward(self, x):
        return self.func(x)

def train_NN(model, input_X, target_Y, batch_size=100, train_set_proportion=0.9, num_epochs=100, lr=1e-5, l2=0):
    "Use empirical data to tune the model"

    train_inputs = torch.tensor(input_X[:int(train_set_proportion * input_X.shape[0])], dtype=torch.float).to(device)
    train_targets = torch.tensor(target_Y[:int(train_set_proportion * target_Y.shape[0])], dtype=torch.float).to(device)
    test_inputs = torch.tensor(input_X[int(train_set_proportion * input_X.shape[0]):], dtype=torch.float).to(device)
    test_targets = torch.tensor(target_Y[int(train_set_proportion * target_Y.shape[0]):], dtype=torch.float).to(device)
    train_dataset = data.TensorDataset(train_inputs, train_targets)
    test_dataset = data.TensorDataset(test_inputs, test_targets)
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle=False)

    loss = nn.MSELoss()
    trainer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    train_epoch_loss = [];
    test_epoch_loss = []
    for _ in tqdm(range(num_epochs)):
        model.train()
        for X, y in tqdm(train_iter):
            y_hat = model(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0;
            total_num = 0
            for X, y in tqdm(train_iter):
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            train_epoch_loss.append(float(total_loss / total_num))
            total_loss = 0;
            total_num = 0
            for X, y in test_iter:
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            test_epoch_loss.append(float(total_loss / total_num))
    return model, train_epoch_loss, test_epoch_loss, trainer

# NPI uisage demo
batch_size = 100
train_set_proportion = 0.9
ROI_num = 246
using_steps = 3

signals = []
inputs = []
targets = []
path_sub = r'.../txt_1200-246_NPIUsed_traintest/'

for file in os.listdir(path_sub):
    path_file = os.path.join(path_sub, file)
    signals.append(np.loadtxt(path_file)[30:, :])
    inputs.append(multi2one(signals[-1], steps=using_steps)[0])
    targets.append(multi2one(signals[-1], steps=using_steps)[1])
    print('inputs append: ', file)

signals = np.vstack(signals)
inputs = np.vstack(inputs)
targets = np.vstack(targets)

# train NN
ANN = ANN_MLP(input_dim = using_steps * ROI_num, hidden_dim = 2 * ROI_num, latent_dim=int(0.8 * ROI_num), output_dim = ROI_num)

ANN, training_loss, testing_loss, trainer = train_NN(ANN, inputs, targets, num_epochs = 100, lr = 1e-5, l2 = 5e-5)

torch.save(ANN.state_dict(), r'.../ANN_model1_epoch100_lre-5.pth')

torch.save({
    'model_state_dict': ANN.state_dict(),  # 保存模型的参数
    'optimizer_state_dict': trainer.state_dict(),  # 保存优化器的状态
    'train_loss': training_loss,  # 保存训练的损失
    'test_loss': testing_loss,  # 保存测试的损失
}, r'.../ANN_checkpoint1_epoch100_lre-5.pth')