import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
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

def multi2one(time_series, steps):

    "Split the data into several input-output pairs"

    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps+i].flatten()
        target_Y[i] = time_series[steps+i].flatten()
    return np.array(input_X), np.array(target_Y)

def corrcoef(signals):

    "Calculate FC of the empirical data"

    return torch.corrcoef(torch.tensor(signals.T, dtype=torch.float).to(device)).detach().cpu().numpy()

def train_NN(model, input_X, target_Y, batch_size, num_epochs, lr = 1e-3, l2 = 1e-5):

    "Use all data to train the model without train-test split"

    inputs = torch.tensor(input_X, dtype=torch.float).to(device)
    targets = torch.tensor(target_Y, dtype=torch.float).to(device)
    dataset = data.TensorDataset(inputs, targets)
    train_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    loss = nn.MSELoss()
    trainer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    for param in model.func[:-1].parameters():
        param.requires_grad = False

    for param in model.func[-1].parameters():
        param.requires_grad = True

    train_epoch_loss = []
    for _ in tqdm(range(num_epochs)):

        model.train()

        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0; total_num = 0
            for X, y in train_iter:
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            train_epoch_loss.append(float(total_loss / total_num))

    return model, train_epoch_loss, trainer

def sim_FC(model, signals, inputs, node_num, steps):

    "simulate data with the first 3 timepoints and calculate model-FC"

    t1_t3 = np.squeeze(inputs[:1])
    noise = 0.1 * np.random.randn(steps * node_num)
    input_t1_t3 = t1_t3 + noise
    NN_sim = []
    for row in signals[:3]:
        NN_sim.append(row)
    t4 = model(torch.tensor(input_t1_t3, dtype=torch.float).to(device)).detach().cpu().numpy()
    NN_sim.append(t4)
    for _ in range(211):
        noise = 0.1 * np.random.randn(steps * node_num)
        model_input = np.array(NN_sim[-steps:]).flatten() + noise
        NN_sim.append(model(torch.tensor(model_input, dtype=torch.float).to(device)).detach().cpu().numpy())
    NN_sim = np.array(NN_sim)
    return corrcoef(NN_sim)

batch_size = 100
ROI_num = 246
using_steps = 3

checkpoint = torch.load(r'...\ANN_checkpoint_epoch100_lre-5.pth', map_location=device)
sub_txt = r'...\lsd01_001.txt'

model = ANN_MLP(input_dim = using_steps * ROI_num, hidden_dim = 2 * ROI_num, latent_dim=int(0.8 * ROI_num), output_dim = ROI_num).to(device)

model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)

signal = np.loadtxt(sub_txt)[3:, :]
inputs = multi2one(signal, steps=using_steps)[0]
targets = multi2one(signal, steps=using_steps)[1]

empirical_FC = corrcoef(signal)

individualized_ANN, training_loss, trainer = train_NN(model, inputs, targets, batch_size, num_epochs = 2000, lr = 1e-3, l2 = 1e-10)

plt.plot(training_loss, label = 'Training loss')
plt.legend(loc = 'upper right')
plt.show()

model_FC = sim_FC(individualized_ANN, signal, inputs, node_num = ROI_num, steps = using_steps)

r_value, p_value = pearsonr(flat_without_diagnal(model_FC), flat_without_diagnal(empirical_FC))

print(f"r_value: {r_value:.5f}")
print(f"p_value: {p_value:.5f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
sns.heatmap(empirical_FC, ax = ax1, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
sns.heatmap(model_FC, ax = ax2, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
ax1.set_title('Empirical FC')
ax2.set_title('Model FC')
plt.show()

# torch.save({
#     'model_state_dict': individualized_ANN.state_dict(),  # 保存模型的参数
#     'optimizer_state_dict': trainer.state_dict(),  # 保存优化器的状态
#     'train_loss': training_loss,  # 保存训练的损失
# }, checkpoint_path)












