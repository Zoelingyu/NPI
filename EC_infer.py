import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100
ROI_num = 246
using_steps = 3

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


def half_standard_deviation(signal):
    # 计算每一列的标准差
    std_devs = np.std(signal, axis=0)

    # 计算每一列标准差的一半
    half_std_devs = std_devs / 2

    return half_std_devs

def model_EC(model, input_X, ROI_num, pert_strength):
    "Infer EC by perturbing the surrogate brain"

    node_num = ROI_num
    steps = int(input_X.shape[1] / node_num)
    t = input_X.shape[0]
    NPI_EC = np.zeros((node_num, node_num))
    for node in tqdm(range(node_num)):
        NPI_EC_list = []
        for i in range(t):
            unperturbed_output = model(torch.tensor(input_X[i], dtype=torch.float).to(device)).detach().cpu().numpy()
            perturbation = np.zeros((steps, node_num)); perturbation[-1, node] = pert_strength[node]
            perturbed_output = model(torch.tensor(input_X[i] + perturbation.flatten(), dtype=torch.float).to(device)).detach().cpu().numpy()
            NPI_EC_list.append(perturbed_output - unperturbed_output)
        vectors = np.array(NPI_EC_list)
        avg_vector = np.mean(vectors, axis=0)
        NPI_EC[node] = avg_vector
    return NPI_EC

bold_folder = r'...\lsd01'
checkpoint_folder = r'...\individualized_model\lsd'
ECmatrix_folder = r'...\EC_matrix\lsd'
for checkpoint_file in os.listdir(checkpoint_folder):
    if checkpoint_file.endswith('.pth'):
        sub_id = checkpoint_file.replace('_ANN_checkpoint.pth', '')
        chechponit_path = os.path.join(checkpoint_folder, checkpoint_file)
        checkpoint = torch.load(chechponit_path, map_location=device)

        sub_txt = os.path.join(bold_folder, sub_id + '.txt')

        model = ANN_MLP(input_dim=using_steps * ROI_num, hidden_dim=2 * ROI_num,
                        latent_dim=int(0.8 * ROI_num), output_dim=ROI_num).to(device)

        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)

        signal = np.loadtxt(sub_txt)[3:, :]
        inputs = multi2one(signal, steps=using_steps)[0]
        targets = multi2one(signal, steps=using_steps)[1]

        perturbation = half_standard_deviation(signal)

        EC_matrix = model_EC(model, inputs, ROI_num, perturbation)
        np.save(os.path.join(ECmatrix_folder, sub_id + '_EC.npy'), EC_matrix)






