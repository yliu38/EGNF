import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import pandas as pd
import random
import gc
import pickle
from sklearn.base import BaseEstimator

# ------------------------- Argument Parser --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n_fea', type=int, required=True, help='Number of features (e.g., 32)')
parser.add_argument('--label_file', type=str, required=True, help='Path to label file')
parser.add_argument('--label_train', type=str, required=True, help='Path to train label file')
parser.add_argument('--label_test', type=str, required=True, help='Path to test label file')
parser.add_argument('--atoms_df', type=str, required=True, help='Path to atoms dataframe CSV')
parser.add_argument('--bonds_df', type=str, required=True, help='Path to bonds dataframe CSV')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory for results')
args = parser.parse_args()

n_fea = args.n_fea
label_file = pd.read_csv(args.label_file)
label_train = pd.read_csv(args.label_train)
label_test = pd.read_csv(args.label_test)
atoms_df = pd.read_csv(args.atoms_df)
bonds_df = pd.read_csv(args.bonds_df)
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# ------------------------- Model Definition --------------------------
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers, dropout=0.5):
        super(GCNClassifier, self).__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

# ------------------------- Helper Functions --------------------------
def train(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_preds.extend(torch.exp(out)[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    accuracies = [(all_preds >= t).astype(int).mean() for t in thresholds]
    return thresholds[np.argmax(accuracies)], auc

# ------------------------- Dataset Preparation --------------------------
atoms_df['samples'] = atoms_df['samples'].astype(str)
nodes = pd.DataFrame([])

for spl_id in label_file['sid']:
    tmp = atoms_df.loc[atoms_df['samples'].str.contains(spl_id, regex=False)]
    tmp.insert(6, 'sample_id', spl_id)
    nodes = nodes.append(tmp.iloc[:, 1:], ignore_index=True)

nodes = pd.merge(nodes, label_file, left_on='sample_id', right_on='sid', how='left')
nodes = nodes.rename(columns={'group': 'y'})
nodes['Row_Index'] = nodes.groupby('sample_id').cumcount()

bonds_df['common_samples'] = bonds_df['common_samples'].astype(str)
bonds_df = bonds_df.iloc[:, [0, 1, 4, 5, 6, 7]]

edges = pd.DataFrame([])
for spl_id in label_file['sid']:
    tmp = bonds_df.loc[bonds_df['common_samples'].str.contains(spl_id, regex=False)]
    tmp.insert(6, 'sample_id', spl_id)
    edges = edges.append(tmp.iloc[:, 1:], ignore_index=True)

id_mapping = dict(zip(nodes['atom_id'].astype(str) + '_' + nodes['sample_id'].astype(str), nodes['Row_Index']))
edges['atom_2'] = [id_mapping[item] for item in edges['atom_0'].astype(str) + '_' + edges['sample_id'].astype(str)]
edges['atom_3'] = [id_mapping[item] for item in edges['atom_1'].astype(str) + '_' + edges['sample_id'].astype(str)]

nodes['y'] = nodes['y'].replace({'Normal': 0, 'Tumor': 1}).astype(int)
nodes = pd.get_dummies(nodes, columns=['gene'], dtype=int)

def create_dataset(label_df):
    dataset = []
    for sid in label_df['sid']:
        y = torch.tensor(nodes[nodes['sample_id'] == sid]['y'].unique(), dtype=torch.int64)
        x = torch.tensor(nodes[nodes['sample_id'] == sid].iloc[:, list(range(1, 4)) + list(range(9, nodes.shape[1]))].to_numpy(), dtype=torch.float32)
        x = F.normalize(x, p=2.0, dim=0)
        if (x == 0).all(1).any(): continue
        edge_attr = torch.tensor(edges[edges['sample_id'] == sid]['num_samples'].to_numpy(), dtype=torch.float32)
        edge_attr = F.normalize(edge_attr, p=2.0, dim=0)
        if torch.all(edge_attr == 0): continue
        edge_index = torch.tensor(edges[edges['sample_id'] == sid][['atom_2', 'atom_3']].to_numpy().T, dtype=torch.int64)
        dataset.append(Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y))
    return dataset

train_val_dataset = create_dataset(label_train)
test_dataset = create_dataset(label_test)

class CustomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

train_val_dataset = CustomGraphDataset(train_val_dataset)
test_dataset = CustomGraphDataset(test_dataset)

# ------------------------- Hyperparameter Optimization --------------------------
class GCNWrapper(BaseEstimator):
    def __init__(self, hidden_dim=16, num_layers=2, learning_rate=0.01, batch_size=32, epochs=100, dropout=0.5, weight_decay=0.0001):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay

    def fit(self, X, y=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_model = None
        best_auc = 0
        for train_idx, val_idx in kf.split(X, [data.y.item() for data in X]):
            train_data = [X[i] for i in train_idx]
            val_data = [X[i] for i in val_idx]
            model = GCNClassifier(n_fea+3, self.hidden_dim, 2, self.num_layers, self.dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size)
            for _ in range(self.epochs): train(train_loader, model, criterion, optimizer, device)
            _, val_auc = test(val_loader, model, device)
            if val_auc > best_auc:
                best_model = model
                best_auc = val_auc
        self.best_model_ = best_model
        self.mean_val_auc_ = best_auc
        return self

    def predict(self, X):
        self.best_model_.eval()
        loader = DataLoader(X, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for data in loader:
                out = self.best_model_(data.x, data.edge_index, data.edge_attr, data.batch)
                preds.append(out.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def score(self, X, y):
        return self.mean_val_auc_

search_space = {
    'hidden_dim': Integer(16, 64),
    'num_layers': Integer(2, 4),
    'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    'batch_size': Integer(8, 32),
    'epochs': Integer(50, 200),
    'dropout': Real(0.1, 0.6),
    'weight_decay': Real(1e-5, 1e-3, prior='log-uniform')
}

opt = BayesSearchCV(estimator=GCNWrapper(), search_spaces=search_space, n_iter=50, cv=[(slice(None), slice(None))], n_jobs=1)
opt.fit(train_val_dataset, [data.y.item() for data in train_val_dataset])

print(f"Best parameters: {opt.best_params_}")
print(f"Best cross-validation AUC: {opt.best_score_:.4f}")

# ------------------------- Final Evaluation --------------------------
best_params = opt.best_params_
model = GCNClassifier(n_fea+3, best_params['hidden_dim'], 2, best_params['num_layers'], best_params['dropout']).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_val_dataset, batch_size=best_params['batch_size'], shuffle=True)

for _ in range(best_params['epochs']):
    train(train_loader, model, criterion, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
acc, auc = test(test_loader, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# ------------------------- Save Result --------------------------
result = {"test_accuracy": acc, "test_auc": auc}
with open(os.path.join(out_dir, 'gcn_result.pickle'), 'wb') as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
