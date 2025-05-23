import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import os
import random
import pandas as pd
from sklearn.base import BaseEstimator
import gc
import pickle


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_fea', type=int, required=True, help='Number of input features')
parser.add_argument('--label_file', type=str, required=True, help='Path to label file')
parser.add_argument('--label_train', type=str, required=True, help='Path to train label file')
parser.add_argument('--label_test', type=str, required=True, help='Path to test label file')
parser.add_argument('--atoms_df', type=str, required=True, help='Path to atoms dataframe')
parser.add_argument('--bonds_df', type=str, required=True, help='Path to bonds dataframe')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save results')
args = parser.parse_args()

n_fea = args.n_fea


# Set max_split_size_mb to reduce fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers, heads=1, dropout=0.5, edge_attr_dim=3):
        super(GATClassifier, self).__init__()
        self.edge_attr_dim = edge_attr_dim
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True, edge_dim=self.edge_attr_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=self.edge_attr_dim))
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, edge_dim=self.edge_attr_dim))
        self.dropout = dropout
    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            # Collect probabilities for the positive class (for ROC curve)
            all_preds.extend(torch.exp(out)[:, 1].cpu().numpy()) 
            all_labels.extend(data.y.cpu().numpy())   
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)  
    # Find the best threshold that maximizes accuracy
    accuracies = []
    for thresh in thresholds:
        predictions = (all_preds >= thresh).astype(int)
        accuracy = (predictions == all_labels).mean()
        accuracies.append(accuracy)
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = np.max(accuracies)
    return best_accuracy, auc

class GATWrapper(BaseEstimator):
    def __init__(self, hidden_dim=16, num_layers=2, learning_rate=0.01, batch_size=8, epochs=100, dropout=0.5, weight_decay=0.0001, edge_attr_dim=3, heads=2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.edge_attr_dim = edge_attr_dim
        self.heads = heads 
    def fit(self, X, y=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = X
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.best_model_ = None
        best_mean_val_auc = 0
        fold_val_AUC = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, [data.y.item() for data in dataset])):
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            model = GATClassifier(  # Assuming you now use 
                in_channels=n_fea+3,  # Adjust according to your input features
                hidden_dim=self.hidden_dim,
                num_classes=2,
                num_layers=self.num_layers,
                dropout=self.dropout,
                heads=self.heads,
                edge_attr_dim=self.edge_attr_dim  # Pass edge_attr_dim to the model
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=2)
            print(f"Hidden Dimension: {self.hidden_dim}, Number of Layers: {self.num_layers}, Learning Rate: {self.learning_rate}, Batch Size: {self.batch_size}, Dropout: {self.dropout}, Weight: {self.weight_decay}")
            for epoch in range(self.epochs):
                train_loss = train(train_loader, model, criterion, optimizer, device)
                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}, Loss: {train_loss:.4f}")    
            val_auc = test(val_loader, model, device)[1]
            fold_val_AUC.append(val_auc)
            print(f"Fold {fold + 1}, Validation AUC: {val_auc:.4f}")
        mean_val_auc = np.mean(fold_val_AUC)
        std_val_auc = np.std(fold_val_AUC)
        print(f"Mean Validation AUC: {mean_val_auc:.4f}")
        print(f"Validation AUC Std Dev: {std_val_auc:.4f}")
        if mean_val_auc > best_mean_val_auc:
            best_mean_val_auc = mean_val_auc
            self.best_model_ = model        
        self.mean_val_auc_ = best_mean_val_auc
        self.std_val_auc_ = std_val_auc
        return self
    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = DataLoader(X, batch_size=self.batch_size, num_workers=2)
        preds = []
        self.best_model_.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = self.best_model_(data.x, data.edge_index, data.edge_attr, data.batch)  # Added data.edge_attr
                pred = out.argmax(dim=1)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)
    def score(self, X, y):
        return self.mean_val_auc_


# set seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load labels
label_file = pd.read_csv(args.label_file)
label_train = pd.read_csv(args.label_train)
label_test = pd.read_csv(args.label_test)

# prepare nodes data for the model
atoms_df = pd.read_csv(args.atoms_df)
atoms_df['samples'] = atoms_df['samples'].astype(str)
atoms_df.shape

nodes = pd.DataFrame([])
for spl_id in label_file['sid']:
    tmp = atoms_df.loc[atoms_df['samples'].str.contains(spl_id, regex=False)]
    tmp.insert(6, 'sample_id', spl_id)
    nodes = nodes.append(tmp.iloc[:,1:], ignore_index=True)

nodes = pd.merge(nodes, label_file, left_on='sample_id', right_on='sid', how='left')
nodes = nodes.rename(columns={'group': 'y'})

nodes['Row_Index'] = nodes.groupby('sample_id').cumcount()



# prepare edges data for the model
bonds_df = pd.read_csv(args.bonds_df)
bonds_df['common_samples'] = bonds_df['common_samples'].astype(str)
bonds_df.shape

bonds_df = bonds_df.iloc[:,[0,1,4,5,6,7]]


edges = pd.DataFrame([])
for spl_id in label_file['sid']:
    tmp = bonds_df.loc[bonds_df['common_samples'].str.contains(spl_id, regex=False)]
    tmp.insert(6, 'sample_id', spl_id)
    edges = edges.append(tmp.iloc[:,1:], ignore_index=True)
    
del atoms_df
del bonds_df


id_mapping = dict(zip(nodes['atom_id'].astype(str) + '_' + nodes['sample_id'].astype(str), nodes['Row_Index']))
edges['atom_2'] = [id_mapping[item] for item in edges['atom_0'].astype(str)+ '_' + edges['sample_id'].astype(str)]
edges['atom_3'] = [id_mapping[item] for item in edges['atom_1'].astype(str)+ '_' + edges['sample_id'].astype(str)]

# create dummy variables or replace labels to numeric values
nodes['y'] = nodes['y'].replace({'Normal': '0', 'Tumor': '1'})
nodes['y'] = nodes['y'].astype(int)
nodes = pd.get_dummies(nodes, columns=['gene'], dtype=int)


# generate inputs for model
train_val_dataset = []
for spl_id in label_train['sid']:
    y = torch.tensor(nodes.loc[nodes['sample_id'] == spl_id]['y'].unique(), dtype=torch.int64)
    x = torch.tensor(nodes.loc[nodes['sample_id'] == spl_id].iloc[:,list(range(1,4)) + list(range(9,nodes.shape[1]))].to_numpy(), dtype=torch.float32)
    x = torch.nn.functional.normalize(x, p=2.0, dim=0)
    zero_rows = (x == 0).all(dim=1)
    if zero_rows.any():
        continue
    edge_attr = torch.tensor(edges.loc[edges['sample_id'] == spl_id].iloc[:,list(range(0,1))+ list(range(3,5))].to_numpy(), dtype=torch.float32)
    edge_attr = torch.nn.functional.normalize(edge_attr, p=2.0, dim=0)
    if torch.all(edge_attr == 0):
        continue
    edge_index = torch.tensor(edges.loc[edges['sample_id'] == spl_id][['atom_2', 'atom_3']].to_numpy().T, dtype=torch.int64)
    train_val_dataset.append(Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y))
    
test_dataset = []
for spl_id in label_test['sid']:
    y = torch.tensor(nodes.loc[nodes['sample_id'] == spl_id]['y'].unique(), dtype=torch.int64)
    x = torch.tensor(nodes.loc[nodes['sample_id'] == spl_id].iloc[:,list(range(1,4)) + list(range(9,nodes.shape[1]))].to_numpy(), dtype=torch.float32)
    x = torch.nn.functional.normalize(x, p=2.0, dim=0)
    zero_rows = (x == 0).all(dim=1)
    if zero_rows.any():
        continue
    edge_attr = torch.tensor(edges.loc[edges['sample_id'] == spl_id].iloc[:,list(range(0,1))+ list(range(3,5))].to_numpy(), dtype=torch.float32)
    edge_attr = torch.nn.functional.normalize(edge_attr, p=2.0, dim=0)
    if torch.all(edge_attr == 0):
        continue
    edge_index = torch.tensor(edges.loc[edges['sample_id'] == spl_id][['atom_2', 'atom_3']].to_numpy().T, dtype=torch.int64)
    test_dataset.append(Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y))

del nodes
del edges
torch.cuda.empty_cache()
gc.collect()

class CustomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

train_val_dataset = CustomGraphDataset(train_val_dataset)
test_dataset = CustomGraphDataset(test_dataset)


# Define the parameter search space
search_space = {
    'hidden_dim': Integer(16, 64),
    'num_layers': Integer(2, 4),
    'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    'batch_size': Integer(4, 8),
    'epochs': Integer(50, 200),
    'dropout': Real(0.1, 0.6),
    'heads': Integer(1, 2),
    'weight_decay': Real(1e-5, 1e-3, prior='log-uniform')
}

# Run Bayesian optimization
opt = BayesSearchCV(estimator=GATWrapper(), search_spaces=search_space, n_iter=50, cv=[(slice(None), slice(None))], n_jobs=1)
opt.fit(train_val_dataset, [data.y.item() for data in train_val_dataset])

print(f"Best parameters: {opt.best_params_}")
print(f"Best cross-validation AUC: {opt.best_score_:.4f}")

# Evaluate the best model on the test set
best_params = opt.best_params_

test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], num_workers=2)
model = GATClassifier(
    in_channels=n_fea+3,  # Adjust according to your input features
    hidden_dim=best_params['hidden_dim'],
    num_classes=2,
    edge_attr_dim=3,
    num_layers=best_params['num_layers'],
    heads=best_params['heads'],  # You can adjust this based on your model's needs
    dropout=best_params['dropout']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
criterion = nn.CrossEntropyLoss()

# Training on the combined train and validation set with best hyperparameters
train_loader = DataLoader(train_val_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)
for epoch in range(best_params['epochs']):
    train(train_loader, model, criterion, optimizer, device)

test_acc, test_auc = test(test_loader, model, device)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

result = {
    "test_accuracy": test_acc,
    "test_auc": test_auc
}

with open(os.path.join(args.out_dir, 'result.pickle') , 'wb') as fl:
        pickle.dump(result, fl, pickle.HIGHEST_PROTOCOL)
