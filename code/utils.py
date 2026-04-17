from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import silhouette_score
from models import MLP, GNN
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def train_model_one_fold(model, data, train_mask, val_mask, test_mask, 
                         lr=0.001, weight_decay=1e-4, max_epochs=500, patience=20, device='cpu'):

    model = model.to(device)
    data = data.to(device)
    
    #fetches the model parameters. this method always exists
    params = model.get_optimizer_params(base_lr=lr)

    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        out, z, h = model(data.x, data.edge_index) 
    
        loss = model.compute_loss(out, h, data.y, train_mask)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out_val, z_val, h_val = model(data.x, data.edge_index)
            val_loss = model.compute_loss(out_val, h_val, data.y, val_mask).item()
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final Test
    model.eval()
    with torch.no_grad():
        out_test, z_test, h_test = model(data.x, data.edge_index)
        
        # Accuracy
        pred = out_test[test_mask].argmax(1)
        test_acc = (pred == data.y[test_mask]).float().mean().item()
        
        # Silhouette Score, always computed 
        test_sil = np.nan
        if h_test is not None:
            h_test_np = h_test[test_mask].cpu().numpy()
            y_test_np = data.y[test_mask].cpu().numpy()
            
            unique_labels = np.unique(y_test_np)
            if len(unique_labels) > 1 and len(h_test_np) > len(unique_labels):
                 test_sil = silhouette_score(h_test_np, y_test_np,metric="cosine")
    
    return test_acc, test_sil

def get_stats(values, n_train=80, n_test=20):

    data = np.array(values)
    avg = np.mean(data)
    std_dev = np.std(data, ddof=1)
    median = np.median(data)
    
    n_total_samples = len(data) # 100
    
    #  (Nadeau & Bengio)
    correction = (1 / n_total_samples) + (n_test / n_train)
    corrected_se = std_dev * np.sqrt(correction)
    
    t_crit = st.t.ppf(0.975, df=99) #this substitutes the 1.96 in the formula and gives a more conservative metric
    margin = t_crit * corrected_se
    
    return median, avg, std_dev, margin

def ttest_corrected(a, b, n_train=80, n_test=20):

    #computes the differences between all metrics
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1) 
    
    #edge case: models are completely equal
    if var_diff == 0:
        return 0.0, 1.0
        
    correction = (1 / n) + (n_test / n_train)
    se_corrected = np.sqrt(correction * var_diff)
    
    t_stat = mean_diff / se_corrected
    df = n - 1
    p_val = st.t.sf(np.abs(t_stat), df) * 2
    
    return t_stat, p_val

def plot_analysis(results_dict, metric_key='accs'):

    models = list(results_dict.keys())
    data = [results_dict[m][metric_key] for m in models]
    
    # Colors: blue for contrastive models, gray for others
    colors = ['steelblue' if '_con' in m else 'lightgray' for m in models]
    
    #BOXPLOT
    plt.figure(figsize=(10, 5))
    bplot = plt.boxplot(data, patch_artist=True, labels=models, showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":8},
                        medianprops={"color": "green", "linewidth": 2})
    #add the colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f'Boxplot Comparison: {metric_key.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='Median'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', label='Mean')]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
