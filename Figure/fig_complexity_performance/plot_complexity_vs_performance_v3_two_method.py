import pandas as pd
import matplotlib.pyplot as plt

# Performance summary of each method
df_resnet = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/summary_model_performance.csv')
df_baseline = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation/summary_model_performance.csv')

# Making figures
fig, ax = plt.subplots(figsize=(5, 5))

# Part1: ResNet
archs = df_resnet['Arch'].unique()
mlp_hidden_layers = df_resnet['MLP_Hidden_Layer'].unique()
colors = [
    '#fbb4ae',
    '#b3cde3',
    '#ccebc5',
    '#decbe4',
    '#fed9a6',
    '#ffffcc',
    '#e5d8bd',
    '#fddaec',
]
jitter_arch = [-0.2, 0, 0.2]
jitter_mlp = [-0.033, 0.033]
fontsize_txt = 10
fontsize_label = 11

# Iterate over each architecture
for i, arch in enumerate(archs):
    for j, mlp_hidden_layer in enumerate(mlp_hidden_layers):
        # Filter the data for the current Arch and MLP_Hidden_Layer
        filtered_data = df_resnet.loc[(df_resnet['Arch'] == arch) & (df_resnet['MLP_Hidden_Layer'] == mlp_hidden_layer),]
        x = filtered_data['Fold'].str.replace('fold-', '').astype(int) + jitter_arch[i] + jitter_mlp[j]
        y = filtered_data['Val_MAE']
        size = filtered_data['Num_Params'] / 100000  # Divide by a constant to scale the size

        # Plot the scatter plot
        model = f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/ hidden layer" if mlp_hidden_layer == 1 else f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/o hidden layer"
        ax.scatter(x, y, 
                   s=size, c=colors[i], marker='o', 
                   label=model,
                   linewidths=1 if mlp_hidden_layer == 1 else 0,
                   edgecolors='k')

# Part2: ROI-based feature engineering (only plot the best n models)
num_model_to_plot = 3  #if you change this value, you should reconsider colors
top = df_baseline.groupby('Hidden_Layer_Sizes')['Val_MAE'].mean().sort_values(ascending=True).head(num_model_to_plot)
list_hidden_layer_sizes = top.keys()
jitter = [-0.3, 0, 0.3]

for i, hidden_layer_sizes in enumerate(list_hidden_layer_sizes):  
    # Filter the data for the current Arch and MLP_Hidden_Layer
    filtered_data = df_baseline.loc[df_baseline['Hidden_Layer_Sizes']==hidden_layer_sizes,]
    x = filtered_data['Fold'].str.replace('fold-', '').astype(int) + jitter[i]
    y = filtered_data['Val_MAE']
    size = filtered_data['Num_Params'] / 1000  # Divide by a constant to scale the size

    # Plot the scatter plot
    ax.scatter(x, y, s=size, c=colors[i+3], marker='v', linewidths=1, edgecolors='k',
               label=f'MLP {hidden_layer_sizes}')

# Annotations and save the figure
for x in range(1,5):
    ax.axvline(x + 0.5, color='gray', linestyle='--', alpha=0.5)

fold_labels = [f"fold-{i}" for i in range(1,6)]
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(fold_labels, fontsize=fontsize_txt)
ax.set_xlabel('5-fold cross validation', fontsize=fontsize_label)
ax.set_ylabel('Mean absolute error on validation set (year)', fontsize=fontsize_label)
legend = ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1.02), handleheight=1,frameon=False)
legend.set_title('Model', prop={'size': fontsize_label})
for text in legend.get_texts():
    text.set_fontsize(fontsize_txt)

fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_complexity_performance/complexity_vs_performance_validation_v3.png',
            bbox_inches='tight',
            dpi=600)
