import pandas as pd
import matplotlib.pyplot as plt

csv_model_summary = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/summary_model_performance.csv'
df = pd.read_csv(csv_model_summary)
print(df)

# Making figures
archs = df['Arch'].unique()
mlp_hidden_layers = df['MLP_Hidden_Layer'].unique()
colors = ['#b3cde3', '#fbb4ae', '#ccebc5']
jitter_arch = [-0.2, 0, 0.2]
jitter_mlp = [-0.033, 0.033]
fontsize_txt = 10
fontsize_label = 11

fig, ax = plt.subplots(figsize=(5, 5))

# Iterate over each architecture
for i, arch in enumerate(archs):
    for j, mlp_hidden_layer in enumerate(mlp_hidden_layers):
        # Filter the data for the current Arch and MLP_Hidden_Layer
        filtered_data = df.loc[(df['Arch'] == arch) & (df['MLP_Hidden_Layer'] == mlp_hidden_layer),]
        x = filtered_data['Fold'].str.replace('fold-', '').astype(int) + jitter_arch[i] + jitter_mlp[j]
        y = filtered_data['Val_MAE']
        size = filtered_data['Num_Params'] / 100000  # Divide by a constant to scale the size

        # Plot the scatter plot
        model = f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/ hidden layer" if mlp_hidden_layer == 1 else f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/o hidden layer"
        ax.scatter(x, y, 
                   s=size, c=colors[i], marker='o', 
                   label=model,
                   linewidths=2 if mlp_hidden_layer == 1 else 0,
                   edgecolors='k')

# Add vertical lines
for x in range(1,5):
    ax.axvline(x + 0.5, color='gray', linestyle='--', alpha=0.5)

fold_labels = [f"fold-{i}" for i in range(1,6)]
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(fold_labels, fontsize=fontsize_txt)
ax.set_xlabel('5-fold cross validation', fontsize=fontsize_label)
ax.set_ylabel('Mean absolute error on validation set (year)', fontsize=fontsize_label)
legend = ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1.02), handleheight=6.3)
legend.set_title('Model', prop={'size': fontsize_label})
for text in legend.get_texts():
    text.set_fontsize(fontsize_txt)

fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_complexity_performance/complexity_vs_performance_validation_v2.png',
            bbox_inches='tight',
            dpi=500)
