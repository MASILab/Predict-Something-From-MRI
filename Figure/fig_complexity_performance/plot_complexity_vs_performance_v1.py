import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models_copy import *
from train_copy import select_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

path_weights_folder = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/weights')

list_model_tested = [fn.name for fn in path_weights_folder.iterdir() if fn.is_dir()]

list_df_fullname = []
list_df_arch = []
list_df_mlp_hidden_layer = []
list_df_num_params = []
list_df_fold = []
list_df_valmse = []

for fullname in list_model_tested:
    arch = fullname.split('_')[0]
    
    if 'noMLP' in fullname:
        mlp_hidden_layer = 0
    else:
        mlp_hidden_layer = 1
    
    num_params = count_parameters(select_model(fullname))
    
    for i in [1,2,3,4,5]:
        
        path_fold = path_weights_folder / fullname / f'fold-{i}'
        pth_fn = [fn.name for fn in path_fold.iterdir() if fn.name.endswith('.pth')]
        pth_fn = pth_fn[0]
        
        valmse = float(pth_fn.split('_valloss-')[1].replace('.pth', ''))
        
        list_df_fullname.append(fullname)
        list_df_arch.append(arch)
        list_df_mlp_hidden_layer.append(mlp_hidden_layer)
        list_df_num_params.append(num_params)
        list_df_fold.append(i)
        list_df_valmse.append(valmse)
    
d = {'Full_Name': list_df_fullname,
     'Arch': list_df_arch,
     'MLP_Hidden_Layer': list_df_mlp_hidden_layer,
     'Num_Params': list_df_num_params,
     'Fold': list_df_fold,
     'MSE_Validation': list_df_valmse}

df = pd.DataFrame(data=d)

# Making figures

archs = df['Arch'].unique()
mlp_hidden_layers = df['MLP_Hidden_Layer'].unique()
colors = ['tab:blue', 'tab:orange', 'tab:green']
jitter_arch = [-0.2, 0, 0.2]
jitter_mlp = [-0.033, 0.033]
fontsize_txt = 10
fontsize_label = 11

fig, ax = plt.subplots(figsize=(5, 5))

# Iterate over each architecture
for i, arch in enumerate(archs):
    for j, mlp_hidden_layer in enumerate(mlp_hidden_layers):
        # Filter the data for the current Arch and MLP_Hidden_Layer
        filtered_data = df[(df['Arch'] == arch) & (df['MLP_Hidden_Layer'] == mlp_hidden_layer)]
        x = filtered_data['Fold'] + jitter_arch[i] + jitter_mlp[j]
        y = filtered_data['MSE_Validation']
        size = filtered_data['Num_Params'] / 100000  # Divide by a constant to scale the size

        # Plot the scatter plot
        model = f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/ hidden layer" if mlp_hidden_layer == 1 else f"{arch.replace('r','R').replace('n','N')}\n+\nMLP w/o hidden layer"
        ax.scatter(x, y, s=size, c=colors[i], marker='o', label=model,
                    linewidths=2 if mlp_hidden_layer == 1 else 0,
                    edgecolors='k')

for x in range(1,5):
    ax.axvline(x + 0.5, color='gray', linestyle='--', alpha=0.5)

fold_labels = [f"fold-{i}" for i in df['Fold'].unique()]
ax.set_xticks(df['Fold'].unique())
ax.set_xticklabels(fold_labels, fontsize=fontsize_txt)
ax.set_xlabel('5-Fold Cross Validation', fontsize=fontsize_label)
ax.set_ylabel('Mean Squared Error on Validation Set', fontsize=fontsize_label)
legend = ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1.02), handleheight=6.3)
legend.set_title('Model', prop={'size': fontsize_label})  # Adjust fontsize here
for text in legend.get_texts():
    text.set_fontsize(fontsize_txt)


fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_complexity_performance/complexity_vs_performance_validation_v1.png',
            bbox_inches='tight',
            dpi=300)




# jitter = 0.1


# arch_palette = sns.color_palette("tab10", len(df['Arch'].unique()))

# # Define the size range for the scatter points based on the number of parameters
# size_range = (100, 600)

# # Plot the scatter plot
# plt.figure(figsize=(10, 6))

# # Plot MLP_Hidden_Layer=0 with regular scatter points
# sns.scatterplot(x='Fold', y='MSE_Validation', data=df[df['MLP_Hidden_Layer'] == 0], hue='Arch', size='Num_Params', sizes=size_range, palette=arch_palette)

# # Plot MLP_Hidden_Layer=1 with thicker lines (linestyles='-')
# sns.scatterplot(x='Fold', y='MSE_Validation', data=df[df['MLP_Hidden_Layer'] == 1], hue='Arch', size='Num_Params', sizes=size_range, linewidth=2, edgecolor='black', palette=arch_palette)

# # Show the legend outside of the plot
# plt.legend(title='Architecture', loc='upper left', bbox_to_anchor=(1, 1))

# # Set axis labels
# plt.xlabel('Fold')
# plt.ylabel('Performance (MSE_Validation)')

# # Show the plot
# plt.show()