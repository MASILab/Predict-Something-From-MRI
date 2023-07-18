import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def make_BlandAltman_plot(df_discuss, save_to, figsize=(8, 8)):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    sns.scatterplot(data=df_discuss, x="average", y="diff", hue="Dataset", style="Dataset", ax=ax)

    ax.axhline(df_discuss['diff'].mean(), color='blue', label='mean')
    ax.axhline(df_discuss['diff'].mean()+1.96*df_discuss['diff'].std(), color='red', linestyle='--', label='1.96 SD')
    ax.axhline(df_discuss['diff'].mean()-1.96*df_discuss['diff'].std(), color='red', linestyle='--')
    
    ax.set_xlabel('(Ground Truth + Prediction) / 2')
    ax.set_ylabel('Ground Truth - Prediction')
    ax.set_title('Bland-Altman Plot')
    
    ax.legend()
        
    fig.savefig(save_to)

    
