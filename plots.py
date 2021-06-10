import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.palettes import color_palette
from sklearn.tree import plot_tree
import seaborn as sns
from scipy.interpolate import interp1d

def plot_clf(clf, title, x_train, y_train):
    x = x_train[:, :2]
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, .02), np.arange(y_min, y_max, .02))

    clf.fit(x, y_train)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.figure(figsize=[10,10])
    plt.contourf(xx, yy, z, cmap=plt.cm.RdYlGn, alpha=0.8)
    plt.scatter(x[:,0], x[:,1], c=y_train, cmap=plt.cm.RdYlGn, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('n° of Collision Avoidance service instances')
    plt.ylabel('n° of See Through service instances')
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()

def my_plot_tree(clf_tree):
    features_list = ['n_instances_CA', 'n_instances_ST', 'n_instances_VS']
    label_list = ['0', '1', '2', '3']
    plt.figure()
    plot_tree(clf_tree, class_names=label_list, feature_names=features_list, filled=True)
    plt.show()

def plot_vnfs_inst():
    df = pd.read_csv('inst_per_vnf_bestforalgo2.csv')
    index = ['1.1','4', '12.2', '12.8', '14.6', '16.1']
    #values = df[['TDA','CIM','DENMg','VE','VC']]
    #values = pd.DataFrame({'TDA': values['TDA'], 'CIM': values['CIM'], 'DENMg': values['DENMg'], 'VE': values['VE'], 'VC': values['VC']}, index=index)
    df.plot.bar(rot=0, linewidth=1, width=0.5, edgecolor='black', zorder=2, x='rates')
    plt.xlabel('Overall service arrival rate [requests/s]')
    plt.ylabel('Average no. of instances sharing an SS')
    plt.grid()
    plt.show()

def plot_histogram(path):
    index2 = ['1:8:1', '2:6:2', '3:4:3', '3.5:3:3.5', '4.5:1:4.5', '4:2:4'] # index to change according to what are you plotting, all columns require 4 values
    #index = ['0.75','1', '1.5', '1.8', '2', '2.25', '3', '3.75', '4', '4.5', '5.25', '6', '8.25', '9']
    index = ['1.1','4', '12.2', '12.8', '14.6', '16.1']
    df = pd.read_csv('bestforalgo.csv')
    df2 = pd.read_csv('notsharing.csv')

    ax = plot_gain(df,df2, index, label='vCPU cores gain')
    plt.xlabel('Overall service arrival rates [requests/s]')
    plt.ylabel("Performance gain [%]")
    plt.show()
    ax.get_figure().savefig("prova.pdf", bbox_inches ='tight')
    df_rate1_5 = pd.read_csv('perc_1.5_rates_ST_best.csv')
    df_rate1_5_2 = pd.read_csv('perc_1.5_rates_ST_notshare.csv')    
    df_rate3 = pd.read_csv('perc_3_rates_ST_best.csv')
    df_rate3_2 = pd.read_csv('perc_3_rates_ST_notshare.csv')
    df_rate6 = pd.read_csv('perc_6_rates_ST_best.csv')
    df_rate6_2 = pd.read_csv('perc_6_rates_ST_notshare.csv')
    ax = plot_gain(df_rate1_5, df_rate1_5_2, index2, withVM=False, label='tot_rate = 1.5')
    ax = plot_gain(df_rate3, df_rate3_2, index2, withVM=False, init_ax=ax, label='tot_rate = 3')
    ax = plot_gain(df_rate6, df_rate6_2, index2, withVM=False, init_ax=ax, label='tot_rate = 6')
    plt.xlabel('Proportion between service arrival rates')
    plt.ylabel("vCPU cores gain [%]")
    plt.show()

def plot_gain(df1, df2, index, withVM=True, init_ax=[], label='vCPU cores gain'):
    vcpu_perbin1 = df1[['n_instances_CA','n_instances_ST','n_instances_VS','n_vnf_req','n_vnf', 'n_vcpu', 'bin_conf']]
    vcpu_perbin2 = df2[['n_instances_CA','n_instances_ST','n_instances_VS','n_vnf_req','n_vnf', 'n_vcpu', 'bin_conf']]

    cpu_gain = []
    vm_gain = []

    for share_cpu, not_share_cpu in zip(vcpu_perbin1['n_vcpu'].tolist(), vcpu_perbin2['n_vcpu'].tolist()):

        gain = (not_share_cpu-share_cpu)/not_share_cpu*100
        cpu_gain.append(gain)
    
    for share_vm, not_share_vm in zip(vcpu_perbin1['n_vnf'].tolist(), vcpu_perbin2['n_vnf'].tolist()):
        gain = (not_share_vm-share_vm)/not_share_vm*100
        vm_gain.append(gain)

    cpu_df = pd.DataFrame({label: cpu_gain}, index=index)
    # f1 = interp1d(np.linspace(0,len(index),len(index)), cpu_df[label], kind='cubic')
    # cpu_df = pd.DataFrame()
    # new_index=np.linspace(0,len(index),50)
    # cpu_df[label] = f1(new_index)
    if init_ax == []:
        ax = cpu_df.plot(rot=0, color=["#fca404"], marker='o', markerfacecolor='none', zorder=3)
    else:
        ax = cpu_df.plot(ax=init_ax, rot=0, marker='o', markerfacecolor='none')
    if withVM:
        vm_df = pd.DataFrame({'VM instances gain': vm_gain}, index=index)
        #vm_df.plot.bar(ax=ax, rot=0, color=["#1c94fc"], linewidth=1, width=0.4, edgecolor='black', zorder=2)
        vm_df.plot(ax=ax, rot=0, color=["#1c94fc"], marker='o', markerfacecolor='none')
    legend = ax.legend(edgecolor='black', facecolor='white', framealpha=1)
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_facecolor('white', framealpha=1)
    plt.grid()
    return ax