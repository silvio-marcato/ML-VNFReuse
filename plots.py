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
    plt.rcParams['font.size'] = '12'
    df = pd.read_csv('inst_per_vnf.csv')
    index = ['0.472','1.792', '5.432', '5.688', '6.471', '7.136']
    #values = df[['TDA','CIM','DENMg','VE','VC']]
    #values = pd.DataFrame({'TDA': values['TDA'], 'CIM': values['CIM'], 'DENMg': values['DENMg'], 'VE': values['VE'], 'VC': values['VC']}, index=index)
    ax = df.plot.bar(rot=0, linewidth=1, width=0.5, edgecolor='black', zorder=2, x='rates', yticks=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5])
    legend = ax.legend(edgecolor='black', facecolor='white', framealpha=1)
    plt.xlabel('Vehicle arrival rate [veh./s]')
    plt.ylabel('Average no. of instances sharing an NSSI')
    plt.grid()
    plt.show()

def plot_histogram(path):
    plt.rcParams['font.size'] = '12'
    index2 = ['10', '20', '40', '50', '70', '80', '100'] # index to change according to what are you plotting, all columns require 4 values
    # index = ['0.5','0.66', '1', '1.2', '1.33', '1.5', '2', '2.5', '2.66', '3', '3.5', '4', '5.5', '6']
    index = ['0.25', '0.5', '1', '1.33', '1.75', '2.75', '5', '5.5']
    index3 = ['10/20', '20/10', '30/60', '60/30', '40/80', '80/40']
    index4 = ['10','20','30','40','50','60','70','80']
    #index = ['1.1','4', '12.2', '12.8', '14.6', '16.1']
    df = pd.read_csv('bestforalgo copy 2.csv')
    df2 = pd.read_csv('notsharing copy 2.csv')

    ax = plot_gain(df,df2, index, label='vCPU cores gain')
    plt.xlabel('Vehicle arrival rate [veh./s]')
    plt.ylabel("Performance gain [%]")
    plt.show()


    plot_diff_prop('perc2.24_best.csv','perc2.24_notsharing.csv',
    'perc4.5_best.csv','perc4.5_notsharing.csv', 
    'perc6.72_best.csv', 'perc6.72_notsharing.csv', 
    index2, 'Percentage of ST and DaBEV service requests [%]', 'vCPU gain [%]')

    plot_diff_prop('perc_random_best.csv','perc_random_notsharing.csv',
    'perc_random_2_best.csv','perc_random_2_notsharing.csv', 
    'perc_random_3_best.csv', 'perc_random_3_notsharing.csv', 
    index3, 'Percentage of ST and DaBEV service requests [%]', 'vCPU gain [%]')

    plot_diff_prop('DaBEV_fixed_2.24_best.csv','DaBEV_fixed_2.24_notsharing.csv',
    'DaBEV_fixed_4.5_best.csv','DaBEV_fixed_4.5_notsharing.csv', 
    'DaBEV_fixed_6.79_best.csv', 'DaBEV_fixed_6.79_notsharing.csv', 
    index4, 'Percentage of ST service requests with DaBEV requests fixed at 30% [%]', 'vCPU gain [%]')
    plt.savefig('pdf_plots/DaBEV30.pdf', bbox_inches='tight')


    plot_diff_prop('DaBEV_fixed_2.24_20_best.csv','DaBEV_fixed_2.24_20_notsharing.csv',
    'DaBEV_fixed_4.5_20_best.csv','DaBEV_fixed_4.5_20_notsharing.csv', 
    'DaBEV_fixed_6.79_20_best.csv', 'DaBEV_fixed_6.79_20_notsharing.csv', 
    index4, 'Percentage of ST service requests with DaBEV requests fixed at 20% [%]', 'vCPU gain [%]')
    plt.savefig('pdf_plots/DaBEV20.pdf', bbox_inches='tight')

    plot_diff_prop('DaBEV_fixed_2.24_10_best.csv','DaBEV_fixed_2.24_10_notsharing.csv',
    'DaBEV_fixed_4.5_10_best.csv','DaBEV_fixed_4.5_10_notsharing.csv', 
    'DaBEV_fixed_6.79_10_best.csv', 'DaBEV_fixed_6.79_10_notsharing.csv', 
    index4, 'Percentage of ST service requests with DaBEV requests fixed at 10% [%]', 'vCPU gain [%]')
    plt.savefig('pdf_plots/DaBEV10.pdf', bbox_inches='tight')


    plot_diff_prop('DaBEV_fixed_2.24_40_best.csv','DaBEV_fixed_2.24_40_notsharing.csv',
    'DaBEV_fixed_4.5_40_best.csv','DaBEV_fixed_4.5_40_notsharing.csv', 
    'DaBEV_fixed_6.79_40_best.csv', 'DaBEV_fixed_6.79_40_notsharing.csv', 
    index4, 'Percentage of ST service requests with DaBEV requests fixed at 40% [%]', 'vCPU gain [%]')
    plt.savefig('pdf_plots/DaBEV40.pdf', bbox_inches='tight')



def plot_gain(df1, df2, index, withVM=True, init_ax=[], label='vCPU gain [%]'):
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
        ax = cpu_df.plot(rot=0, color=["#fca404"], marker='o', markerfacecolor='none', zorder=3, yticks=[0,10,20,30,40,50,60,70,80], linewidth=2)
    else:
        ax = cpu_df.plot(ax=init_ax, rot=0, marker='o', markerfacecolor='none', yticks=[0,10,20,30,40,50,60,70,80], linewidth=2)
    if withVM:
        vm_df = pd.DataFrame({'Active VMs gain': vm_gain}, index=index)
        #vm_df.plot.bar(ax=ax, rot=0, color=["#1c94fc"], linewidth=1, width=0.4, edgecolor='black', zorder=2)
        vm_df.plot(ax=ax, rot=0, color=["#1c94fc"], marker='o', markerfacecolor='none', yticks=[0,10,20,30,40,50], linewidth=2)
    legend = ax.legend(edgecolor='black', facecolor='white', framealpha=1)
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_facecolor('white', framealpha=1)
    plt.grid()
    return ax

def plot_diff_prop(path_best1, path_ns1, path_best2, path_ns2, path_best3, path_ns3, index, xlabel, ylabel, sub_ax=[]):
    df_best1 = pd.read_csv(path_best1)
    df_best2 = pd.read_csv(path_best2)
    df_best3 = pd.read_csv(path_best3)
    df_ns1 = pd.read_csv(path_ns1)
    df_ns2 = pd.read_csv(path_ns2)
    df_ns3 = pd.read_csv(path_ns3)
    if sub_ax == []:
        ax = plot_gain(df_best1, df_ns1, index=index, withVM=False, label='veh. arrival rate 2.24')
    else:
        ax = plot_gain(df_best1, df_ns1, index=index, withVM=False, init_ax=sub_ax, label='veh. arrival rate 2.24')
    ax = plot_gain(df_best2, df_ns2, index=index, withVM=False, init_ax=ax, label='veh. arrival rate 4.5')
    ax = plot_gain(df_best3, df_ns3, index=index, withVM=False, init_ax=ax, label='veh. arrival rate 6.79')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
       