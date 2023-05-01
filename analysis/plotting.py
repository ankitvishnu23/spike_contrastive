from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
plt.rc("figure", dpi=100)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def plot_all_pts(og_reps, title, save_name=None):
    dim = og_reps.shape[1]
    num_axes = 2 if dim > 2 else 1

    fig = plt.figure(figsize=(num_axes*5, 5))

    ax0 = fig.add_subplot(1, num_axes, 1)
    ax0.scatter(og_reps[:, 0], og_reps[:, 1], c='blue', clip_on=False)
    # ax0.scatter(tform_reps[:, 0], tform_reps[:, 1], c='blue', clip_on=False)
    ax0.set_title('first 2 dims')
    if dim > 2:
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.scatter(og_reps[:, 2], og_reps[:, 3], og_reps[:, 4], c='red', clip_on=False)
        # ax1.scatter(tform_reps[:, 2], tform_reps[:, 3], tform_reps[:, 4], c='red', clip_on=False)
        ax1.set_title('next 3 dims')
    plt.suptitle(title)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
        
        
def plot_contr_v_pca(pca_reps, contr_reps, wfs, wf_interest, title=None, save_name=None, wf_selection=None):
    og_wfs = wfs[wf_interest]
    n_temps = len(pca_reps)
    lat_dim = pca_reps.shape[1]
    num_wfs = len(og_wfs)
    
    max_chan_max = np.max(np.max(og_wfs, axis=1))
    max_chan_min = np.min(np.min(og_wfs, axis=1))
    # max_chan_max = max([np.max(temp) for temp in tot_temps])
    # max_chan_min = min([np.min(temp) for temp in tot_temps])
    if wf_selection is None:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][:num_wfs]
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][wf_selection[0]:wf_selection[1]]
        print(colors)
    num_reps = int(len(pca_reps) / num_wfs)
    print(num_reps)
    labels = np.array([[colors[i] for j in range(num_reps)] for i in range(num_wfs)])
    labels = labels.flatten()
    print(labels.shape)
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(4, num_wfs, figure=fig)
    
    ax0 = fig.add_subplot(gs[:3, :int(num_wfs/2)])
    ax0.title.set_text('PCA wf representations')
    ax0.scatter(pca_reps[:, 0], pca_reps[:, 1], c=labels, clip_on=False)
    
    ax1 = fig.add_subplot(gs[:3, int(num_wfs/2):])
    ax1.title.set_text('Contrastive wf representations')
    ax1.scatter(contr_reps[:, 0], contr_reps[:, 1], c=labels, clip_on=True) 
    # ax1.set_xlim([0, 25])
    # ax1.set_ylim([-7, 15])
    
    axs = [fig.add_subplot(gs[3, i]) for i in range(num_wfs)]
        
    x = np.arange(0, 121)

    for i in range(num_wfs):
        # axs[0] = fig.add_subplot(gs[i//2, 2 + 2*(i%2)])
        axs[i].set_ylim(max_chan_min-0.5, max_chan_max+0.5)
        axs[i].title.set_text('unit {}'.format(str(wf_interest[i])))
        axs[i].plot(x, og_wfs[i], linewidth=2, markersize=12, color=colors[i])
        axs[i].get_xaxis().set_visible(False)
    
    # fig.subplots_adjust(wspace=0)

    fig.suptitle(title)
    
    if save_name is not None:
        plt.savefig(save_name)
        
def plot_recon_v_spike(wf_train, wf_test, wfs, wf_interest, ckpt, lat_dim, title, save_name=None, wf_selection=None):
    og_wfs = wfs[wf_interest]
    tot_spikes, n_times = wf_test.shape
    spike_sel = np.random.choice(tot_spikes)
    spike = wf_test[spike_sel]
    num_wfs = 10
    
    pca_aug = PCA_Reproj()
    pca_train = np.array([pca_aug(wf) for wf in wf_train])
    pca_test = np.array([pca_aug(wf) for wf in wf_test])
    
    _, contr_spikes_test, contr_spikes_test_pca, _, pca_spikes_test = get_ckpt_results(ckpt, lat_dim, wf_train, wf_test)
    # contr_spikes_test_pca = contr_spikes_test_pca.reshape(4, num_ex, -1)
    # pca_spikes_test = pca_spikes_test.reshape(4, num_ex, -1)
    
    _, contr_recon_test, contr_recon_test_pca, _, pca_recon_test = get_ckpt_results(ckpt, lat_dim, pca_train, pca_test)
    # contr_recon_test_pca = contr_recon_test_pca.reshape(4, num_ex, -1)
    # pca_spikes_test = pca_spikes_test.reshape(4, num_ex, -1)
    
    max_chan_max = np.max(np.max(og_wfs, axis=1))
    max_chan_min = np.min(np.min(og_wfs, axis=1))
    # max_chan_max = max([np.max(temp) for temp in tot_temps])
    # max_chan_min = min([np.min(temp) for temp in tot_temps])
    if wf_selection is None:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][:num_wfs]
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink'][wf_selection[0]:wf_selection[1]]
        print(colors)
    num_reps = int(len(wf_test) / num_wfs)
    print(num_reps)
    labels = np.array([[colors[i] for j in range(num_reps)] for i in range(num_wfs)])
    labels = labels.flatten()
    print(labels.shape)
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(4, num_wfs, figure=fig)
    
    ax0 = fig.add_subplot(gs[:3, :int(num_wfs/2)])
    ax0.title.set_text('Contrastive spike representations')
    ax0.scatter(contr_spikes_test_pca[:, 0], contr_spikes_test_pca[:, 1], c=labels, clip_on=False)
    
    ax1 = fig.add_subplot(gs[:3, int(num_wfs/2):])
    ax1.title.set_text('Contrastive pca recon. spike representations')
    ax1.scatter(contr_recon_test_pca[:, 0], contr_recon_test_pca[:, 1], c=labels, clip_on=True) 
    # ax1.set_xlim([0, 25])
    # ax1.set_ylim([-7, 15])
    
    axs = [fig.add_subplot(gs[3, i]) for i in range(num_wfs)]
        
    x = np.arange(0, 121)

    for i in range(num_wfs):
        # axs[0] = fig.add_subplot(gs[i//2, 2 + 2*(i%2)])
        axs[i].set_ylim(max_chan_min-0.5, max_chan_max+0.5)
        axs[i].title.set_text('unit {}'.format(str(wf_interest[i])))
        axs[i].plot(x, og_wfs[i], linewidth=2, markersize=12, color=colors[i])
        axs[i].get_xaxis().set_visible(False)
    
    # fig.subplots_adjust(wspace=0)

    fig.suptitle(title)
    
    if save_name is not None:
        plt.savefig(save_name)
        
def plot_spike_loc_classes(locs, labels, num_classes, geom, title, save_name=None):
    true_labels = np.array([[i for j in range(300)] for i in range(num_classes)]).flatten()
    print(true_labels.shape)
    cmap = plt.cm.get_cmap('hsv', num_classes)
    colors = np.array([cmap(i) for i in labels])
    true_colors = np.array([cmap(i) for i in true_labels])
#     colors = [cmap(i) for i in range(10)]
#     colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'cyan', 'violet', 'maroon', 'pink']
#     alphas = np.linspace(0.1, 1, num=10)
    alphas = np.ones(num_classes)
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # fig, ax = plt.subplots(4, 6, figsize=(18, 30))
    fig = plt.figure(figsize=(10, 15), constrained_layout=True)
    fig.tight_layout()
    gs = GridSpec(10, 4, figure=fig)
    
    # assigned labels plot
    ax0 = fig.add_subplot(gs[:, :2])
    ax0.scatter(geom[:,1], geom[:,2])
    ax0.scatter(locs[:, 0], locs[:, 1], c=colors, alpha=0.1)
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    ax0.set_title('Predicted clusters from features')
    
    # true labels plot
    ax1 = fig.add_subplot(gs[:, 2:])
    ax1.scatter(geom[:,1], geom[:,2])
    ax1.scatter(locs[:, 0], locs[:, 1], c=true_colors, alpha=0.1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax1.set_title('True clusters')
    
    fig.suptitle(title)
    fig.subplots_adjust(top=0.93)
    # fig.subplots_adjust(wspace=0.12)
    
    fig.subplots_adjust(hspace=0.2)
    
    if save_name is not None:
        plt.savefig(save_name)