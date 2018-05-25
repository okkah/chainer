import matplotlib.pyplot as plt


def output(train, file):
    fig,ax = plt.subplots(nrows=9,ncols=12,sharex=True,sharey=True)
    ax = ax.flatten()
    for i in range(108):
        img = train[i][0]
        ax[i].imshow(img[0], cmap='gray', interpolation='none')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(file)
