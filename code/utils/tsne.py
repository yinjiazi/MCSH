# from tsnecuda import TSNE
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
# from sklearn.manifold import TSNE
# digits = load_digits()
# print(digits.data.shape)
# print(digits.target)

def plot_tsne(feature, labels):
    #[bs * seq , dim]
    feature = torch.cat(feature).cpu().detach().numpy()
    labels = torch.cat(labels).cpu().detach().numpy()
    # print(feature.shape)
    # print(labels.shape)
    # print(labels)
    embeddings = TSNE(n_jobs=20).fit_transform(feature)
    # print(embeddings.shape)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.scatter(vis_x, vis_y, c = labels, cmap='plasma', marker='.')
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    plt.show()
    # plt.savefig('')
