from sklearn.manifold import TSNE


def compute_tsne(features, dim=2):
    tsne = TSNE(n_components=dim, perplexity=30, random_state=42)
    xy = tsne.fit_transform(features)
    return xy
