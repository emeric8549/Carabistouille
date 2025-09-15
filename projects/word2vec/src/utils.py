import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random
import numpy as np

def visualize_embeddings(W1, idx2word, nb_words=20, filename="plots/embeddings.png"):
    os.makedirs("plots", exist_ok=True)

    words_indices = random.sample(range(1, W1.shape[0]), nb_words)
    words = np.array(list(idx2word.values()))[words_indices]
    vectors = W1[words_indices]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.001, y + 0.001, word, fontsize=9)
    plt.title("CBOW embeddings visualization (PCA)")
    plt.grid(True)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()