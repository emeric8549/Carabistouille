import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def visualize_embeddings(W1, idx2word, nb_words=20, filename="embeddings.png", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    words = list(idx2word.values())[:nb_words]
    vectors = W1[:nb_words]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=9)
    plt.title("CBOW embeddings visualization (PCA)")
    plt.grid(True)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()