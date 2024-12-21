from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def plot_probabilities(
    words: list[str],
    word_probs: list[float],
    renormalize: bool = False,
    title: str = "Top-k words with their probabilities",
) -> None:
    """
    Small utility function for creating a bar chart that plots the probabilities.
    :param words: The words in string format.
    :param word_probs: The probabilities for each word.
    :param renormalize: # If true, it shows the probabilities as to how they will be sampled. If false, it shows the original probabilities.
    """
    if renormalize:
        # rescale the probs to ensure sum(word_probs) == 1.0
        word_probs = normalize_probs(word_probs)
    plt.figure(figsize=(12, 6))
    plt.barh(words, word_probs)
    plt.xlabel("Probability")
    plt.ylabel("Word")
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest probability at the top
    plt.show()


def normalize_probs(probs: list[float]) -> list[float]:
    """
    Normalize a list of probabilities to sum to 1.
    :param probs: The list of probabilities.
    :return: The normalized list of probabilities that sum to 1.
    """
    assert len(probs) > 0, "The list of probabilities should not be empty."
    assert all(
        [prob >= 0 for prob in probs]
    ), "The list of probabilities should only contain positive values."
    return [prob / sum(probs) for prob in probs]


def plot_embeddings_interactively(
    embeddings: np.ndarray,
    titles: np.ndarray,
    plot_title: str,
    n_cluster: int | None = None,
):
    """
    Plot the embeddings interactively using plotly.
    :param embeddings: The embeddings matrix of shape (n_samples, n_features).
    :param titles: The title to show when hovering over a point of shape (n_samples,).
    :param plot_title: The title of the plot.
    :param n_cluster: The number of cluster/different colors to use in the plot.
    """

    # First, we cluster the embeddings using k-means such that can show some groups in the plot.
    kmeans = KMeans(n_clusters=n_cluster, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_
    values_per_cluster = Counter(clusters)
    # We use the average number of values per cluster as perplexity for t-SNE.
    avg_per_cluster = sum(values_per_cluster.values()) / len(values_per_cluster)

    # We transform the embeddings to 2D using t-SNE.
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=avg_per_cluster, metric="cosine"
    )
    vis_dims = tsne.fit_transform(embeddings)
    x = vis_dims[:, 0]
    y = vis_dims[:, 1]

    # Here we create a dataframe that will be plotted using plotly.
    df = pd.DataFrame(
        {
            "title": titles,
            "cluster": [
                f"Cluster {cluster_idx} with {values_per_cluster[cluster_idx]} values"
                for cluster_idx in clusters
            ],
            "x": x,
            "y": y,
        }
    )
    df.sort_values(by="cluster", inplace=True)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_name="title",
        hover_data=["cluster"],
        title=plot_title,
    )
    fig.show()


def plot_similarity_head_map(
    similarities: np.ndarray, titles: np.ndarray, plot_title: str
):
    """
    Plot the similarity matrix as a head map.
    :param similarities: The similarity matrix of shape (n_samples, n_samples).
    :param titles: The title to show when hovering over a point of shape (n_samples,).
    :param plot_title: The title of the plot.
    """
    assert (
        similarities.shape[0] == similarities.shape[1] == len(titles)
    ), f"The similarity matrix should be square and have the same length as the titles but got {similarities.shape} and {len(titles)}"

    # Convert the similarity matrix into a DataFrame
    similarity_df = pd.DataFrame(similarities, index=titles, columns=titles)

    # Use Seaborn to create a heatmap
    plt.figure(figsize=(10, 8))  # Increase the size as per your requirement
    sns.heatmap(
        similarity_df, annot=True, cmap="coolwarm"
    )  # Here, annot=True will display the similarity scores
    plt.title(plot_title)

    # Rotate x-labels with 45 degrees
    plt.xticks(rotation=45, ha="right")
    plt.show()
