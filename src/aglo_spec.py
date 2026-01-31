import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

OUTPUT_DIR = current_dir.parent / 'img'

def trace_specific_cluster(target_word, tech_data, embeddings, threshold=1.0):
    techList = list(tech_data.keys())
    try:
        target_idx = techList.index(target_word)
    except ValueError:
        print(f"Error: Word '{target_word}' not found!")
        return

    print(f"Word: {target_word}")

    linked = linkage(embeddings, method='ward')
    labels = fcluster(linked, t=threshold, criterion='distance')
    target_cluster_id = labels[target_idx]

    df = pd.DataFrame({'Skill': techList, 'Cluster_ID': labels, 'Original_Index': range(len(techList))})
    cluster_members = df[df['Cluster_ID'] == target_cluster_id]


    subset_indices = cluster_members['Original_Index'].tolist()
    subset_embeddings = embeddings[subset_indices]
    subset_labels = cluster_members['Skill'].tolist()
    subset_linked = linkage(subset_embeddings, method='ward')

    plt.figure(figsize=(10, 6))
    plt.title(f"Cluster Lineage: {target_word}\n(Threshold: {threshold})")

    dendrogram(
        subset_linked,
        orientation='right',
        labels=subset_labels,
        distance_sort='descending',
        show_leaf_counts=True
    )

    plt.xlabel('Distance')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    threshold_dir = OUTPUT_DIR / f"t_{threshold}"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    safe_name = target_word.replace('/', '_').replace('\\', '_').replace(':', '')
    save_path = threshold_dir / f"aglo_{safe_name}.png"
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)
    plt.close()
