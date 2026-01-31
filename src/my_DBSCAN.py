import os

import matplotlib.pyplot as plt

import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import DBSCAN

current_dir = Path(__file__).parent
CACHE_FILE = current_dir.parent / 'data' / 'embeddings_cache.npy'


def make_embeddings(techList):

    if os.path.exists(CACHE_FILE):
        print("Loading embeddings from cache")
        embeddings = np.load(CACHE_FILE)
        if len(embeddings) == len(techList):
            return embeddings
        else:
            print("Cache not up to date")


    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(techList)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(CACHE_FILE, embeddings)
    return embeddings


def investigate_k_option(embeddings,k_values=[10]):
    X = np.array(embeddings)
    plt.figure(figsize=(12,8))


    for k in k_values:
        neighs = NearestNeighbors(n_neighbors=k+1,metric='cosine',algorithm='brute').fit(X)

        distances, indexes = neighs.kneighbors(X)

        k_neigh_distances = distances[:,k]

        k_neigh_distances = np.sort(k_neigh_distances)

        plt.plot(k_neigh_distances, label=f'k={k}')

    plt.ylim(0, 1)
    plt.title('Wykres K-Distance (Szukanie "kolanka")')
    plt.ylabel('Odległość Cosinusowa (Cosine Distance)')
    plt.xlabel('Punkty posortowane wg odległości')
    plt.grid(True)
    plt.legend()
    plt.savefig('test_wykresu.png')
    print("Wykres zapisany jako test_wykresu.png - sprawdź folder!")

def print_sorted_clusters(sorted_clusters):
    print(f"{'LIDER':<60} | {'SIŁA (CTR)':<10} | {'ELEMENTY'}")
    print("-" * 60)

    for leader, data in sorted_clusters:
        techs_str = ", ".join(data['techs'])

        print(f"{leader:<40} | {data['ctr']:<10.2f} | {techs_str}")

def normalizeDBSCAN(tech_data,embeddings,eps=0.65,k=4):
    clustering = DBSCAN(eps=eps,min_samples=k).fit(embeddings)

    techList = list(tech_data.keys())
    tech_array = np.array(techList)

    cluster_dict = {}

    unique_labels = set(clustering.labels_)

    for clusterID in unique_labels:
        mask = (clustering.labels_ == clusterID)
        cluster_dict[clusterID] = tech_array[mask].tolist()


    final_dict = {}

    for cluster_id, cluster_elms in cluster_dict.items():
        #We will ahndle noise later
        if cluster_id == -1:
            continue

        cluster_strength = sum(tech_data[el] for el in cluster_elms)

        name_candid = max(cluster_elms, key=lambda x: tech_data[x])

        final_dict[name_candid] = {
            'ctr': cluster_strength,
            'techs': cluster_elms
        }
    #Handle Noise
    for tech in cluster_dict.get(-1, []):
        tech_strength = tech_data[tech]

        final_dict[tech] = {
            'ctr': tech_strength,
            'techs': [tech]
        }

    sorted_clusters = sorted(final_dict.items(), key=lambda x: x[1]['ctr'], reverse=True)
    print_sorted_clusters(sorted_clusters)
    lookup_table = {}

    for leader_name, data in final_dict.items():
        for original_name in data['techs']:
            lookup_table[original_name] = leader_name

    print(f'Liczba wpisów w techdata: {len(tech_data)}')
    print(f"Liczba wpisów w lookupTable: {len(lookup_table)}")

    return lookup_table