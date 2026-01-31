import numpy as np
from sklearn.cluster import HDBSCAN


def print_sorted_clusters(sorted_clusters):
    # Nagłówek z ładnym formatowaniem
    print(f"{'LIDER':<40} | {'SIŁA (CTR)':<10} | {'ELEMENTY'}")
    print("-" * 100) # Wydłużyłem linię dla czytelności

    for leader, data in sorted_clusters:
        techs_str = ", ".join(data['techs'])
        if len(techs_str) > 60:
            techs_str = techs_str[:57] + "..."

        print(f"{leader:<40} | {data['ctr']:<10.2f} | {techs_str}")

def normalizeHDBSCAN(tech_data, embeddings, min_cluster_size=2, min_samples=3):

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )

    labels = clusterer.fit_predict(embeddings)

    techList = list(tech_data.keys())
    tech_array = np.array(techList)

    cluster_dict = {}
    unique_labels = set(labels)

    for clusterID in unique_labels:
        mask = (labels == clusterID)
        cluster_dict[clusterID] = tech_array[mask].tolist()

    final_dict = {}

    for cluster_id, cluster_elms in cluster_dict.items():
        if cluster_id == -1:
            continue

        cluster_strength = sum(tech_data[el] for el in cluster_elms)

        name_candid = max(cluster_elms, key=lambda x: tech_data[x])

        final_dict[name_candid] = {
            'ctr': cluster_strength,
            'techs': cluster_elms
        }

    for tech in cluster_dict.get(-1, []):
        tech_strength = tech_data[tech]

        # osobny wpis
        final_dict[tech] = {
            'ctr': tech_strength,
            'techs': [tech]
        }

    sorted_clusters = sorted(final_dict.items(), key=lambda x: x[1]['ctr'], reverse=True)
    print_sorted_clusters(sorted_clusters)

    # 6. Lookup Table
    lookup_table = {}
    for leader_name, data in final_dict.items():
        for original_name in data['techs']:
            lookup_table[original_name] = leader_name

    print("-" * 60)
    print(f'Liczba wpisów wejściowych (tech_data): {len(tech_data)}')
    print(f"Liczba zmapowanych wpisów (lookupTable): {len(lookup_table)}")
    print(f"Liczba unikalnych technologii po klastrowaniu: {len(final_dict)}")

    return lookup_table