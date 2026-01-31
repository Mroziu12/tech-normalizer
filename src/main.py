import json
import logging

from pathlib import Path


from my_DBSCAN import normalizeDBSCAN, investigate_k_option, make_embeddings
from my_HDBSCAN import normalizeHDBSCAN
from src.aglo_spec import trace_specific_cluster

# Konfiguracja logowania - wygląda profesjonalnie
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent
INPUT_DATA = current_dir.parent / 'data' / 'temp_normalizer_input.json'
OUTPUT_LOOKUP_DBSCAN = current_dir.parent / 'data' / 'output_lookup_dbscan.json'
OUTPUT_LOOKUP_HDBSCAN = current_dir.parent / 'data' / 'output_lookup_hdbscan.json'


def handleDBSCAN(data):
    techList = list(data.keys())
    embeddings = make_embeddings(techList)

    #investigate_k_option(embeddings=embeddings,k_values=range(2,15))
    lookup_tableDBSCAN = normalizeDBSCAN(data,embeddings,0.65,3)

    #print(lookup_tableDBSCAN)


    with open(OUTPUT_LOOKUP_DBSCAN, 'w', encoding='utf-8') as f:
        json.dump(lookup_tableDBSCAN, f, ensure_ascii=False, indent=4)

    print("Zapisano do output_lookup_dbscan.json")

def handleHDBSCAN(data):
    techList = list(data.keys())
    embeddings = make_embeddings(techList)

    lookup_tableHDBSCAN = normalizeHDBSCAN(tech_data=data,embeddings=embeddings,min_cluster_size=2,min_samples=3)

    with open(OUTPUT_LOOKUP_HDBSCAN, 'w', encoding='utf-8') as f:
        json.dump(lookup_tableHDBSCAN, f, ensure_ascii=False, indent=4)

    print("Zapisano do output_lookup_hdbscan.json")

def handleAglomerative(data):
    techList = list(data.keys())
    embeddings = make_embeddings(techList)
    #Specify tech and treshold
    trace_specific_cluster("Python", data, embeddings, threshold=1.0)

def main():
    with open(INPUT_DATA, 'r',encoding='utf-8') as f:
        data = json.load(f)

    #handleAglomerative(data)
    #handleDBSCAN(data)
    #handleHDBSCAN(data)



if __name__ == "__main__":
    main()
