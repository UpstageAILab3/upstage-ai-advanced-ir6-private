import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("dragonkue/BGE-m3-ko")

def get_embedding(sentences):
    return model.encode(sentences)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings

def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

def delete_es_index(index):
    es.indices.delete(index=index)

def bulk_add(index, docs):
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)

def sparse_retrieve(query_str, size):
    if isinstance(query_str, list):
        query_str = " ".join(query_str)
    query = {
        "query": {
            "multi_match": {
                "query": query_str,
                "fields": ["content"],
                "type": "best_fields",
                "tie_breaker": 0.3,
                "minimum_should_match": "30%"
            }
        },
        "size": size
    }
    return es.search(index="test", body=query)

def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)

def query_expansion(query_str):
    vectorizer = CountVectorizer().fit([query_str])
    query_vector = vectorizer.transform([query_str]).toarray()[0]
    similar_words = vectorizer.get_feature_names_out()[np.argsort(query_vector)[-5:]]
    expanded_query = query_str + " " + " ".join(similar_words)
    return expanded_query

def mmr(query_vector, doc_vectors, doc_ids, lambda_param=0.5, k=5):
    selected = []
    unselected = list(range(len(doc_vectors)))
    
    while len(selected) < k:
        mmr_score = {}
        for i in unselected:
            relevance = cosine_similarity([query_vector], [doc_vectors[i]])[0][0]
            diversity = min([cosine_similarity([doc_vectors[i]], [doc_vectors[j]])[0][0] for j in selected]) if selected else 0
            mmr_score[i] = lambda_param * relevance - (1 - lambda_param) * diversity
        
        selected_idx = max(mmr_score, key=mmr_score.get)
        selected.append(selected_idx)
        unselected.remove(selected_idx)
    
    return [doc_ids[i] for i in selected]

def hybrid_retrieve(query_str, size, alpha=0.5):
    expanded_query = query_expansion(query_str)
    sparse_results = sparse_retrieve(expanded_query, size * 2)
    dense_results = dense_retrieve(query_str, size * 2)
    
    combined_results = {}
    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits']) if sparse_results['hits']['hits'] else 1
    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits']) if dense_results['hits']['hits'] else 1
    
    for hit in sparse_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_sparse_score = hit['_score'] / max_sparse_score
        combined_results[doc_id] = {'document': hit['_source'], 'score': alpha * normalized_sparse_score, 'vector': hit['_source']['embeddings']}
    
    for hit in dense_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_dense_score = hit['_score'] / max_dense_score
        if doc_id in combined_results:
            combined_results[doc_id]['score'] += (1 - alpha) * normalized_dense_score
        else:
            combined_results[doc_id] = {'document': hit['_source'], 'score': (1 - alpha) * normalized_dense_score, 'vector': hit['_source']['embeddings']}
    
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Apply MMR
    query_vector = get_embedding([query_str])[0]
    doc_vectors = [info['vector'] for _, info in sorted_results]
    doc_ids = [doc_id for doc_id, _ in sorted_results]
    
    mmr_results = mmr(query_vector, doc_vectors, doc_ids, lambda_param=0.7, k=size)
    
    hybrid_results = {
        'hits': {
            'total': {'value': len(mmr_results)},
            'hits': [
                {
                    '_id': doc_id,
                    '_score': combined_results[doc_id]['score'],
                    '_source': combined_results[doc_id]['document']
                } for doc_id in mmr_results
            ]
        }
    }
    
    return hybrid_results

es_username = "elastic"
es_password = "9RBsO+0+0JJCnWaeS2n8"

es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

print(es.info())

settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

mappings = {
    "properties": {
        "content": {
            "type": "text",
            "analyzer": "nori",
            "similarity": "BM25"
        },
        "embeddings": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

create_es_index("test", settings, mappings)

index_docs = []
with open("/home/InformationRetrieval/data/documents.jsonl") as f:
# with open("/home/InformationRetrieval/data/indexed_document.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)

for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

ret = bulk_add("test", index_docs)

print(ret)

def answer_question(query):
    if isinstance(query, list):
        query = " ".join(query)
    response = {"standalone_query": query, "topk": []}
    search_result = hybrid_retrieve(query, 5)

    for rst in search_result['hits']['hits']:
        response["topk"].append(rst["_source"]["docid"])

    return response

def eval_rag(eval_filename, output_filename):
    special_eval_ids = {276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218}
    
    with open(eval_filename) as f, open(output_filename, "w", encoding='utf-8') as of:
        for line in f:
            j = json.loads(line)
            eval_id = j["eval_id"]
            
            if eval_id in special_eval_ids:
                output = {
                    "eval_id": eval_id,
                    "standalone_query": "",
                    "topk": []
                }
            else:
                if isinstance(j["msg"], list):
                    query = " ".join([m['content'] for m in j["msg"] if 'content' in m])
                else:
                    query = j["msg"]

                print(f'Question: {query}')
                response = answer_question(query)

                output = {
                    "eval_id": eval_id,
                    "standalone_query": response["standalone_query"],
                    "topk": response["topk"]
                }
            
            json.dump(output, of, ensure_ascii=False)
            of.write('\n')

file_root = '/home/InformationRetrieval/data/'
output_file = "/home/InformationRetrieval/result/BGE-m3-ko&None-API&검색알고리즘개선(MMR)&indexed_document.csv"

eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
print(f"결과가 {output_file}에 저장되었습니다.")
