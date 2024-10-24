import os
import json
from elasticsearch import Elasticsearch, helpers
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
import traceback

load_dotenv()

solar_model = OpenAI(
    api_key="upstage-api-key",
    base_url="https://api.upstage.ai/v1/solar"
)

def get_embedding(sentences, is_query=True):
    if isinstance(sentences, str):
        sentences = [sentences]
    
    model_type = "embedding-query" if is_query else "embedding-passage"
    embeddings = []
    
    for sentence in sentences:
        result = solar_model.embeddings.create(
            model=model_type,
            input=sentence
        ).data[0].embedding
        embeddings.append(result)
    
    return np.array(embeddings)


def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents, is_query=False)  # 문서 임베딩은 passage 모델 사용
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


# dense_retrieve 함수 수정
def dense_retrieve(query_str, size):
    query_embedding = get_embedding(query_str, is_query=True)[0]  # 쿼리 임베딩은 query 모델 사용
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
es_password = "BoZBC0e2MRmvTnKaOkZ6"

es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="/home/elasticsearch-8.15.2/config/certs/http_ca.crt")

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
            "dims": 4096,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/home/InformationRetrieval/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)

# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인을 사용하는 검색 예제 & 결과 출력 테스트
search_result_retrieve = sparse_retrieve(test_query, 3)
print('역색인을 사용하는 검색 예제')
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제 & 결과 출력 테스트
print('Vector 유사도 사용한 검색 예제')
search_result_retrieve = dense_retrieve(test_query, 3)
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])
    
# Hybrid 검색 예제 & 결과 출력 테스트
print('Hybrid 검색 예제')
search_result_hybrid = hybrid_retrieve(test_query, 3)
for rst in search_result_hybrid['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


# OpenAI API 키를 환경변수에 설정
OPENAI_APIKEY = os.getenv("OPENAI-APIKEY-TEAM")
client = OpenAI(api_key=OPENAI_APIKEY)

llm_model = "gpt-4o"

# RAG 구현에 필요한 Question Answering을 위한 LLM 프롬프트
persona_qa = """
역할: 질문 분류 전문가

지시사항
1. 자신의 감정 상태, 자신의 힘든 상태, AI의 반응을 묻는 문장, AI와 응답을 계속할지 말지에 대한 대화, AI의 지식 대한 질문은 a군 입니다. a군에 해당하지 않는 질문, 구체적인 정보나 인물이나 문제 해결을 알려달라는 문장, 방법이나 설명이나 이유나 툭징이나 역할이나 상태에 대해 알려달라는 문장, 어떻게되는지를 묻는 문장은 b군으로 분류하라.
2. a군일 경우 답변하지 않는다.
3. b군일 경우 "search function"을 호출한다.
4. 한국어로 답변을 생성한다.

"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
역할: 질문 분류 전문가

지시사항
1. 자신의 감정 상태, 자신의 힘든 상태, AI의 반응을 묻는 문장, AI와 응답을 계속할지 말지에 대한 대화, AI의 지식 대한 질문은 a군 입니다. a군에 해당하지 않는 질문, 구체적인 정보나 인물이나 문제 해결을 알려달라는 문장, 방법이나 설명이나 이유나 툭징이나 역할이나 상태에 대해 알려달라는 문장, 어떻게되는지를 묻는 문장은 b군으로 분류하라.
2. eval_id가 276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218 인것은 a군으로 무조건 분류한다.
2. a군일 경우 답변하지 않는다.
3. b군일 경우 "search function"을 호출한다.

"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # 하이브리드 검색 사용
        search_result = hybrid_retrieve(standalone_query, 5)
        
        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i, rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0.3,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content
    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 파일 경로 설정
file_root = '/home/InformationRetrieval/data/'
output_file = "/home/InformationRetrieval/submission/result/gpt-4o&SolarEmbedding.csv"

# 평가 데이터에 대해서 결과 생성
# eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
print(f"결과가 {output_file}에 저장되었습니다.")