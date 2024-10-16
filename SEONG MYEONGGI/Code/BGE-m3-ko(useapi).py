import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
import traceback

load_dotenv()

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("jhgan/ko-sroberta-multitask")
model = SentenceTransformer("dragonkue/BGE-m3-ko")


# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색 (기존과 동일)
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색 (기존과 동일)
def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)


# 새로 추가된 하이브리드 검색 함수
# def hybrid_retrieve(query_str, size, alpha=0.5):
#     sparse_results = sparse_retrieve(query_str, size)
#     dense_results = dense_retrieve(query_str, size)
    
#     combined_results = {}
#     max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits']) if sparse_results['hits']['hits'] else 1
#     max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits']) if dense_results['hits']['hits'] else 1
    
#     for hit in sparse_results['hits']['hits']:
#         doc_id = hit['_id']
#         normalized_sparse_score = hit['_score'] / max_sparse_score
#         combined_results[doc_id] = {'document': hit['_source'], 'score': alpha * normalized_sparse_score}
    
#     for hit in dense_results['hits']['hits']:
#         doc_id = hit['_id']
#         normalized_dense_score = hit['_score'] / max_dense_score
#         if doc_id in combined_results:
#             combined_results[doc_id]['score'] += (1 - alpha) * normalized_dense_score
#         else:
#             combined_results[doc_id] = {'document': hit['_source'], 'score': (1 - alpha) * normalized_dense_score}
    
#     sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
#     hybrid_results = {
#         'hits': {
#             'total': {'value': len(sorted_results)},
#             'hits': [
#                 {
#                     '_id': doc_id,
#                     '_score': info['score'],
#                     '_source': info['document']
#                 } for doc_id, info in sorted_results[:size]
#             ]
#         }
#     }
    
#     return hybrid_results


# 개선된 하이브리드 검색 함수
def hybrid_retrieve(query_str, size, alpha=0.5, beta=0.3):
    sparse_results = sparse_retrieve(query_str, size * 2)  # 더 많은 결과를 가져옵니다
    dense_results = dense_retrieve(query_str, size * 2)
    
    combined_results = {}
    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits']) if sparse_results['hits']['hits'] else 1
    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits']) if dense_results['hits']['hits'] else 1
    
    # TF-IDF 벡터라이저 초기화
    tfidf = TfidfVectorizer()
    
    # 쿼리와 문서들의 텍스트 준비
    all_texts = [query_str] + [hit['_source']['content'] for hit in sparse_results['hits']['hits']] + [hit['_source']['content'] for hit in dense_results['hits']['hits']]
    
    # TF-IDF 벡터 생성
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    # 쿼리 벡터
    query_vector = tfidf_matrix[0]
    
    for i, hit in enumerate(sparse_results['hits']['hits'] + dense_results['hits']['hits']):
        doc_id = hit['_id']
        normalized_score = hit['_score'] / (max_sparse_score if i < len(sparse_results['hits']['hits']) else max_dense_score)
        
        # TF-IDF 유사도 계산
        doc_vector = tfidf_matrix[i + 1]  # +1 because the first vector is the query
        tfidf_similarity = cosine_similarity(query_vector, doc_vector)[0][0]
        
        if doc_id in combined_results:
            combined_results[doc_id]['score'] += (1 - alpha - beta) * normalized_score + beta * tfidf_similarity
        else:
            combined_results[doc_id] = {
                'document': hit['_source'],
                'score': (alpha if i < len(sparse_results['hits']['hits']) else (1 - alpha - beta)) * normalized_score + beta * tfidf_similarity
            }
    
    # 결과 정렬 및 상위 k개 선택
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)[:size]
    
    # 최종 결과 포맷팅
    hybrid_results = {
        'hits': {
            'total': {'value': len(sorted_results)},
            'hits': [
                {
                    '_id': doc_id,
                    '_score': info['score'],
                    '_source': info['document']
                } for doc_id, info in sorted_results
            ]
        }
    }
    
    return hybrid_results

es_username = "elastic"
es_password = "9RBsO+0+0JJCnWaeS2n8"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())


# 색인을 위한 setting 설정
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
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 1024,  # dragonkue/BGE-m3-ko 모델의 출력 차원에 맞춰 1024로 변경
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
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

llm_model = "gpt-3.5-turbo-1106"

# RAG 구현에 필요한 Question Answering을 위한 LLM 프롬프트
persona_qa = """
## 역할: 다학제적 과학 전문가 또는 다분야 지식전문가

## 지침
- 질문의 의도를 정확히 파악하고 관련된 과학 분야를 식별한다.
- 사용자의 메시지 분석하고 알고 싶어하는 정보가 무엇인지 명확하게한다.
- <context> 정보만을 활용하여 간결하게 답변을 생성한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## 역할: 질문 분류 전문가

## 지침
- 사용자의 대화 메시지에 대해서 분석, 분류하고 아래 1,2 중 해당되는 것 하나를 실행한다.
1. 개인적인 감정 표현(예: 힘들다, 즐거웠다, 기분 좋다), 상호작용 및 자기소개 요청(예: 너는 누구야? 너 뭘 잘해?), 대화 종료 요청 또는 일상적인 인사(예: 이제 그만 얘기하자, 안녕 반가워), 정서적 상태 변화(예: 우울한데 신나는 얘기)에 대한 대화 메시지에는 적절한 대답을 생성한다.
2. 정보 탐구 또는 구체적인 질문, 특정 주제에 대한 공부 및 탐구, 이야기나 사실 요청의 대화 메시지에는 search api를 호출한다.
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
output_file = "/home/InformationRetrieval/result/gpt-3.5-turbo&topk5&검색개선.csv"

# 평가 데이터에 대해서 결과 생성
# eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
print(f"결과가 {output_file}에 저장되었습니다.")