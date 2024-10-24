import json
import pandas as pd
from collections import defaultdict, Counter

# 파일 경로
file_paths = [
    "/home/InformationRetrieval/submission/voting/hardvoting/hardvoting(명기솔라&리랭킹+동건님베스트모델).csv",
    # "/home/InformationRetrieval/submission/result/gpt-4o&hyde_merge.csv",
    "/home/InformationRetrieval/submission/result/gpt-4o&SolarEmbedding.csv",
    # "/home/InformationRetrieval/submission/voting/hardvoting/hardvoting(voting&T7).csv",
    # "/home/InformationRetrieval/submission/result/유정수(gemini).csv",
]

# 데이터를 저장할 딕셔너리 초기화
eval_data = defaultdict(list)

# 각 파일에서 데이터를 읽고 통합
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            eval_id = entry["eval_id"]
            eval_data[eval_id].extend(entry["topk"])

# 소프트보팅 결과 생성
soft_voting_results = []

for eval_id, topk_items in eval_data.items():
    # 가장 많이 등장한 topk 항목을 빈도 순으로 정렬하고 3개로 제한
    topk_voted = [item for item, _ in Counter(topk_items).most_common(3)]
    result = {
        "eval_id": eval_id,
        "standalone_query": "",  # 필요에 따라 수정 가능
        "topk": topk_voted
    }
    soft_voting_results.append(result)

# 결과를 JSON 라인으로 저장
output_path = "/home/InformationRetrieval/submission/voting/softvoting/리랭킹&SolarEmbedding.jsonl"
# output_path = "/home/InformationRetrieval/result/voting_result/soft_voting_result.csv"
with open(output_path, 'w', encoding='utf-8') as f:
    for result in soft_voting_results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Soft voting 결과가 {output_path}에 저장되었습니다.")
