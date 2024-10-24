import json
import pandas as pd
from collections import defaultdict

# 파일 경로와 각 파일의 가중치 설정
file_weights = {
    "/home/InformationRetrieval/submission/voting/hardvoting/hardvoting(명기솔라&리랭킹+동건님베스트모델).csv": 2,
    "/home/InformationRetrieval/submission/result/gpt-4o&hyde_merge.csv": 1,
    "/home/InformationRetrieval/submission/result/gpt-4o&SolarEmbedding.csv": 1.5,
}

# 모든 eval_id를 저장할 set
all_eval_ids = set()

# 데이터와 점수를 저장할 딕셔너리 초기화
eval_data = defaultdict(lambda: defaultdict(float))

# 각 파일에서 데이터를 읽고 가중치를 적용하여 통합
for path, weight in file_weights.items():
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            eval_id = entry["eval_id"]
            all_eval_ids.add(eval_id)  # 모든 eval_id 수집
            
            # topk가 있는 경우에만 점수 계산
            if "topk" in entry and entry["topk"]:
                for rank, item in enumerate(entry["topk"]):
                    # 순위가 높을수록 높은 점수 부여 (예: 1위=3점, 2위=2점, 3위=1점)
                    rank_score = (3 - rank) * weight
                    eval_data[eval_id][item] += rank_score

# 소프트보팅 결과 생성
soft_voting_results = []

# 모든 eval_id에 대해 처리
for eval_id in sorted(all_eval_ids):
    # 해당 eval_id에 대한 점수가 있는 경우
    if eval_data[eval_id]:
        # 점수를 기준으로 정렬하여 상위 3개 항목 선택
        topk_voted = sorted(eval_data[eval_id].items(), key=lambda x: (-x[1], x[0]))[:3]
        topk_list = [item for item, _ in topk_voted]
    else:
        # topk가 없는 경우 빈 리스트 유지
        topk_list = []
    
    result = {
        "eval_id": eval_id,
        "standalone_query": "",  # 필요에 따라 수정 가능
        "topk": topk_list
    }
    soft_voting_results.append(result)

# 결과를 JSON 라인으로 저장
output_path = "/home/InformationRetrieval/submission/voting/softvoting/weighted_리랭킹&SolarEmbedding&softvoting.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for result in soft_voting_results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Weighted soft voting 결과가 {output_path}에 저장되었습니다.")