## Baseline code를 실행하기 위해 2가지 작업이 필요합니다.

1. install_elasticsearch.sh 실행 후 마지막에 'y' 입력하면 변경된 elasticsearch password가 출력됩니다.
이 password를 통해 elasticsearch client를 사용합니다.
이를 위해 rag_with_elasticsearch.py(85번째 라인)의 es_password를 교체해야 합니다.

2. OpenAI API를 사용하기 위해서는 API 키가 필요합니다.
OpenAI 사이트를 통해 발급받은 API 키를 rag_with_elasticsearch.py(169번째 라인)의 os.environ["OPENAI_API_KEY"]에 넣어 주어야 합니다.

## 코드 실행
```python
pip install requirments.txt
```
```
python ./파일명.py
```

## Code에 대한 설명
- `baselinecode.py`는 기존 대회 주최측에서 제공된 베이스코드입니다.
- `main.py`는 성능이 가장 좋은 코드를 업데이트합니다.
- `KOE5.py`는 nlpai-lab/KoE5 임베딩 모델을 사용한 코드입니다.