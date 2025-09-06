# Introduction
RSML(Relational Symmetric Metric Learning)은 추천 시스템을 위한 발전된 메트릭 러닝 모델입니다. 이 모델은 SML(Symmetric Metric Learning)의 안정적인 대칭적 학습 구조를 기반으로 하며, 여기에 각 사용자-아이템 쌍의 고유한 관계를 동적으로 학습하는 RelationModule을 도입합니다. 이 모듈은 상호작용별 관계 벡터(r_uv)를 생성하여, 정적인 유클리드 거리를 개인화된 '관계적 거리'로 변환합니다. 이를 통해 모델은 사용자의 복잡하고 미묘한 선호도를 더 잘 포착하여 추천 성능을 향상시킵니다.

# Running
python main_rsml.py --hidden_size 256 --lr 1.0e-05 --lamda 0.5 --gama 0.1
# Requirements
본 프로젝트는 아래의 라이브러리들을 필요로 합니다.

Python 3.8+

PyTorch 1.12+

PyTorch Lightning 1.7+

NumPy

Pandas

# Dataset
이 코드는 Netflix Prize 데이터셋을 기준으로 작성되었습니다. dataset/ 폴더에 데이터 파일을 위치시켜야 합니다. utils.py의 preprocessing 함수가 이 파일들을 자동으로 파싱합니다.
