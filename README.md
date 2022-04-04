# Recommendation_System_paper_Code

추천시스템 모델 활용 실습 코드


## File Directory 📂

```shell
Recommendation_System_paper_Code
├── 1. data
│   └── movies / ratings.csv, movies_metadata.csv   # 영화, 랭킹 데이터  
│
└── 2. model
    └── Recommendation_Code.ipynb     # Recommendation Models

```

## 실습 모델

### DeepFM
데이터 : [competition website](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) and put them here.
To train DeepFM model for this dataset, run
```
$ cd example
$ python main.py
```