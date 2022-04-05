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

## 1. DeepFM
### Datasets
[competition website](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) and put them here.

### Model Training
To train DeepFM model for this dataset, run
```
$ cd example
$ python main.py
```

## 2. Deep-AutoEncoder-FM
### Datasets

The current version support only the MovieLens ml-1m.zip dataset obtained from https://grouplens.org/datasets/movielens/. 

### Model Training
- Devide the ```ratings.dat``` file from ml-1m.zip into training and testing datasets ```train.dat``` and ```test.dat```. by using the command 

       python src\data\train_test_split.py 
          
- Use shell to make TF_Record files out of the both ```train.dat``` and ```test.dat``` files by executing the command: 

       python src\data\tf_record_writer.py 
      
- Use shell to start the training by executing the command (optionally parse your hyperparameters):

        python training.py 