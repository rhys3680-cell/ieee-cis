from pathlib import Path
import pandas as pd

def load_merged_data(data_dir="../../data/ieee-fraud-detection"):
    """
    IEEE-CIS 데이터셋(transaction, identity)을 불러와
    TransactionID 기준으로 left join하여 반환하는 함수
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {data_path}")
        
    # 데이터 불러오기
    train = pd.read_csv(data_path / "train_transaction.csv")
    train_id = pd.read_csv(data_path / "train_identity.csv")
    
    # 데이터 병합
    merged_df = pd.merge(train, train_id, on="TransactionID", how="left")
    
    print(f"데이터 결합 완료! Shape: {merged_df.shape}")
    return merged_df