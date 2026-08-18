
from pathlib import Path
import pandas as pd

data = Path("data/test_transaction.csv")


print(data.exists())


df = pd.read_csv(data)
print(df)