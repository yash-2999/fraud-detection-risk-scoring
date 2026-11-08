import pandas as pd
import pytest
from src.fraud.data import FEATURES
@pytest.fixture
def tiny_df():
    rows=[]
    for i in range(100):
        rows.append({"Time":float(i),"Amount":float(i+1),**{f"V{j}":float(i*j/10) for j in range(1,29)},"Class":int(i%10==0)})
    return pd.DataFrame(rows)
