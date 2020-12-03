import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import os
from zipfile import ZipFile


dfs = [pd.read_csv(x) for x in sys.argv[1:-1]]
output = sys.argv[-1]

for i in range(len(dfs)):
    dfs[i]["prediction"] = rankdata(dfs[i]["prediction"], method="min")

ranked_sum = np.mean([list(dfs[x]["prediction"]) for x in range(len(dfs))], axis=0)

dfs[0]["prediction"] = ranked_sum

dfs[0]["prediction"] = rankdata(dfs[0]["prediction"], method="min")
dfs[0].to_csv(output, index=False)

with ZipFile(os.path.splitext(output)[0] + ".zip", "w") as myzip:
    myzip.write(output, arcname=os.path.basename(output))
