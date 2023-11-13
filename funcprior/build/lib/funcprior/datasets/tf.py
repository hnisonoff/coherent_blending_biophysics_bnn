import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *

def get_8mer_data(fn='ARX_L343Q_R1_8mers.txt', y_col="Median"):
    basefn = Path('/home/hunter_nisonoff/projects/uncertainty/protein_dna/datasets')
    df = pd.read_table(basefn / fn)
    assert(y_col in df.columns)
    data = df.loc[:, ["8-mer", y_col]].to_numpy()
    X, y = data[:, 0], data[:, 1]
    X = pd.DataFrame([list(x) for x in X])
    X = pd.get_dummies(X).to_numpy()
    y = np.asarray(y, dtype=np.float)
    return X, y
