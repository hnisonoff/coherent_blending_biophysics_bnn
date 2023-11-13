#!/usr/bin/env python

import sys
import os
import lightgbm as lgbm
from rdkit import Chem
from rdkit.Chem import Descriptors


class LogDPredictor:
    def __init__(self, model_file_name):
        if not os.path.exists(model_file_name):
            print(f"Error: Could not find model file {model_file_name}", file=sys.stderr)
            sys.exit(0)

        self.mdl = lgbm.Booster(model_file=model_file_name)
        self.descList = self.mdl.feature_name()
        self.fns = [(x, y) for x, y in Descriptors.descList if x in self.descList]

    def predict_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return self.predict(mol)
        else:
            return None

    def predict(self, mol):
        res = []
        for x, y in self.fns:
            res.append(y(mol))
        return self.mdl.predict([res])[0]


def main():
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} infile.smi")
        sys.exit(0)
    logd_predictor = LogDPredictor("model_logd.txt")
    suppl = Chem.SmilesMolSupplier(sys.argv[1], titleLine=False)
    for mol in suppl:
        pred_logd = logd_predictor.predict(mol)
        print(Chem.MolToSmiles(mol), mol.GetProp("_Name"), f"{pred_logd:.2f}")


if __name__ == "__main__":
    main()
