import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns)


class NaNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, value='UNKNOWN'):
        self.columns = columns
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(self.value)
        return X


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.means = X[self.columns].mean()
        return self
    
    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(self.means)
        return X


class MappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.replace(self.mapping)


class InconclusiveReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, replacement='UNKNOWN'):
        self.columns = columns
        self.replacement = replacement
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.columns] = X[self.columns].replace('INCONCLUSIVE', self.replacement)
        return X


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for smiles in tqdm(X['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            feature_dict = {
                'MolWt': Descriptors.MolWt(mol),
                #'MolLogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHDonors': Descriptors.NumHDonors(mol)
            }
            features.append(feature_dict)
        features_df = pd.DataFrame(features)
        return pd.concat([X.reset_index(drop=True), features_df], axis=1)