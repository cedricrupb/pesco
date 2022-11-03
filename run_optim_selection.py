import sys
import argparse

import random
import joblib
import numpy as np

from tqdm import tqdm

from sklearn.decomposition import  PCA
from sklearn.decomposition import  TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline

from pesco.data.utils import load_dataset
from pesco.eval       import load_evaluator_from_file
from pesco.data.utils import Dataset, _index_elements
from pesco.data.utils import stratified_cv

from pesco.models     import SelectorPipeline
from pesco.models     import PAR10BucketSelector, RankingSelector
from pesco.models     import PAR10BucketSelectorConstructor, RankingSelectorConstructor
from pesco.optim      import KMeansBoosting


# Data utils ------------------------------------------------------------------------

def clean_dataset(dataset):
    """Clean the dataset from unsolvable instance"""

    # Identify trivial instances
    nontrivial = dataset.labels.max(axis = 1) !=  0
    nontrivial = set(instance for i, instance in enumerate(dataset.instance_index) if nontrivial[i])
    nontrivial_instances = list(sorted(nontrivial))

    # Filter the dataset
    index = {k: i for i, k in enumerate(dataset.instance_index)}
    data_indices = [index[k] for k in nontrivial_instances]
    return Dataset(*map(lambda x: _index_elements(x, data_indices), dataset[:-1]), dataset.label_index)


# Preprocessor ----------------------------------------------------------------------


def build_preprocessor(config, feature_size):
    
    processing_steps = []

    if not config.no_scale and not config.tfidf:
        processing_steps.append(StandardScaler())
    
    if config.tfidf:
        processing_steps.append(TfidfTransformer())

    if config.pca_dim == -1:
            config.pca_dim = feature_size

    if not config.no_pca and not config.svd:
        processing_steps.append(PCA(n_components = config.pca_dim, whiten = config.whiten))
    
    if config.svd:
        if config.whiten: print("WARNING: TruncatedSVD does not support whitening.")
        processing_steps.append(TruncatedSVD(n_components = config.pca_dim))

    if not config.no_scale and config.tfidf:
        processing_steps.append(StandardScaler())

    if len(processing_steps) == 0:
        class Dummy:
            def fit_transform(self, X):
                return X
            def transform(self, X):
                return X
        return Dummy()
    
    if len(processing_steps) == 1:
        return processing_steps[0]

    return make_pipeline(*processing_steps)
    
# Build model -----------------------------------------------------------------------


def build_model(config, feature_size):
    
    if config.boost == 0:
        if config.rank:
            clf = RankingSelector(C = config.C, weight = config.weight)
        else:
            clf = PAR10BucketSelector(C = config.C, weight = config.weight)

    else:
        if config.rank:
            base_clf = RankingSelectorConstructor
        else:
            base_clf = PAR10BucketSelectorConstructor
        
        clf = KMeansBoosting(base_clf, k = config.boost, add_base = True)

    return clf


def train(config, dataset):
    feature_size = dataset.embedding.shape[1]

    # Preprocess
    preprocessor = build_preprocessor(config, feature_size)
    embeddings   = preprocessor.fit_transform(dataset.embedding)

    # Train model
    model = build_model(config, feature_size)
    model.fit(embeddings, dataset.labels, dataset.runtimes)

    return SelectorPipeline(model, dataset.label_index, preprocessor = preprocessor)


def evaluate(config, evaluator, train_dataset, test_dataset):
    selector       = train(config, train_dataset)
    predictions    = selector.predict(test_dataset.embedding)
    mapping    = {test_dataset.instance_index[i]: tool for i, tool in enumerate(predictions)}

    return evaluator.eval(mapping, timelimit = config.timelimit)


def crossval(config, evaluator, dataset):

    solved = 0
    total  = 0

    for train_dataset, test_dataset in tqdm(stratified_cv(dataset, num_splits = config.cvk), total = config.cvk):
        result = evaluate(config, evaluator, train_dataset, test_dataset)
        solved += result.score()
        total  += test_dataset.embedding.shape[0]

    return solved / total, solved


# Main function ---------------------------------------------------------------------

class TrainConfig:

    REQUIRED = {"feature_path", "label_path"}

    def __init__(self, **kwargs):
        self.feature_path = ""
        self.label_path   = ""
        self.timelimit    = 900

        self.train = False
        self.crossval  = False
        self.eval      = False

        # Model config
        self.no_scale = False 
        self.tfidf    = False
        
        self.svd      = False
        self.no_pca   = False
        self.pca_dim  = -1
        self.whiten   = False

        self.rank     = False
        self.C        = 0.1
        self.weight   = 9_000

        self.boost    = 0

        # Cross validation configs
        self.cvk = 10 # Number of splits for cross validation

        # Train config
        self.model_path = "model.jbl"

        self.seed = 42

        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k): raise AttributeError("Attribute %s is undefined." % k)
            setattr(self, k, v)

    def __repr__(self):
        return "TrainConfig(%s)" % ",".join(f"{k}={v}" for k, v in self.__dict__.items())


def args_to_config(args = None):
    if args is None: args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    dummy  = TrainConfig()
    for k, v in dummy.__dict__.items():
        vtype = type(v)
        required = k in TrainConfig.REQUIRED

        if vtype is bool:
            parser.add_argument(f"--{k}", action="store_true", required = required)
        else:
            parser.add_argument(f"--{k}", type = vtype, default = v, required = required)
    
    args = parser.parse_args(args = args)
    return TrainConfig(**args.__dict__)


def main(args = None):
    config = args_to_config(args)

    # Set the seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    print("Current config:", config)
   
    print("Load dataset...")
    dataset = load_dataset(
        config.feature_path, config.label_path, fill_unknown = True
    )

    if config.timelimit != 900:
        runtime_mask = np.clip(config.timelimit - dataset.runtimes, 0, 1)
        labels  = dataset.labels * runtime_mask.astype(np.int64)
        dataset = Dataset(dataset[0], labels, *dataset[2:])

    dataset = clean_dataset(dataset)

    print(f"Loaded {dataset.embedding.shape[0]} instances.")
    print(f"Embeddings have {dataset.embedding.shape[1]} dimensions.")
    print(f"Ranked verifiers: {dataset.label_index}")

    if config.crossval:
        print("Perform cross validation for current parameter setting...")
        evaluator = load_evaluator_from_file(config.label_path)
        score, plain_score = crossval(config, evaluator, dataset)

        print("Oracle score for current config: %.2f (%d instances)" % (100 * score, plain_score))


    if config.train:
        print("Train model with current parameter setting")
        selector = train(config, dataset)
        print("Finished training...")
        print("Save selector to %s" % config.model_path)
        joblib.dump(selector, config.model_path, compress = 3)
    
    if config.eval:
        print("Load selector from %s" % config.model_path)
        evaluator = load_evaluator_from_file(config.label_path)
        selector = joblib.load(config.model_path)

        predictions    = selector.predict(dataset.embedding)
        mapping    = {dataset.instance_index[i]: tool for i, tool in enumerate(predictions)}
        result     = evaluator.eval(mapping, timelimit = config.timelimit)
        
        solved = result.score()
        total  = len(dataset.instance_index)
        print("Oracle score for current config: %.2f (%d instances)" % (100 * (solved / total), solved))





if __name__ == '__main__':
    main()