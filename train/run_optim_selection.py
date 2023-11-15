import os
import sys

lib_dir = os.path.join(os.path.dirname(__file__), "..", "lib", "python")

sys.dont_write_bytecode = True  # prevent creation of .pyc files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pesco")) # Add PeSCo also
sys.path.insert(0, lib_dir)

if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] += os.pathsep + str(
    lib_dir
)  # necessary so subprocesses also use libraries


import argparse

import random
import json
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
from pesco.data.utils import Dataset, _index_elements, _path_to_task_id
from pesco.data.utils import stratified_cv

from pesco.models     import SelectorPipeline
from pesco.models     import PAR10BucketSelector, RankingSelector
from pesco.models     import PAR10BucketSelectorConstructor, RankingSelectorConstructor
from pesco.models     import RerankSelection
from pesco.models     import VerifierSubsetSelection
from pesco.models     import EnsembleModel
from pesco.optim      import KMeansBoosting, PortfolioBoosting, StaticBoosting


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


def filter_dataset(config, dataset):
    with open(config.index, "r") as i:
        index = set([_path_to_task_id(c.strip()) for c in i])
    
    common_instances = set(instance for instance in dataset.instance_index if instance in index)
    print("Can index", len(common_instances), "/", len(index), "elements")

    common_instances = list(sorted(common_instances))

    # Filter the dataset
    index = {k: i for i, k in enumerate(dataset.instance_index)}
    data_indices = [index[k] for k in common_instances]
    return Dataset(*map(lambda x: _index_elements(x, data_indices), dataset[:-1]), dataset.label_index)



# Preprocessor ----------------------------------------------------------------------


def build_preprocessor(config, feature_size):

    if os.path.exists(config.preprocessor_path):
        return joblib.load(config.preprocessor_path)
    
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
        preprocessor = processing_steps[0]
    else:
        preprocessor = make_pipeline(*processing_steps)
    
    if len(config.preprocessor_path) > 0:
        joblib.dump(preprocessor, config.preprocessor_path)

    return preprocessor


def feature_union(config, instance_index, embedding = None):

    for feature_path in config.feature_union:

        print("Load other features %s..." % feature_path)
        dataset = load_dataset(
            feature_path, config.label_path, fill_unknown = True
        )

        print(f"Loaded {dataset.embedding.shape[0]} additional features")

        num_features = dataset.embedding.shape[1]
        add_features = []
        new_instance_index = {ix: i for i, ix in enumerate(dataset.instance_index)}

        for instance in instance_index:
            if instance in new_instance_index:
                add_features.append(dataset.embedding[new_instance_index[instance]])
            else:
                print("No features for", instance)
                add_features.append(np.zeros((num_features,)))
        
        add_features = np.stack(add_features)

        # Transform for all
        if add_features.max() > 1000:
            add_features = np.floor(np.log10(1 + add_features)) # TODO: REMOVE

        if embedding is not None:
            embedding = np.concatenate((embedding, add_features), axis = 1)
        else:
            embedding = add_features

    return embedding

    
# Build model -----------------------------------------------------------------------


def build_model(config, feature_size):
    
    if config.boost == 0:
        if config.rank:
            clf = RankingSelector(C = config.C, weight = config.weight)
        elif config.xgb:
            raise NotImplementedError("This option is currently not supported")
        else:
            clf = PAR10BucketSelector(C = config.C, weight = config.weight)

    else:
        if config.rank:
            base_clf = RankingSelectorConstructor
        else:
            base_clf = PAR10BucketSelectorConstructor

        if config.boost_type == "portfolio":
            clf = PortfolioBoosting(base_clf, n_solver = config.boost, weight = 9000, add_base = True, max_runtime = config.timelimit)
        elif os.path.exists(config.boost_type):
            boost_from = joblib.load(config.boost_type)
            assert hasattr(boost_from, "_tools"), "This type of boosting is not supported"
            clf = StaticBoosting(base_clf, boost_from._tools)
        else:
            clf = KMeansBoosting(base_clf, k = config.boost, add_base = True, max_runtime = config.timelimit)

    if len(config.ensemble) > 0:
        ensemble_model = joblib.load(config.ensemble)
        ensemble_model = ensemble_model.model
        clf = EnsembleModel(clf, ensemble_model)

    if config.subset_selection != 0:
        clf = VerifierSubsetSelection(clf, subset_size = config.subset_selection)

    if config.rerank:
        clf = RerankSelection(clf)

    return clf


def _compute_sample_weight(dataset):
    per_category_count = {}
    for instance in dataset.instance_index:
        category, _ = instance.split("/")
        per_category_count[category] = per_category_count.get(category, 0) + 1
    
    sample_weight = []
    for instance in dataset.instance_index:
        category, _ = instance.split("/")
        sample_weight.append(per_category_count[category])
    
    return sum(per_category_count.values()) / np.array(sample_weight)


def train(config, dataset):
    feature_size = dataset.embedding.shape[1]

    # Preprocess
    embeddings = dataset.embedding

    if len(config.feature_union) > 0:
        embeddings = feature_union(config, dataset.instance_index, embeddings)

    preprocessor = build_preprocessor(config, feature_size)
    embeddings   = preprocessor.fit_transform(embeddings)

    # Train model
    model = build_model(config, feature_size)

    sample_weights = None
    if config.norm_samples:
        sample_weights = _compute_sample_weight(dataset)

    model.fit(embeddings, dataset.labels, dataset.runtimes, sample_weight = sample_weights)

    return SelectorPipeline(model, dataset.label_index, preprocessor = preprocessor)


def evaluate(config, evaluator, train_dataset, test_dataset):
    selector       = train(config, train_dataset)

    embeddings = test_dataset.embedding
    if len(config.feature_union) > 0:
        embeddings = feature_union(config, test_dataset.instance_index, embeddings)

    predictions    = selector.predict(embeddings)
    mapping    = {test_dataset.instance_index[i]: tool for i, tool in enumerate(predictions)}

    return evaluator.eval(mapping, timelimit = config.timelimit)


def crossval(config, evaluator, dataset):

    solved = 0
    total  = 0

    all_results = []

    for train_dataset, test_dataset in tqdm(stratified_cv(dataset, num_splits = config.cvk), total = config.cvk):
        result = evaluate(config, evaluator, train_dataset, test_dataset)
        solved += result.score()
        total  += test_dataset.embedding.shape[0]

        all_results.append(result)

    with open(config.eval_path, "w") as o:
        for result in all_results:
            for entry in result:
                o.write(json.dumps({"name": entry[0], "run": entry[4], "result": entry[1]})+"\n")

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
        self.xgb      = False
        self.rerank   = False
        self.C        = 0.1
        self.weight   = 9_000
        self.norm_samples = False

        self.boost       = 0
        self.boost_type  = "kmeans"
        self.subset_selection   = 0

        # Cross validation configs
        self.cvk = 10 # Number of splits for cross validation

        # Train config
        self.eval_path = "eval_results.jsonl"
        self.model_path = "model.jbl"
        self.preprocessor_path = ""
        self.feature_union = []
        self.ensemble = ""

        self.index = ""
        self.seed = 42

        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k): raise AttributeError("Attribute %s is undefined." % k)
            setattr(self, k, v)

    def __repr__(self):
        return "TrainConfig(%s)" % ",".join(f"{k}={v}" for k, v in self.__dict__.items())


def args_to_config(args = None, no_check = False):
    if args is None: args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    dummy  = TrainConfig()
    for k, v in dummy.__dict__.items():
        vtype = str if no_check else type(v)
        required = k in TrainConfig.REQUIRED

        if vtype is list:
            parser.add_argument(f"--{k}", nargs="+", default = v, required = required)
        elif vtype is bool:
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

    if len(config.index) > 0:
        dataset = filter_dataset(config, dataset)

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
        print("Selector selects from this tools: %s" % str(selector.tools()))
        print("Save selector to %s" % config.model_path)
        joblib.dump(selector, config.model_path, compress = 3)
    
    if config.eval:
        print("Load selector from %s" % config.model_path)
        evaluator = load_evaluator_from_file(config.label_path)
        selector = joblib.load(config.model_path)
        print("Selector selects from this tools: %s" % str(selector.tools()))

        embeddings = dataset.embedding
        if len(config.feature_union) > 0:
            embeddings = feature_union(config, dataset.instance_index, embeddings)

        predictions    = selector.predict(embeddings)
        mapping    = {dataset.instance_index[i]: tool for i, tool in enumerate(predictions)}
        result     = evaluator.eval(mapping, timelimit = config.timelimit)

        with open(config.eval_path, "w") as o:
            for entry in result:
                o.write(json.dumps({"name": entry[0], "run": entry[4], "result": entry[1]})+"\n")
        
        solved = result.score()
        total  = len(dataset.instance_index)
        print("Oracle score for current config: %.2f (%d instances)" % (100 * (solved / total), solved))


if __name__ == '__main__':
    main()