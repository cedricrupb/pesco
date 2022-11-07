import numpy as np

from sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.decomposition import  TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer


def preprocess(embedding):
    n_components = embedding.shape[1]
    lsa = make_pipeline(StandardScaler(), PCA(n_components = n_components, whiten = True))
    #lsa = make_pipeline(TfidfTransformer(), TruncatedSVD(n_components = 128), StandardScaler())
    embedding = lsa.fit_transform(embedding)
    
    return embedding, lsa 


# Baseline 1: Standard logistic regression

class LogisticRegressionSelector:

    def __init__(self, C = 0.1):
        self.C = C
        self.models_ = None

    def fit(self, X, y, runtimes = None):
        models = []
        for i in range(y.shape[1]):
            labels = y[:, i] 
            clf    =  LogisticRegression(C = self.C)
            clf.fit(X, labels)
            models.append(clf)
        
        self.models_ = models

    def predict(self, X, return_scores = False):
        assert self.models_ is not None, "You have to train the selector first"
        predictions = []
        for model in self.models_:
            prediction = model.predict_proba(X)[:, 1]
            predictions.append(prediction)

        predictions = np.stack(predictions).transpose()
        selection   = predictions.argmax(axis = 1)

        if return_scores:
            return selection, predictions

        return selection


# Baseline 2: PAR10 predictor

class PAR10Selector:

    def __init__(self, C = 0.1, weight = 9_000):
        self.C = C
        self.weight = weight

        self.models_ = None

    def fit(self, X, y, runtimes):
        weight = (y * runtimes) + ((1 - y) * self.weight)

        models = []
        for i in range(weight.shape[1]):
            labels = weight[:, i] 
            clf    =  Ridge(alpha = self.C)
            clf.fit(X, labels)
            models.append(clf)
        
        self.models_ = models

    def predict(self, X, return_scores = False):
        assert self.models_ is not None, "You have to train the selector first"
        predictions = []
        for model in self.models_:
            prediction = model.predict(X)
            predictions.append(prediction)

        predictions = np.stack(predictions).transpose()
        selection   = predictions.argmin(axis = 1)

        if return_scores:
            return selection, predictions

        return selection


class PAR10BucketSelector:

    BUCKETS = [0, 10, 100, 900, 9000]

    def __init__(self, C = 0.1, weight = 9_000, max_iter = 10_000):
        self.C = C
        self.weight = weight
        self.max_iter = max_iter

        self.models_ = None

    def fit(self, X, y, runtimes):
        weight = (y * runtimes) + ((1 - y) * self.weight)

        bucket_weight = np.zeros(weight.shape + (len(self.BUCKETS) - 1,))
        for i in range(bucket_weight.shape[-1]):
            lower, upper = self.BUCKETS[i], self.BUCKETS[i + 1]
            accept_upper = np.clip(upper - weight + 1, 0, 1).astype(int)
            accept_lower = np.clip(weight - lower, 0, 1).astype(int)
            bucket_weight[:, :, i] = accept_upper * accept_lower
        
        y = bucket_weight.argmax(axis = -1)

        models = []
        for i in range(weight.shape[1]):
            clf    =  LogisticRegression(C = self.C, max_iter = self.max_iter)
            clf.fit(X, y[:, i])
            models.append(clf)
        
        self.models_ = models

    def predict(self, X, return_scores = False):
        assert self.models_ is not None, "You have to train the selector first"
        predictions = []

        weight = np.array(self.BUCKETS[1:])

        for model in self.models_:
            prediction = model.predict_proba(X)
            pweight    = weight[model.classes_]
            prediction = (prediction * pweight).sum(axis = 1)
            predictions.append(prediction)

        predictions = np.stack(predictions).transpose()
        selection   = predictions.argmin(axis = 1)

        if return_scores:
            return selection, predictions

        return selection



class OrdinalPAR10BucketSelector:

    BUCKETS = [0, 10, 100, 900, 9000]

    def __init__(self, C = 0.1, weight = 9_000, max_iter = 10_000):
        self.C = C
        self.weight = weight
        self.max_iter = max_iter

        self.models_ = None

    def fit(self, X, y, runtimes):
        weight = (y * runtimes) + ((1 - y) * self.weight)

        bucket_weight = np.zeros(weight.shape + (len(self.BUCKETS) - 1,))
        for i in range(bucket_weight.shape[-1]):
            lower, upper = self.BUCKETS[i], self.BUCKETS[i + 1]
            accept_upper = np.clip(upper - weight + 1, 0, 1).astype(int)
            #accept_lower = np.clip(weight - lower, 0, 1).astype(int)
            bucket_weight[:, :, i] = accept_upper #* accept_lower
        
        y = bucket_weight

        models = []
        for i in range(y.shape[1]):
            clfs = []
            for j in range(y.shape[2]):
                clf    =  LogisticRegression(C = self.C, max_iter = self.max_iter)
                try:
                    clf.fit(X, y[:, i, j])
                except ValueError:
                    clf = y[:, i, j].min()
                clfs.append(clf)
            models.append(clfs)
        
        self.models_ = models

    def predict(self, X, return_scores = False):
        assert self.models_ is not None, "You have to train the selector first"
        predictions = []

        weight = np.array(self.BUCKETS[1:])

        for model in self.models_:
            model_predictions = []

            for clf in model:
                try:
                    pred = clf.predict_proba(X)[:, 1]
                except Exception:
                    pred = np.full((X.shape[0]), clf)
                
                model_predictions.append(pred)

            for i in range(len(model_predictions) - 1, 0, -1):
                model_predictions[i] = model_predictions[i] - model_predictions[i - 1]

            prediction = np.stack(model_predictions).transpose()
            prediction = (prediction * weight).sum(axis = 1)
            predictions.append(prediction)

        predictions = np.stack(predictions).transpose()
        selection   = predictions.argmin(axis = 1)

        if return_scores:
            return selection, predictions

        return selection



# Baseline 3: Cost-sensitive ranking ------

class RankingSelector:

    def __init__(self, C = 0.1, max_iter = 10_000, weight = 9_000):
        self.C = C
        self.weight = weight
        self.max_iter = max_iter

        self.models_      = None
        self.rank_pos_    = None
        self.num_options_ = None

    def fit(self, X, y, runtimes):
        weight = (y * runtimes) + ((1 - y) * self.weight)

        models   = []
        rank_pos = []

        for i in range(weight.shape[1] - 1):
            for j in range(i + 1, weight.shape[1]):
                rank_pos.append((i, j))
                rank_weight = weight[:, j] - weight[:, i]

                labels, sample_weight = np.clip(np.sign(rank_weight), 0, 1), np.abs(rank_weight)

                if labels.min() != labels.max():
                    clf = LogisticRegression(C = self.C, max_iter = self.max_iter)
                    clf.fit(X, labels, sample_weight = sample_weight)
                else:
                    clf = labels.min()
                
                models.append(clf)

        self.models_      = models
        self.rank_pos_    = rank_pos
        self.num_options_ = y.shape[1] 


    def predict(self, X, return_scores = False):
        assert self.models_ is not None, "You have to train the selector first"
        predictions = []
        for model in self.models_:
            try:
                prediction = model.predict_proba(X)[:, 1]
            except Exception:
                prediction = np.array([model] * X.shape[0])
            predictions.append(prediction)

        predictions = np.stack(predictions).transpose()

        scores = np.zeros((predictions.shape[0], self.num_options_))
        for i, (p, k) in enumerate(self.rank_pos_):
            pred = predictions[:, i]

            scores[:, p] += pred
            scores[:, k] += (1 - pred)

        selection   = scores.argmax(axis = 1)

        if return_scores:
            return selection, scores

        return selection


# Ranking model constructor --------------------------------

class RankingSelectorConstructor:

    def __init__(self, X, C = 0.1, max_iter = 10_000, weight = 9_000):
        self.X = X
        self.C = C
        self.max_iter = max_iter
        self.weight = weight

        self.weights_ = []

        self.models_      = []
        self.rank_pos_    = []
        self.num_options_ = 0

    def build(self):
        ranker = RankingSelector(self.C, self.max_iter, self.weight)
        ranker.models_ = list(self.models_)
        ranker.rank_pos_ = list(self.rank_pos_)
        ranker.num_options_ = self.num_options_
        return ranker

    def add(self, y, runtimes):
        candidate = self.num_options_
        weight = (y * runtimes) + ((1 - y) * self.weight)

        if candidate == 0:
            self.num_options_ += 1
            self.weights_.append(weight)
            return

        for p, cweight in enumerate(self.weights_):
            self.rank_pos_.append((p, candidate))
            advantage = weight - cweight
            labels, sample_weight = np.clip(np.sign(advantage), 0, 1), np.abs(advantage)
            try:
                clf = LogisticRegression(C = self.C, max_iter = self.max_iter)
                clf.fit(self.X, labels, sample_weight = sample_weight)
            except Exception:
                clf = labels.min()
            
            self.models_.append(clf)
        
        self.num_options_ += 1
        self.weights_.append(weight)

# Logistic Regression constructor -----

class LogisticRegressionSelectorConstructor:

    def __init__(self, X, C = 0.1):
        self.X = X
        self.C = C
        self.models_      = []

    def build(self):
        model = LogisticRegressionSelector(self.C)
        model.models_ = self.models_
        return model

    def add(self, y, runtimes = None):
        clf    =  LogisticRegression(C = self.C)
        clf.fit(self.X, y)
        self.models_.append(clf)


class PAR10BucketSelectorConstructor:

    def __init__(self, X, C = 0.1, weight = 9_000):
        self.X = X
        self.C = C
        self.weight = weight
        self.models_      = []

    def build(self):
        model = PAR10BucketSelector(self.C, max_iter = 10_000)
        model.models_ = self.models_
        return model

    def add(self, y, runtimes):
        weight = (y * runtimes) + ((1 - y) * self.weight)

        bucket_weight = np.zeros(weight.shape + (len(PAR10BucketSelector.BUCKETS) - 1,))
        for i in range(bucket_weight.shape[-1]):
            lower, upper = PAR10BucketSelector.BUCKETS[i], PAR10BucketSelector.BUCKETS[i + 1]
            accept_upper = np.clip(upper - weight + 1, 0, 1).astype(int)
            accept_lower = np.clip(weight - lower, 0, 1).astype(int)
            bucket_weight[:, i] = accept_upper * accept_lower
        
        y = bucket_weight.argmax(axis = -1)

        clf    =  LogisticRegression(C = self.C)
        clf.fit(self.X, y)
        self.models_.append(clf)


# Selector pipeline ----------------------------------------------

class SelectorPipeline:

    def __init__(self, model, label_index, preprocessor = None):
        self.model        = model
        self.label_index  = label_index
        self.preprocessor = preprocessor

    def _prediction_to_label(self, prediction):
        label_index = self.label_index

        try:
            return label_index[prediction]
        except TypeError:
            pass

        name_and_times = []
        for pred in prediction:
            if isinstance(pred, int):
                pred = (pred, 900)
            name_and_times.append((label_index[pred[0]], pred[1]))
        
        return ",".join(f"{p[0]}:{p[1]}" for p in name_and_times)

    def predict(self, features):
        
        if self.preprocessor is not None:
            features = self.preprocessor.transform(features)
        
        prediction = self.model.predict(features)

        return list(map(self._prediction_to_label, prediction))

