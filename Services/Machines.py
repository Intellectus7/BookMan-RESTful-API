import os
import warnings
import copy
import importlib
import shelve
import subprocess, sys, os
warnings.filterwarnings("ignore", category=UserWarning)
def require(package, import_as=None):
    """
    Ensure that a package is installed and imported.
    Example:
        requests = require("requests")
        np = require("numpy", "np")
    """
    try:
        return importlib.import_module(import_as or package)
    except ImportError:
        print(f"⚠️  {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(import_as or package)
pd =  require("pandas")
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle



def load_data(path, sep=','):
    """Load any CSV file."""
    path = os.path.normpath(os.path.abspath(path))
    return pd.read_csv(path, sep=sep)

def preprocess_data(df, cols_to_drop=None, target_col=None, encode_categoricals=True, Num=False):
    """Clean data: drop columns/NaNs, encode categoricals, split X/y."""
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    if Num:
        df = df.dropna()
    
    if encode_categoricals:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    if target_col is None:
        raise ValueError("Please provide the target column name.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y
def KNN(n_neighbours, dataset, index=-1):
# Train model
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)
    # Make copies
    prediction = data.copy()
    columnNames = data.columns
    data.drop(columnNames[index], axis=1, inplace=True)
    prediction.drop(columnNames[:index], axis=1, inplace=True)
    modelScores = []
    X = data.values.tolist()
    original_X = copy.deepcopy(X)

    for i in range(len(X)):
        for j in range(len(X[i])):
            if isinstance(X[i][j], str):
                try:
                    X[i][j] = float(X[i][j])
                except:
                    X[i][j] = 0

    if original_X == X:
        print("Same X")
    else:
        print("Converted X.")

    # Convert Y (target) and replace all strings with 0
    Y = prediction.values.ravel().tolist()
    original_Y = copy.deepcopy(Y)

    for i in range(len(Y)):
        if isinstance(Y[i], str):
            try:
                Y[i] = float(Y[i])
            except:
                Y[i] = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=43)
    model = KNeighborsClassifier(n_neighbors=n_neighbours)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # Evaluate
    modelAccuracy = model.score(X_test, Y_test)
    print("Model Accuracy:", modelAccuracy)

    for i in range(len(predicted)):
        print("Predicted: ", predicted[i], "Actual value: ", X_test[i])
    print("Model Accuracy:", modelAccuracy)
    modelScores.append(modelAccuracy)
    return modelAccuracy
def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into train and test sets."""
    return train_test_split(X.values, y.values, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    """Train linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def column_type(db, column=None, isObject=False, isTuple=False):
    try:
        if not column:
            return db.dtypes  # <-- Fixed here
        else:
            if isTuple:
                return list(db.dtypes.items())
            if not isObject:
                return db.dtypes[column]
            else:
                print("In Machines, the value is", db[column].dtype)
                return db[column].dtype
    except Exception as e:
        return e


def evaluate_model(model, X_test, y_test):
    """Calculate R^2 score."""
    return model.score(X_test, y_test)

def save_model(model, filename):
    """Save model to file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load model from file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_metrics(labels, values, colors=None, title="Metrics", xlabel="Metrics", ylabel="Score"):
    """Plot bar chart of metrics."""
    plt.bar(labels, values, color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def unsup(queue, estimator, name, data, true_labels=None):
    """
    Benchmark any clustering estimator.
    
    queue        -> Multiprocessing queue for result passing
    estimator    -> An untrained clustering model (e.g., KMeans(n_clusters=3))
    name         -> String name for the algorithm
    data         -> Feature matrix (numpy array or pandas DataFrame)
    true_labels  -> Optional array of true labels (for benchmarking only)
    """
    start = time.time()
    estimator.fit(data)
    
    # Many clustering algorithms store labels_ after fitting
    try:
        pred_labels = estimator.labels_
    except AttributeError:
        pred_labels = estimator.predict(data)
    
    # If true labels are given, align cluster IDs to maximize accuracy
    if true_labels is not None:
        if accuracy_score(true_labels, pred_labels) < accuracy_score(true_labels, 1 - pred_labels):
            pred_labels = 1 - pred_labels
        model_accuracy = accuracy_score(true_labels, pred_labels)
    else:
        model_accuracy = None  # no accuracy without labels

    end = time.time()
    queue.put((name, model_accuracy, end - start, None))

from sklearn.metrics import accuracy_score
import numpy as np
import time

def kMeans( estimator, name, data, true_labels=None):
    """
    Benchmark any clustering estimator.
    
    queue        -> Multiprocessing queue for result passing
    estimator    -> An untrained clustering model (e.g., KMeans(n_clusters=3))
    name         -> String name for the algorithm
    data         -> Feature matrix (numpy array or pandas DataFrame)
    true_labels  -> Optional array of true labels (for benchmarking only)
    """
    estimator.fit(data)
    
    # Many clustering algorithms store labels_ after fitting
    try:
        pred_labels = estimator.labels_
    except AttributeError:
        pred_labels = estimator.predict(data)
    
    # If true labels are given, align cluster IDs to maximize accuracy
    if true_labels is not None:
        if accuracy_score(true_labels, pred_labels) < accuracy_score(true_labels, 1 - pred_labels):
            pred_labels = 1 - pred_labels
        model_accuracy = accuracy_score(true_labels, pred_labels)
    else:
        model_accuracy = None  # no accuracy without labels
    return model_accuracy

def import_outside_directory(module_name: str, file_or_dir_path: str):
    """
    Dynamically import a module or package given an absolute path.
    
    Args:
        module_name: The name to register the module under (e.g., "Models", "Helper")
        file_or_dir_path: Either a path to a .py file OR a package directory with __init__.py
    """
    # If it's a directory, assume it's a package and point to its __init__.py
    if os.path.isdir(file_or_dir_path):
        file_or_dir_path = os.path.join(file_or_dir_path, "__init__.py")

    if not os.path.exists(file_or_dir_path):
        raise FileNotFoundError(f"Cannot find module at {file_or_dir_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_or_dir_path)
    if spec is None:
        raise ImportError(f"Cannot create spec for {module_name} at {file_or_dir_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
import shelve
import dbm

class ReactiveValue:
    def __init__(self, storage, key, value):
        self._storage = storage
        self._key = key
        self._value = value
        self._storage.setItem(self._key, self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self._storage.setItem(self._key, self._value)


class ReactiveList(list):
    def __init__(self, storage, key, iterable=None):
        super().__init__(iterable or [])
        self._storage = storage
        self._key = key
        self._persist()

    def _persist(self):
        self._storage.setItem(self._key, list(self))

    def append(self, item):
        super().append(item)
        self._persist()

    def extend(self, iterable):
        super().extend(iterable)
        self._persist()

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._persist()


class localStorage:
    def __init__(self, filename="local_storage"):
        self.filename = filename
        self.passkey = (
            self.filename
            + "4$jduy9y93yhsi03yg_3g9s-+jsehu%=azbajahisols223#ihihdihdsj@ihihsi!hgkshiheljhdroruh^"
        )

    def _open(self, flag="c", writeback=False):
        """Try default dbm, fall back to dumb if needed"""
        try:
            return shelve.open(self.filename, flag=flag, writeback=writeback)
        except dbm.error:
            return shelve.Shelf(dbm.dumb.open(self.filename, flag))

    def setItem(self, key, value):
        with self._open() as db:
            db[key] = value

    def getItem(self, key):
        with self._open() as db:
            val = db.get(key, None)
        # Wrap in reactive object automatically
        if isinstance(val, list):
            return ReactiveList(self, key, val)
        elif val is not None:
            return ReactiveValue(self, key, val)
        return None

    def removeItem(self, key):
        with self._open() as db:
            if key in db:
                del db[key]

    def clear(self):
        with self._open() as db:
            db.clear()

    def generate(self, name, value):
        with self._open(flag="c", writeback=True) as db:
            if name in db:
                return db[name]
            else:
                db[name] = value
                return db[name]

    def getAll(self, passkey):
        if passkey == self.passkey:
            with self._open() as db:
                return dict(db)  # return as normal dict, not Shelf
        else:
            return "Invalid password for the file"

