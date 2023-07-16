import glob
import os
import sys
import time
from functools import wraps
from typing import List, Tuple, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import numpy as np
from loguru import logger
from tqdm import tqdm

from factorization.ilargi.IlargiMatrix import IlargiMatrix, csp, xp
from factorization.ilargi.algorithms.GaussianNMF import GaussianNMF
from factorization.ilargi.algorithms.KMeans import KMeans
from factorization.ilargi.algorithms.LinearRegression import LinearRegression
from factorization.ilargi.algorithms.LogisticRegression import LogisticRegression
from data_generator.generator.configuration import Configuration
from cost_estimation.estimator.CostEstimator import CostEstimator
from cost_estimation.estimator.PlannerFeatures import PlannerFeatures
from factorization.util import TimingResult, FeatureResult, create_ilargi_matrix, GENERATED

# log results to file immediately to not lose results



logger.remove()
logger.add(sys.stderr, level=os.environ.get('EXPERIMENT_LOG_LEVEL', "INFO"))
resultlogger = logger.bind(name='result')
featurelogger=logger.bind(name='feature')

kmeans_k = 3
gaussianNFM_r = 2


if os.environ.get("EXPERIMENT_TYPE", "HAMLET") == "HAMLET":
    logger.info("Running hamlet experiments")
    from experiments.configs.hamlet_experiment_config import experiment_config as config
else:
    logger.info("Running synthetic experiments")
    from experiments.configs.synthetic_experiment_config import experiment_config as config


def create_model_tuples(models: List[str], NM: IlargiMatrix) -> List[Tuple[Any, str]]:
    model_list = []
    if 'materialized' in models:
        model_list.append((NM.materialize(), 'materialized'))
    if 'factorized' in models:
        model_list.append((NM, 'factorized'))
    return model_list


def run_experiments(datasets=config['datasets'], joins=config['joins'], operators=config['operators'],
                    num_repeats=config['repeats'], models=config['models'], resultfile=None):
    """

    :param datasets: What datasets to run. If this is a list we assume the datasets are a list of names from the
        hamlet datasets. If it is a string we assume it is a path to a directory containing data generated
        by the Ilargi Data Generator and the accompanying configs.
    :param joins: Which joint types to run
    :param operators: What operators to run
    :param num_repeats: How many times to repeat
    :param models: What models to run (baseline and/or ilargi)
    :param resultfile: Path to file to store results
    :return: None
    """
    logger.add(resultfile or "results/{time}_results.log", format="{message}",
               filter=lambda message: message["extra"].get("name") == 'result')
    logger.add('results/features_{time}.log', format="{message}",filter=lambda message: message["extra"].get("name") == 'feature')
    num_cores = os.environ.get('NUM_CORES', -1)
    record_config = os.environ.get('RECORD_CONFIG', "True") == "True"
    cupy_profile = os.environ.get('CUPY_PROFILE', "False") == 'True'
    logger.info(F"Running experiment.py with {num_cores=}, {record_config=}, {cupy_profile=}")
    # allow parrallel run of single dataset collection
    task_num = int(os.environ.get("TASK_NUM", "0"))
    total_tasks = int(os.environ.get("TOTAL_TASKS", "1"))

    dataset_type = "hamlet"
    # Handle data from data generator
    if isinstance(datasets, str):
        logger.info(f"Running experiment with Data from data generator (in dir: {datasets})")
        # generated data
        dataset_type = GENERATED
        datasets = glob.glob(datasets)
        joins = ['preset']
        logger.info(f"Found {len(datasets)} datasets")

    logger.info(f"Starting experiments with {total_tasks=}, {task_num=}")
    for i, dataset in enumerate(tqdm(sorted(datasets), desc="Dataset progression")):
        if (i+1) % total_tasks != task_num:
            logger.info(f"Skipping dataset {i}:{dataset} ")
            continue
        logger.info(f"Running dataset: {dataset}")
        for join in tqdm(joins):
            logger.info(f"Running join {join}")

            exception = False
            try:
                ilargi_matrix: IlargiMatrix = create_ilargi_matrix(dataset, join, dataset_type)
            except np.core._exceptions._ArrayMemoryError as _:
                logger.exception(f"Not enough memory for dataset: {dataset}, skipping")
                exception = "Memoryerror"
            except AttributeError as e:
                logger.exception(f"Attribute error: {e}, skipping")
                exception = "AttributeError"
            except FileNotFoundError as e:
                logger.exception(f"Dataset not found")
                exception = "Filenotfound"
            except Exception as e:
                logger.exception(f"CAUGHT GENERIC EXCEPTION {e}")
                exception = 'generic'

            if exception:
                res = TimingResult(dataset, join, f'fail: {exception}', '', 0, 0, 0, 0,
                                   0, 0, [0])
                resultlogger.info(res)
                continue

            try:
                model_list = create_model_tuples(models, ilargi_matrix)
            except ValueError as e:
                print("Inconsistent shapes, skipping")
                continue
            est = CostEstimator(ilargi_matrix)
            fea= PlannerFeatures(est.AM,est.T)

            data_characteristics = None
            if dataset_type == GENERATED and record_config:
                try:
                    conf = Configuration(glob.glob(f"{dataset}/../*.ini")[0])
                    data_characteristics = conf.to_dict(conf)
                    data_characteristics['Tnonzero'], data_characteristics['Snonzero'] = est.scalar_op(est)
                except IndexError:
                    logger.warning(f"Configuration file for dataset {dataset} not found")
            # Collect metrics
            for operator in operators:
                logger.debug(f"Running operator {operator}, {join=}, {dataset=}")
                operator_function, complexity, arguments = get_operator_arguments(operator, ilargi_matrix, est)
                feature_info=get_features(operator,ilargi_matrix,fea,dataset)
                featurelogger.info(feature_info)
                for table, model in model_list:
                    # model can be 'factorized', 'materialized'
                    args = (
                        num_repeats, operator_function, arguments, table, operator, model, dataset, join, est,
                        complexity,
                        num_cores, data_characteristics)
                    try:
                        res = run_repeats(*args)
                        resultlogger.info(res)
                    except Exception as e:
                        logger.exception(e)
                        res = TimingResult(dataset, join, f'{operator}= fail: {e}', model, est.TR, est.FR,
                                           est.cardinality_T, est.cardinality_S, complexity[model], est.selectivity,
                                           [0], num_cores, data_characteristics)
                        resultlogger.info(res)
                        continue


d_X = config['d_X']


def run_repeats(num_repeats, operator_function, arguments, table, operator, model, dataset, join, est,
                complexity, num_cores, data_characteristics) -> TimingResult:
    timing_results = []
    for k in range(num_repeats):
        logger.debug(f"Performing function {operator_function.__name__}")
        logger.trace(f"with args: {arguments=}")
        t = operator_function(table, *arguments)
        logger.debug(f"operator {operator} took {t :.4f}s for {model}")
        timing_results += [t]
    res = TimingResult(dataset, join, operator, model, est.TR, est.FR, est.cardinality_T,
                       est.cardinality_S, complexity[model],
                       est.selectivity, timing_results, num_cores, data_characteristics)
    return res



def get_complexity(est: CostEstimator, operator: str):
    estfunc = lambda: (-1, -1)

    standard, factorized = estfunc()
    return {
        ""
    }

def get_features(operator, ilargi_matrix, est: PlannerFeatures, dataset):
    """
    Gets the function that should be performed on the tables, as well as any
    extra arguments needed to perform the operations, such as a matrix to multiply
    with for Matrix Multiplication.

    Also gets the complexity from the CostEstimator
    :param operator: String representation of operation to perform
    :param ilargi_matrix: Table to perform the operation on
    :return: A 3-tuple (operator function, dict containing the complexity for Factorized/Materialized,
    any extra arguments needed to perform the operation)
    """
    extra_args = ()

    # placeholder complexity function for operations with unknown complexity: materialization
    def complexity_function(*args, **kwargs):
        return (None, None)

    complexity_function_arguments = None

    if operator in ["Left multiply", "left_multiply"]:
        complexity_function = est.scalar_op

    elif operator in ["Right multiply", "right_multiply"]:
        complexity_function = est.scalar_op

    elif operator in ["Row summation", "row_sum"]:
        complexity_function = est.scalar_op

    elif operator in ["Column summation", "col_sum"]:
        complexity_function = est.scalar_op

    elif operator in ["LMM", "left_matrix_multiply"]:
        X = xp.ones((ilargi_matrix.shape[1], d_X))
        complexity_function_arguments = (X.shape,)
        complexity_function = est.LMM

    elif operator in ["RMM", "right_matrix_multiply"]:
        X = csp.random(d_X, ilargi_matrix.shape[0], format='csr')
        complexity_function_arguments = (X.shape,)
        complexity_function = est.RMM

    elif operator in ["Left multiply T", "left_matrix_multiply_transpose"]:
        complexity_function = est.scalar_op

    elif operator in ["Right multiply T", "right_matrix_multiply_transpose"]:
        complexity_function = est.scalar_op

    elif operator in ["Row summation T", "row_sum"]:
        complexity_function = est.scalar_op

    elif operator in ["Column summation T", "col_sum"]:
        complexity_function = est.scalar_op

    elif operator in ["LMM T", "left_matrix_multiply_transpose"]:
        X = csp.random(ilargi_matrix.shape[0], d_X, format='csr')
        complexity_function_arguments = (X.shape,)
        complexity_function = est.LMM_T

    elif operator in ["RMM T", "right_matrix_multiply_transpose"]:
        X = csp.random(m=d_X, n=ilargi_matrix.shape[1], format='csr')
        complexity_function_arguments = (X.shape,)
        complexity_function = est.RMM_T

    if operator in ["Linear Regression", "linear_regression"]:
        extra_args = (csp.csr_matrix(xp.ones((ilargi_matrix.shape[0], 1))),)
        complexity_function = est.LinR

    elif operator in ["Logistic Regression", "logistic_regression"]:
        extra_args = (csp.csr_matrix(xp.ones((ilargi_matrix.shape[0], 1))),)
        complexity_function = est.LogR

    elif operator in ["Gaussian", "gaussian"]:
        complexity_function_arguments = (2,)
        complexity_function = est.GaussianNMF

    elif operator in ["KMeans", "kmeans"]:
        complexity_function = est.KMeans
        complexity_function_arguments = (3,5,)

    # else:
    #     raise ValueError(f"Supplied operator {operator} is not recognized.")

    features = complexity_function(est,
                                               *complexity_function_arguments) if complexity_function_arguments else complexity_function(
        est)
    r_S = [est.AM.S[k].shape[0] for k in range(est.AM.K)]
    c_S = [est.AM.S[k].shape[1] for k in range(est.AM.K)]
    nnz_S = [est.AM.S[k].nnz for k in range(est.AM.K)]
    r_T = est.T.shape[0]
    c_T = est.T.shape[1]
    nnz_T = est.T.nnz
    TR = est.r_S[0] / sum(est.r_S[1:])
    FR = sum(est.c_S[1:]) / est.c_S[0]

    res = FeatureResult(dataset, operator, features,r_S,c_S,nnz_S,r_T,c_T,nnz_T,TR,FR)

    return res


def get_operator_arguments(operator, ilargi_matrix, cost_estimator: CostEstimator):
    """
    Gets the function that should be performed on the tables, as well as any
    extra arguments needed to perform the operations, such as a matrix to multiply
    with for Matrix Multiplication.

    Also gets the complexity from the CostEstimator
    :param operator: String representation of operation to perform
    :param ilargi_matrix: Table to perform the operation on
    :return: A 3-tuple (operator function, dict containing the complexity for Factorized/Materialized,
    any extra arguments needed to perform the operation)
    """
    extra_args = ()

    # placeholder complexity function for operations with unknown complexity: materialization
    def complexity_function(*args, **kwargs):
        return (None, None)

    complexity_function_arguments = None
    if operator in ["Left multiply", "left_multiply"]:
        operator_function = left_multiply
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Right multiply", "right_multiply"]:
        operator_function = right_multiply
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Row summation", "row_sum"]:
        operator_function = row_sum
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Column summation", "col_sum"]:
        operator_function = col_sum
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Materialization", "materialization"]:
        operator_function = materialization

    elif operator in ["LMM", "left_matrix_multiply"]:
        operator_function = left_matrix_multiply
        X = xp.ones((ilargi_matrix.shape[1], d_X))
        complexity_function_arguments = (X.shape,)
        extra_args = (X,)
        complexity_function = cost_estimator.LMM

    elif operator in ["RMM", "right_matrix_multiply"]:
        operator_function = right_matrix_multiply
        X = csp.random(d_X, ilargi_matrix.shape[0], format='csr')
        complexity_function_arguments = (X.shape,)
        extra_args = (X,)
        complexity_function = cost_estimator.RMM

    elif operator in ["Left multiply T", "left_matrix_multiply_transpose"]:
        operator_function = left_multiply_transpose
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Right multiply T", "right_matrix_multiply_transpose"]:
        operator_function = right_multiply_transpose
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Row summation T", "row_sum"]:
        operator_function = row_sum_transpose
        complexity_function = cost_estimator.scalar_op

    elif operator in ["Column summation T", "col_sum"]:
        operator_function = col_sum_transpose
        complexity_function = cost_estimator.scalar_op

    elif operator in ["LMM T", "left_matrix_multiply_transpose"]:
        operator_function = left_matrix_multiply_transpose
        X = csp.random(ilargi_matrix.shape[0], d_X, format='csr')
        complexity_function_arguments = (X.shape,)
        extra_args = (X,)
        complexity_function = cost_estimator.LMM_T

    elif operator in ["RMM T", "right_matrix_multiply_transpose"]:
        operator_function = right_matrix_multiply_transpose
        X = csp.random(m=d_X, n=ilargi_matrix.shape[1], format='csr')
        complexity_function_arguments = (X.shape,)
        extra_args = (X,)
        complexity_function = cost_estimator.RMM_T

    elif operator in ["Linear Regression", "linear_regression"]:
        operator_function = linear_regression
        extra_args = (csp.csr_matrix(xp.ones((ilargi_matrix.shape[0], 1))),)
        complexity_function = cost_estimator.LinR

    elif operator in ["Logistic Regression", "logistic_regression"]:
        operator_function = logistic_regression
        extra_args = (csp.csr_matrix(xp.ones((ilargi_matrix.shape[0], 1))),)
        complexity_function = cost_estimator.LogR

    elif operator in ["Gaussian", "gaussian"]:
        operator_function = gaussianNMF
        complexity_function_arguments = (2,)
        complexity_function = cost_estimator.GaussianNMF

    elif operator in ["KMeans", "kmeans"]:
        operator_function = kmeans
        complexity_function = cost_estimator.KMeans
        complexity_function_arguments = (3,)

    elif operator in ["noop", "Noop", "NOOP", None]:
        operator_function = do_nothing

    else:
        raise ValueError(f"Supplied operator {operator} is not recognized.")

    standard, factorized = complexity_function(cost_estimator,
                                               *complexity_function_arguments) if complexity_function_arguments else complexity_function(
        cost_estimator)
    complexity = {
        "materialized": standard,
        "factorized": factorized
    }

    return operator_function, complexity, extra_args


def timeit(func):
    """
    Decorator to time function calls.
    Replaces return value with time taken to execute
    :param func: function to time
    :return: time taken
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        _ = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return total_time

    return timeit_wrapper


@timeit
def do_nothing(*args, **kwargs):
    return


@timeit
def left_multiply(T):
    T * 2


@timeit
def left_multiply_transpose(T):
    T.T * 2


@timeit
def right_multiply(T):
    2 * T


@timeit
def right_multiply_transpose(T):
    2 * T.T


@timeit
def row_sum(T):
    T.sum(axis=1)


@timeit
def row_sum_transpose(T):
    T.T.sum(axis=1)


@timeit
def col_sum(T):
    T.sum(axis=0)


@timeit
def col_sum_transpose(T):
    T.T.sum(axis=0)


@timeit
def materialize(NM: IlargiMatrix):
    NM.materialize()


def materialization(T):
    if isinstance(T, IlargiMatrix):
        res = materialize(T)
    else:
        res = -1.
    return res


@timeit
def left_matrix_multiply(T, X):
    T.dot(X)


def right_matrix_multiply(T, Y):
    if isinstance(T, IlargiMatrix):
        res = right_matrix_multiply_ilargi(T, Y)
    else:
        res = right_matrix_multiply_baseline(T, Y)
    return res


@timeit
def left_matrix_multiply_transpose(T, X):
    T.T.dot(X)


def right_matrix_multiply_transpose(T, Y):
    if isinstance(T, IlargiMatrix):
        res = right_matrix_multiply_ilargi(T.T, Y)
    else:
        res = right_matrix_multiply_baseline(T.T, Y)
    return res


@timeit
def right_matrix_multiply_baseline(T, Y):
    Y.dot(T)


@timeit
def right_matrix_multiply_ilargi(NM, Y):
    NM.__rmul__(Y)


@timeit
def linear_regression(X, Y):
    model = LinearRegression(0.1, 20)
    LinearRegression(0.1, 20).fit(model, X, Y)


@timeit
def logistic_regression(X, Y):
    model = LogisticRegression(0.1, 5)
    LogisticRegression(0.1, 5).fit(model, X, Y)


@timeit
def gaussianNMF(X):
    model = GaussianNMF(gaussianNFM_r, 5)
    GaussianNMF(gaussianNFM_r, 5).fit(model, X)


@timeit
def kmeans(X):
    model = KMeans(kmeans_k, X, 5)
    KMeans(kmeans_k, X, 5).fit(model, X)


if __name__ == '__main__':
    run_experiments()
