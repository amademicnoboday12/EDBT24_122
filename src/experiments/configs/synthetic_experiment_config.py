import os


experiment_config = dict(
    models=["materialized", "factorized"],
    repeats=int(os.environ.get("NUM_REPEATS", 3)),
    datasets=os.environ["DATA_GLOB_PATH"],
    joins=["inner", "outer", "left"],
    operators=[
        "Noop",
        "Left multiply",
        "Right multiply",
        "Row summation",
        "Column summation",
        "Materialization",
        "LMM",
        "RMM",
        "Left multiply T",
        "Right multiply T",
        "Row summation T",
        "Column summation T",
        "LMM T",
        "RMM T",  # NM if baseline else T
        "Linear Regression",
        "Gaussian",
        "KMeans",
        "Logistic Regression",
    ],
    d_X=10,
)
