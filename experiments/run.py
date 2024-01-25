from breast_cancer import run_breast_cancer
# from connect_4 import run_connect_4
from aids import run_aids
# from mushroom import run_mushroom
# from malware import run_malware
# from student_grades import run_student_grades
from thyroid import run_thyroid

datasets = {
    # "breast_cancer": run_breast_cancer,
    # "aids": run_aids,
    # "connect_4": run_connect_4,
    # "mushroom": run_mushroom,
    # "malware": run_malware,
    # "student_grades": run_student_grades,
    "thyroid": run_thyroid
}

seeds = [7,15,23,9,3,11,1,8,2,6]  # Define the number of splits for the grid search
num_splits_optimisation = 3  # Define the number of splits for optimisation
seed = 42  # Random seed for reproducibility
verbose = 3  # Verbosity level for grid search
num_explanations = 10
explainers_to_use = ["kernel_shap", "deep_shap", "gradient_shap", "guided_backpropagation", "layerwise_relevance_propagation",  "maple", "smoothgrad"]

for dataset in list(datasets.keys()):
    run_dataset = datasets[dataset]
    run_dataset(seeds, num_splits_optimisation, seed, verbose, num_explanations, explainers_to_use)
