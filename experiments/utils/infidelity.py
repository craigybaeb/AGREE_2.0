import numpy as np

# Calculate infidelity
def calculate_infidelity(instance, explanation, model, num_samples=50):
    total_infidelity = 0
    for _ in range(num_samples):
        perturbation = np.random.normal(0, 0.1, instance.shape)
        perturbed_instance = instance - perturbation
        pred_perturbed = model.predict(perturbed_instance.reshape(1, -1)).argmax(axis=1)[0]
        pred_original = model.predict(instance.reshape(1, -1)).argmax(axis=1)[0]
        output_perturbation = pred_perturbed - pred_original
        input_perturbation = np.dot(explanation, perturbation)
        total_infidelity += (input_perturbation - output_perturbation)**2
    
    return total_infidelity / num_samples
