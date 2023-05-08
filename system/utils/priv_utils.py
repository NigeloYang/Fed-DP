import time
import numpy as np
import math

from system.utils.utils import transform

from opacus import PrivacyEngine


#################################ADD NOISE#######################################
def add_laplace(updates, sensitivity, epsilon):
    '''
    inject laplacian noise to a vector
    '''
    lambda_ = sensitivity * 1.0 / epsilon
    updates += np.random.laplace(loc=0, scale=lambda_, size=updates.shape)
    return updates


def add_gaussian(updates, eps, delta, sensitivity):
    '''
    inject gaussian noise to a vector
    '''
    sigma = (sensitivity / eps) * math.sqrt(2 * math.log(1.25 / delta))
    updates += np.random.normal(0, sigma)
    return updates


def one_gaussian(eps, delta, sensitivity):
    '''
    sample a gaussian noise for a scalar
    '''
    sigma = (sensitivity / eps) * math.sqrt(2 * math.log(1.25 / delta))
    return np.random.normal(0, sigma)


def one_laplace(eps, sensitivity):
    '''
    sample a laplacian noise for a scalar
    '''
    return np.random.laplace(loc=0, scale=sensitivity / eps)


def full_randomizer(vector, clip_C, eps, delta, mechanism, left=0, right=1):
    clipped = np.clip(vector, -clip_C, clip_C)
    normalized_updates = transform(clipped, -clip_C, clip_C, left, right)
    if mechanism == 'gaussian':
        perturbed = add_gaussian(normalized_updates, eps, delta, sensitivity=right - left)
    elif mechanism == 'laplace':
        perturbed = add_laplace(normalized_updates, sensitivity=1, epsilon=eps)
    return perturbed


def sampling_randomizer(vector, choices, clip_C, eps, delta, mechanism, left=0, right=1):
    # cpu
    start = time.time()
    vector = np.clip(vector, -clip_C, clip_C)
    for i, v in enumerate(vector):
        if i in choices:
            normalize_v = transform(vector[i], -clip_C, clip_C, left, right)
            if mechanism == 'gaussian':
                vector[i] = normalize_v + one_gaussian(eps, delta, right - left)
            elif mechanism == 'laplace':
                vector[i] = normalize_v + one_laplace(eps, right - left)
        else:
            vector[i] = 0
    print('sampling_randomizer cpu time: {:.4f}'.format(time.time() - start))
    return vector


MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA
