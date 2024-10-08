import torch 
import numpy as np
from lifelines.utils import concordance_index

def PartialLogLikelihood(logits, fail_indicator, ties):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    if ties == 'noties':
        likelihood = logits - torch.logcumsumexp(logits, dim=0)
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood.squeeze() * fail_indicator.squeeze()
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations


def calc_concordance_index(logits, fail_indicator, fail_time):
    """
    Compute the concordance-index value.
    Parameters:
        label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        y_pred: np.array, predictive proportional risk of network.
    Returns:
        concordance index.
    """
    hr_pred = -logits 
    try:
        for a in [fail_time, hr_pred, fail_indicator]:
            if np.isnan(a).any():
                raise ValueError("NaNs detected in inputs, please correct or drop.")
    except ValueError as e:
        print(e)
        return 0.5
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)
    return ci
