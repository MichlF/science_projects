from scipy.stats import t
from math import sqrt
import warnings


def wconfidence_int(data, p_value=.05, tail='two', morey=True):
    """
    within_confidence_int

    Cousineau's method (2005) for calculating within-subject confidence intervals
    If needed, Morey's correction (2008) can be applied (recommended).

    Parameters
    ----------
    data : ndarray
        Data for which CIs should be calculated
    p_value : float, optional
        p-value for determining t-value (the default is .05).
    tail : string, optional
        Two-tailed ('two') or one-tailed t-value.
    morey : bool, optional
        Apply Morey correction (the default is True)

    Returns
    -------
    CI : ndarray
        Confidence intervals for each condition
    """

    if tail == 'two':
        p_value = p_value/2
    elif tail not in ['two', 'one']:
        p_value = p_value/2
        warnings.warn('Incorrect argument for tail: using default ("two")')

    # normalize the data by subtracting the participants mean performance from each observation,
    # and then add the grand mean to each observation
    ind_mean = data.mean(axis=1).reshape(data.shape[0], 1)
    grand_mean = data.mean(axis=1).mean()
    data = data - ind_mean + grand_mean
    # Look up t-value and calculate CIs
    t_value = abs(t.ppf([p_value], data.shape[0]-1)[0])
    CI = data.std(axis=0, ddof=1)/sqrt(data.shape[0])*t_value

    # correct CIs according to Morey (2008)
    if morey:
        CI = CI*(data.shape[1]/float((data.shape[1] - 1)))

    return CI
