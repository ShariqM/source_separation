"""Python implementation of Blind Source Separation metric*

  This code implements BSS using method the time distortion method from III.B
  in the paper*.

  This code is a slim version of:
	-  https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py

  Which was implemented based on the following bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

  - :func:`mir_eval.separation.bss_eval_sources`: Computes the bss_eval_sources
    metrics from bss_eval, which optionally optimally match the estimated sources
    to the reference sources and measure the distortion and artifacts present in
    the estimated sources as well as the interference between them.

  * [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
     Févotte, "Performance measurement in blind audio source separation," IEEE
     Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.
"""

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)

def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
    return (sdr, sir, sar)

def _project(reference_sources, estimated_source, flen):
    """least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via fft ##
    # zero padding and fft of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    g = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(scipy.fftpack.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
            g[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            g[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    d = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        d[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # computing projection
    # distortion filters
    try:
        c = np.linalg.solve(g, d).reshape(flen, nsrc, order='f')
    except np.linalg.linalg.linalgerror:
        c = np.linalg.lstsq(g, d)[0].reshape(flen, nsrc, order='f')
    # filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(c[:, i], reference_sources[i])[:nsampl + flen - 1]
    return sproj

def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    nsampl = estimated_source.size
    # decomposition
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,
                      flen) - s_true
    # interference
    e_interf = _project(reference_sources,
                        estimated_source, flen) - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source
    return (s_true, e_spat, e_interf, e_artif)

def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    #validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)

def mir_eval_SDR_impl(sources, source_estimates):
  batch_size = sources.shape[0]

  SDRs = np.zeros(batch_size, dtype=np.float32)
  for b in range(batch_size):
    (sdr, sir, sar, perm) = bss_eval_sources(sources[b, :, :], source_estimates[b, :, :])
    SDRs[b] = np.mean(sdr)

  return SDRs

def mir_eval_SDR(sources, source_estimates):
  SDRs = tf.py_func(mir_eval_SDR_impl, [sources, source_estimates], tf.float32)
  SDRs.set_shape(sources.shape[0])
  return tf.reduce_mean(SDRs)
