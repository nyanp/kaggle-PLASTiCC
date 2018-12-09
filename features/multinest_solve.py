"""
A pythonic interface to MultiNest
"""

from __future__ import absolute_import, unicode_literals, print_function

from pymultinest.run import run
from pymultinest.analyse import Analyzer
import numpy
import tempfile
import shutil

"""
A pythonic interface to MultiNest. The arguments are the same as in 
*run* (see there), except that the functions are defined differently for 
convenience.

@param Prior:
	Takes a numpy array and returns the transformed array.
	Example:
	def myprior(cube):
		return cube * 20 - 10

@param Loglikelihood:
	takes a numpy array of the transformed parameters,
	and should return the loglikelihood


@param nparams:
	dimensionality of the problem

"""


def solve(LogLikelihood, Prior, n_dims, **kwargs):
    kwargs['n_dims'] = n_dims
    files_temporary = False
    if 'outputfiles_basename' not in kwargs:
        files_temporary = True
        tempdir = tempfile.mkdtemp('pymultinest')
        kwargs['outputfiles_basename'] = tempdir + '/'
    outputfiles_basename = kwargs['outputfiles_basename']

    def SafePrior(cube, ndim, nparams):
        try:
            a = numpy.array([cube[i] for i in range(n_dims)])
            b = Prior(a)
            for i in range(n_dims):
                cube[i] = b[i]
        except Exception as e:
            import sys
            sys.stderr.write('ERROR in prior: %s\n' % e)
            sys.exit(1)

    def SafeLoglikelihood(cube, ndim, nparams, lnew):
        a = numpy.array([cube[i] for i in range(n_dims)])
        l = float(LogLikelihood(a))
        if not numpy.isfinite(l):
            import sys
            sys.stderr.write('WARNING: loglikelihood not finite: %f\n' % (l))
            sys.stderr.write('         for parameters: %s\n' % a)
            sys.stderr.write('         returned very low value instead\n')
            return -1e100
        return l

    kwargs['LogLikelihood'] = SafeLoglikelihood
    kwargs['Prior'] = SafePrior
    run(**kwargs)

    analyzer = Analyzer(n_dims, outputfiles_basename=outputfiles_basename)
    best = analyzer.get_best_fit()

    if files_temporary:
        shutil.rmtree(tempdir, ignore_errors=True)

    return [best['log_likelihood']]+best['parameters']
