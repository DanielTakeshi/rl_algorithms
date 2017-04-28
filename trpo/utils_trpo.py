"""
Some TRPO-specific stuff, which conatins the conjugate gradient and backtracking
line search methods. Both of these methods are borrowed from John Schulman's
code:

    https://github.com/joschu/modular_rl

They have been slightly modified to suit my code.
"""

import numpy as np


def cg(f_Ax, b, cg_iters=10, verbose=False, residual_tol=1e-10):
    """ Conjugate gradient, from John Schulman's code. 
    
    Sculman used Demmel's book on applied linear algebra, page 312. Fortunately
    I have a copy of it!! Shewchuk also has a version of this in his paper.
    However, Shewchuk emphasizes that this is most useful for *sparse* matrices
    `A`. We certainly have a *large* matrix since the number of rows/columns is
    equal to the number of neural network parameters, but is it sparse?

    This is used for solving linear systems of `Ax = b`, or `x = A^{-1}b`. In
    TRPO, we don't want to compute `A` (let alone its inverse).  In addition,
    `b` is our usual policy gradient. The goal is to find `A^{-1}b` and then
    later (outside this code) scale that by `alpha`, and then we get the update
    at last. I *think* the alpha-scaling comes from the line search, but I'm not
    sure yet.

    Params:
        f_Ax: A function designed to mimic A*(input). However, we *don't* have
            the entire matrix A formed. It's the "Fisher-vector" product and
            should be computed with Tensorflow. [TODO HOW??]
        b: A known vector. In TRPO, it's the vanilla policy gradient (I think).
        cg_iters: Number of iterations of CG.
        verbose: Print extra information for debugging.
        residual_tol: Exit CG if ||r||_2^2 is small enough.

    Returns:
        Our estimate of `A^{-1}b` where A is (approximately?) the Hessian of the
        KL divergence and `b` is given to us.
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print titlestr % ("iter", "residual norm", "soln norm")

    for i in xrange(cg_iters):
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    if verbose: print fmtstr % (i+1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x


def backtracking_line_search(f, x, fullstep, expected_improve_rate, 
                             max_backtracks=10, accept_ratio=0.1):
    """ Backtracking line search, from John Schulman's code.

    I think this is the same as what's listed in Boyd & Vandenberghe's book.
    Remember that with backtracking line search, we have a fixed descent
    direction (typically the negative gradient, as I explain in my blog post)
    and we have to progressively decrease the step size. Here, that's
    `stepfrac`.

    Remember, this is *one* case of backtracking line search. We are *not*
    changing any directions, i.e. `fullstep` is our only direction we have.
    Also, conjugate gradient comes *before* this because we need to get the
    `fullstep` from it.
    
    Params:
        f: The function we're trying to minimize.
        x: The starting point for backtracking line search. Remember, it's
            really `theta` since it's the parameters of our policy net.
        fullstep: The descent direction, provided by conjugate gradient!
        expected_improve_rate: The slope dy/dx at the initial point.
        max_backtracks: The maximum amount of iterations (i.e. backtracks).
        accept_ratio: The ratio which helps to determine our stopping criterion.
            It seems like a variant of most textbook descriptions of
            backtracking line search; here, I think it's to ensure we get
            above a threshold of improvement.

    Returns:
        A tuple (Y, x) where Y is a boolean indicating whether we've
        successfully found a new point to go to, and x is that final point, or
        the original x if the line search didn't find a better point.
    """
    fval = f(x)
    print("fval before {}".format(fval))
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print("actual_i / expected_i / ratio = {:.4f} / {:.4f} / {:.4f}".format(
                actual_improve, expected_improve, ratio))
        if ratio > accept_ratio and actual_improve > 0:
            print("fval after {:.4f}".format(newfval))
            return (True, xnew)
    return (False, x)
