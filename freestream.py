# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate as interp

__all__ = ['ideal_eos', 'FreeStreamer']

__version__ = '1.0.1'


"""
References:

[1] J. Liu, C. Shen, U. Heinz
    Pre-equilibrium evolution effects on heavy-ion collision observables
    PRC 91 064906 (2015)
    arXiv:1504.02160 [nucl-th]
    http://inspirehep.net/record/1358669

[2] W. Broniowski, W. Florkowski, M. Chojnacki, A. Kisiel
    Free-streaming approximation in early dynamics
        of relativistic heavy-ion collisions
    PRC 80 034902 (2009)
    arXiv:0812.3393 [nucl-th]
    http://inspirehep.net/record/805616
"""


def ideal_eos(e):
    """
    Ideal equation of state: P = e/3

    """
    return e/3


class FreeStreamer(object):
    """
    Free streaming and Landau matching for boost-invariant hydrodynamic initial
    conditions.

    Parameters:

        initial  -- square (n, n) array containing the initial state
        grid_max -- x and y max of the grid in fm (see online readme)
        time     -- time to free stream in fm

    After creating a FreeStreamer object, extract the various hydro quantities
    using its methods

        Tuv, energy_density, flow_velocity, shear_tensor, bulk_pressure

    See the online readme and the docstring of each method.

    """
    def __init__(self, initial, grid_max, time):
        initial = np.asarray(initial)

        if initial.ndim != 2 or initial.shape[0] != initial.shape[1]:
            raise ValueError('initial must be a square array')

        nsteps = initial.shape[0]

        # grid_max is the outer edge of the outermost grid cell;
        # xymax is the midpoint of the same cell.
        # They are different by half a cell width, i.e. grid_max/nsteps.
        xymax = grid_max*(1 - 1/nsteps)

        # Initialize the 2D interpolating splines.
        # Need both linear and cubic splines -- see below.
        # The scipy class has the x and y dimensions reversed,
        # so give it the transpose of the initial state.
        xy = np.linspace(-xymax, xymax, nsteps)
        spline1, spline3 = (
            interp.RectBivariateSpline(xy, xy, initial.T, kx=k, ky=k)
            for k in [1, 3]
        )

        # Prepare for evaluating the T^μν integrals, Eq. (7) in [1] and
        # Eq. (10) in [2].  For each grid cell, there are six integrals
        # (for the six independent components of T^μν), each of which is a
        # line integral around a circle of radius tau_0.

        # The only way to do this with reasonable speed in python is to
        # pre-determine the integration points and vectorize the calculation.
        # Among the usual fixed-point (non-adaptive) integration rules, the
        # trapezoid rule was found to converge faster than both the Simpson
        # rule and Gauss-Legendre quadrature.

        # Set the number of points so the arc length of each step is roughly
        # the size of a grid cell.  Clip the number of points to a reasonable
        # range.
        npoints = min(max(int(np.ceil(np.pi*time*nsteps/grid_max)), 30), 100)
        phi = np.linspace(0, 2*np.pi, npoints, endpoint=False)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Cache the x and y evaluation points for the integrals.
        # X and Y are (nsteps, npoints) arrays.
        X = np.subtract.outer(xy, time*cos_phi)
        Y = np.subtract.outer(xy, time*sin_phi)

        # Create lists of the upper-triangle indices and corresponding weight
        # functions for the integrals.
        u, v, K = zip(*[
            (0, 0, np.ones_like(phi)),
            (0, 1, cos_phi),
            (0, 2, sin_phi),
            (1, 1, cos_phi*cos_phi),
            (1, 2, cos_phi*sin_phi),
            (2, 2, sin_phi*sin_phi),
        ])

        # K (6, npoints) contains the weights for each integral.
        K = np.array(K)
        K /= phi.size

        # Initialize T^μν array.
        Tuv = np.empty((nsteps, nsteps, 3, 3))

        # Compute the integrals one row at a time; this avoids significant
        # python function call overhead compared to computing one cell at a
        # time.  In principle everything could be done in a single function
        # call, but this would require a very large temporary array and hence
        # may even be slower.  Vectorizing each row sufficiently minimizes the
        # function call overhead with a manageable memory footprint.
        for row, y in zip(Tuv, Y):
            # Evaluate the splines on all the integration points for this row.
            # (These lines account for ~90% of the total computation time!)
            # Cubic interpolation (Z3) accurately captures the curvature of the
            # initial state, but can produce artifacts and negative values near
            # the edges; linear interpolation (Z1) cannot capture the
            # curvature, but behaves correctly at the edges.  To combine the
            # advantages, use Z3 where both splines are positive, otherwise set
            # to zero.
            Z1 = spline1(X, y, grid=False)
            Z3 = spline3(X, y, grid=False)
            Z3 = np.where((Z1 > 0) & (Z3 > 0), Z3, 0)

            # Z3 (nsteps, npoints) contains the function evaluations along the
            # circles centered at each grid point along the row.  Now compute
            # all six integrals in a single function call to the inner product
            # and write the result into the T^μν array.  np.inner calculates
            # the sum over the last axes of Z3 (nsteps, npoints) and K (6,
            # npoints), returning an (nsteps, 6) array.  In other words, it
            # sums over the integration points for each grid cell in the row.
            # np.inner is a highly-optimized linear algebra routine so this is
            # very efficient.
            row[:, u, v] = np.inner(Z3, K)

        # Copy the upper triangle to the lower triangle.
        u, v = zip(*[(0, 1), (0, 2), (1, 2)])
        Tuv[..., v, u] = Tuv[..., u, v]

        # Normalize the tensor for boost-invariant longitudinal expansion.
        Tuv /= time

        # Initialize class members.
        self._Tuv = Tuv
        self._energy_density = None
        self._flow_velocity = None
        self._shear_tensor = None
        self._total_pressure = None

    def Tuv(self, u=None, v=None):
        """
        Energy-momentum tensor T^μν.

        With no arguments, returns an (n, n, 3, 3) array containing the full
        tensor at each grid point.

        With two integer arguments, returns an (n, n) array containing the
        requested component of the tensor at each grid point.  For example
        FreeStreamer.Tuv(0, 0) returns T00.

        """
        if u is None and v is None:
            return self._Tuv
        elif u is not None and v is not None:
            return self._Tuv[..., u, v]
        else:
            raise ValueError('must provide both u and v')

    def _compute_energy_density_flow_velocity(self):
        """
        Compute energy density and flow velocity by solving the eigenvalue
        equation from the Landau matching condition.

        """
        # Ignore empty grid cells.
        T00 = self._Tuv[..., 0, 0]
        nonzero = T00 > 1e-16 * T00.max()

        # The Landau matching condition expressed as an eigenvalue equation is
        #
        #   T^μ_ν u^ν = e u^μ
        #
        # where the timelike eigenvector u^μ is the four-velocity required to
        # boost to the local rest frame of the fluid, and the eigenvalue e is
        # the energy density in the local rest frame.

        # Construct the mixed tensor Tu_v (n, 3, 3), where n is the number of
        # nonzero grid cells.
        Tu_v = np.copy(self._Tuv[nonzero])
        Tu_v[..., :, 1:] *= -1

        # The mixed tensor is NOT symmetric, so must use the general
        # eigensystem solver.  Recent versions of numpy can solve all the
        # eigensystems in a single function call (there's still an outer loop
        # over the array, but it is executed in C).
        eigvals, eigvecs = np.linalg.eig(Tu_v)

        # Eigenvalues/vectors can sometimes be complex.  This is numerically
        # valid but clearly the physical energy density must be real.
        # Therefore take the real part and ignore any complex
        # eigenvalues/vectors.
        if np.iscomplexobj(eigvals):
            imag = eigvals.imag != 0
            eigvals = eigvals.real
            eigvals[imag] = 0
            eigvecs = eigvecs.real
            eigvecs.transpose(0, 2, 1)[imag] = 0

        # eigvals (n, 3) contains the 3 eigenvalues for each nonzero grid cell.
        # eigvecs (n, 3, 3) contains the eigenvectors, where in each (3, 3)
        # block the columns are the vectors and the rows are the (t, x, y)
        # components.

        # The physical flow velocity and energy density correspond to the
        # (unique) timelike eigenvector.  Given eigenvectors (t, x, y) the
        # timelike condition may be written (t^2 > x^2 + y^2).  Since the
        # vectors are normalized to t^2 + x^2 + y^2 = 1, the timelike condition
        # may be simplified to t^2 > 1/2.  However, t^2 == 1/2 corresponds to a
        # perfectly lightlike vector, which is numerically undesirable.
        # Testing reveals that the maximum realistic gamma (Lorentz) factor is
        # ~40, but sometimes a few cells will have gamma >> 1000 due to
        # numerical errors.  Therefore ignore cells above a threshold.
        gamma_max = 100
        timelike = eigvecs[:, 0]**2 > 1/(2 - 1/gamma_max**2)

        # "timelike" is an (n, 3) array of booleans denoting the timelike
        # eigenvector (if any) for each grid cell.  This line updates the
        # "nonzero" mask to ignore cells that lack a timelike eigenvector.
        # Effectively it is a logical and, i.e. each grid cell must be nonzero
        # AND have a timelike eigvec.
        nonzero[nonzero] = timelike.any(axis=1)

        # Save the physical eigenvalues in the internal energy density array.
        self._energy_density = np.zeros(self._Tuv.shape[:2])
        self._energy_density[nonzero] = eigvals[timelike]

        # Select the timelike eigenvectors and correct the overall signs, if
        # necessary (the overall sign of numerical eigenvectors is arbitrary,
        # but u^0 should always be positive).
        u = eigvecs.transpose(0, 2, 1)[timelike]
        u0 = u[..., 0]
        u[u0 < 0] *= -1

        # Normalize the flow velocity in Minkowski space.  The numerical solver
        # returns vectors A*u normalized in Euclidean space as
        # A^2*(u0^2 + u1^2 + u2^2) = 1, which need to be renormalized as
        # u0^2 - u1^2 - u2^2 = 1.  The prefactor A may be derived by equating
        # these two normalizations.
        u /= np.sqrt(2*u0*u0 - 1)[..., np.newaxis]

        # Save internal flow velocity array.
        self._flow_velocity = np.zeros(self._Tuv.shape[:3])
        self._flow_velocity[..., 0] = 1
        self._flow_velocity[nonzero] = u

    def energy_density(self):
        """
        Energy density in the local rest frame from Landau matching.

        Returns an (n, n) array.

        """
        if self._energy_density is None:
            self._compute_energy_density_flow_velocity()

        return self._energy_density

    def flow_velocity(self, u=None):
        """
        Fluid flow velocity u^μ from Landau matching.

        With no arguments, returns an (n, n, 3) array containing the flow
        vector at each grid point.

        With a single integer argument, returns an (n, n) array containing the
        requested component of the flow vector at each grid point.

        """
        if self._flow_velocity is None:
            self._compute_energy_density_flow_velocity()

        if u is None:
            return self._flow_velocity
        else:
            return self._flow_velocity[..., u]

    def _compute_viscous_corrections(self):
        """
        Use T^μν and the results of Landau matching to calculate the shear
        pressure tensor π^μν and the total pressure (P + Π).

        """
        T = self.Tuv()

        # Flow velocity "outer product" u^μ u^ν.
        u = self.flow_velocity()
        uu = np.einsum('...i,...j', u, u)

        # Metric tensor g^μν in Minkowski space.
        g = np.diag([1., -1., -1.])

        # Projection operator Δ^μν.
        Delta = g - uu

        # Compute and save the total pressure = ideal + bulk = P + Π.
        # See Eq. (11) in [1].
        self._total_pressure = np.einsum('au,bv,...ab,...uv', g, g, Delta, T)
        self._total_pressure /= -3

        # Add two trailing dimensions to the energy density and total pressure
        # arrays (n, n) -> (n, n, 1, 1) so that they can broadcast onto the uu
        # and Delta arrays (n, n, 3, 3).
        e = self.energy_density()[..., np.newaxis, np.newaxis]
        Ptotal = self._total_pressure[..., np.newaxis, np.newaxis]

        # Compute and save the shear pressure tensor π^μν.
        # See Eq. (13) in [1].
        self._shear_tensor = T - e*uu + Ptotal*Delta

    def shear_tensor(self, u=None, v=None):
        """
        Shear pressure tensor π^μν.

        With no arguments, returns an (n, n, 3, 3) array containing the full
        tensor at each grid point.

        With two integer arguments, returns an (n, n) array containing the
        requested component of the tensor at each grid point.  For example
        FreeStreamer.shear_tensor(1, 2) returns pi12.

        """
        if self._shear_tensor is None:
            self._compute_viscous_corrections()

        if u is None and v is None:
            return self._shear_tensor
        elif u is not None and v is not None:
            return self._shear_tensor[..., u, v]
        else:
            raise ValueError('must provide both u and v')

    def bulk_pressure(self, eos=ideal_eos):
        """
        Bulk viscous pressure Π.

        Optional parameter eos must be a callable object that evaluates the
        equation of state P(e). The default is the ideal EoS, P(e) = e/3.

        Returns an (n, n) array.

        """
        if self._total_pressure is None:
            self._compute_viscous_corrections()

        # Compute Π = (P + Π) - P = (total pressure) - P, P = P(e) from eos.
        self._bulk_pressure = self._total_pressure - eos(self.energy_density())

        return self._bulk_pressure
