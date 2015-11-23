# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate as interp

__all__ = ['ideal_eos', 'FreeStreamer']

__version__ = '1.0-dev'


def ideal_eos(e):
    """
    Ideal equation of state: P = e/3

    """
    return e/3


class FreeStreamer(object):
    """
    Free streaming and Landau matching for boost-invariant hydrodynamic initial
    conditions.

    """
    def __init__(self, initial, grid_max, time):
        initial = np.asarray(initial)

        if initial.ndim != 2 or initial.shape[0] != initial.shape[1]:
            raise ValueError('initial must be a square array')

        nsteps = initial.shape[0]
        xymax = grid_max*(1 - 1/nsteps)
        xy = np.linspace(-xymax, xymax, nsteps)
        spline = interp.RectBivariateSpline(xy, xy, initial.T)

        # XXX
        npoints = min(max(int(np.ceil(np.pi*time*nsteps/grid_max)), 30), 100)
        phi = np.linspace(0, 2*np.pi, npoints, endpoint=False)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        X = np.subtract.outer(xy, time*cos_phi)
        Y = np.subtract.outer(xy, time*sin_phi)

        u, v, K = zip(*[
            (0, 0, np.ones_like(phi)),
            (0, 1, cos_phi),
            (0, 2, sin_phi),
            (1, 1, cos_phi*cos_phi),
            (1, 2, cos_phi*sin_phi),
            (2, 2, sin_phi*sin_phi),
        ])
        K = np.array(K)
        K /= phi.size

        Tuv = np.empty((nsteps, nsteps, 3, 3))

        for row, y in zip(Tuv, Y):
            Z = spline(X, y, grid=False)
            Z.clip(min=0, out=Z)
            row[:, u, v] = np.inner(Z, K)

        u, v = zip(*[(0, 1), (0, 2), (1, 2)])
        Tuv[..., v, u] = Tuv[..., u, v]

        Tuv /= time

        self._Tuv = Tuv
        self._energy_density = None
        self._flow_velocity = None
        self._eos = None

    def Tuv(self, u=None, v=None):
        """
        Energy-momentum tensor T^μν.

        """
        if u is None and v is None:
            return self._Tuv
        elif u is not None and v is not None:
            return self._Tuv[..., u, v]
        else:
            raise ValueError('must provide both u and v')

    def _compute_energy_density_flow_velocity(self):
        """
        Perform Landau matching to obtain the energy density and flow velocity
        profiles.

        """
        # ignore empty grid cells
        T00 = self._Tuv[..., 0, 0]
        nonzero = T00 > 1e-16 * T00.max()

        # The Landau matching condition is
        #   T^μν u_ν = e u^μ
        # which is not quite an eigenvalue equation because the flow velocity
        # index is lowered on the LHS and raised on the RHS.
        # Instead, we solve
        #   T^μ_ν u^ν = e u^μ.
        # Note that lowering the second index on T makes the matrix NOT
        # symmetric, so we must use the general eigensystem solver.
        Tu_v = np.copy(self._Tuv[nonzero])
        Tu_v[..., :, 1:] *= -1
        eigvals, eigvecs = np.linalg.eig(Tu_v)

        t, x, y = eigvecs.transpose(1, 0, 2)
        timelike = t*t > x*x + y*y

        nonzero[nonzero] = timelike.any(axis=1)

        self._energy_density = np.zeros(self._Tuv.shape[:2])
        self._energy_density[nonzero] = eigvals[timelike]

        u = eigvecs.transpose(0, 2, 1)[timelike]
        u0 = u[..., 0]
        u[u0 < 0] *= -1

        u /= np.sqrt(2*u0*u0 - 1)[..., np.newaxis]

        self._flow_velocity = np.zeros(self._Tuv.shape[:3])
        self._flow_velocity[..., 0] = 1
        self._flow_velocity[nonzero] = u

    def energy_density(self):
        """
        Energy density in the local rest frame from Landau matching.

        """
        if self._energy_density is None:
            self._compute_energy_density_flow_velocity()

        return self._energy_density

    def flow_velocity(self, u=None):
        """
        Fluid flow velocity u^μ from Landau matching.

        """
        if self._flow_velocity is None:
            self._compute_energy_density_flow_velocity()

        if u is None:
            return self._flow_velocity
        else:
            return self._flow_velocity[..., u]

    def _compute_viscous_corrections(self, eos):
        """
        Use T^μν and the results of Landau matching to calculate the shear
        pressure tensor π^μν and the bulk viscous pressure Π.

        """
        if eos is self._eos:
            return

        self._eos = eos

        T = self.Tuv()
        e = self.energy_density()
        P = self._eos(e)

        # flow velocity "outer product" u^μ u^ν
        u = self.flow_velocity()
        uu = np.einsum('...i,...j', u, u)

        # metric tensor g^μν in Minkowski space
        g = np.diag([1., -1., -1.])

        # projection operator Δ^μν
        Delta = g - uu

        # effective pressure = ideal + bulk = P + Π
        Peff = -np.einsum('au,bv,...ab,...uv', g, g, Delta, T)/3

        broadcast = np.index_exp[..., np.newaxis, np.newaxis]

        self._shear_tensor = T - e[broadcast]*uu + Peff[broadcast]*Delta

        self._bulk_pressure = Peff - P

    def shear_tensor(self, u=None, v=None, eos=ideal_eos):
        """
        Shear pressure tensor π^μν.

        """
        self._compute_viscous_corrections(eos)

        if u is None and v is None:
            return self._shear_tensor
        elif u is not None and v is not None:
            return self._shear_tensor[..., u, v]
        else:
            raise ValueError('must provide both u and v')

    def bulk_pressure(self, eos=ideal_eos):
        """
        Bulk viscous pressure Π.

        """
        self._compute_viscous_corrections(eos)

        return self._bulk_pressure
