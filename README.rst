freestream
==========
*Free streaming and Landau matching for boost-invariant hydrodynamic initial conditions.*

.. image:: http://giant.gfycat.com/AffectionateQuerulousAfricanwilddog.gif
   :target: http://gfycat.com/AffectionateQuerulousAfricanwilddog

|

``freestream`` is a Python implementation of pre-equilibrium free streaming for heavy-ion collisions, as described in

- J. Liu, C. Shen, U. Heinz,
  "Pre-equilibrium evolution effects on heavy-ion collision observables",
  `PRC 91 064906 (2015) <http://journals.aps.org/prc/abstract/10.1103/PhysRevC.91.064906>`_,
  `arXiv:1504.02160 [nucl-th] <http://inspirehep.net/record/1358669>`_.
- W. Broniowski, W. Florkowski, M. Chojnacki, A. Kisiel,
  "Free-streaming approximation in early dynamics of relativistic heavy-ion collisions",
  `PRC 80 034902 (2009) <http://journals.aps.org/prc/abstract/10.1103/PhysRevC.80.034902>`_,
  `arXiv:0812.3393 [nucl-th] <http://inspirehep.net/record/805616>`_.

Installation
------------
Simply run ::

   pip install freestream

The only requirements are numpy (1.8.0 or later) and scipy (0.14.0 or later).

Usage
-----
``freestream`` has an object-oriented interface through the ``FreeStreamer`` class, which takes three parameters:

.. code-block:: python

   freestream.FreeStreamer(initial, grid_max, time)

where

- ``initial`` is a square array containing the initial state,
- ``grid_max`` is the *x* and *y* maximum of the grid in fm, i.e. half the grid width (see following example),
- ``time`` is the time to free stream in fm/c.

The ``initial`` array must contain a two-dimensional (boost-invariant) initial condition discretized onto a uniform square grid.
It is then interpreted as a density profile of non-interacting massless partons at time *τ* = 0+.

The ``grid_max`` parameter sets the outermost *edge* of the grid, *not* the midpoint of the outer grid cell, e.g.

- A 200 × 200 grid with a max of 10.0 fm has cell edges at -10.00, -9.90, ..., +10.00 and cell midpoints at -9.95, -9.85, ..., +9.95.
- A 201 × 201 grid with a max of 10.05 fm has cell edges at -10.05, -9.95, ..., +10.05 and cell midpoints at -10.00, -9.90, ..., +10.00.

This is the same definition as the `trento <https://github.com/Duke-QCD/trento>`_ ``--grid-max`` parameter.

**It is very important that the grid max is set correctly to avoid superluminal propagation.**

Suppose ``initial`` is an *n* × *n* initial condition array with a grid max of 10.0 fm and we want to free stream for 1.0 fm.
We first create a ``FreeStreamer`` object:

.. code-block:: python

   import freestream

   fs = freestream.FreeStreamer(initial, 10.0, 1.0)

We can now extract the various quantities needed to initialize hydro from ``fs``.

Energy-momentum tensor *T*\ :sup:`μν`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   Tuv = fs.Tuv()

``Tuv`` is an *n* × *n* × 3 × 3 array containing the full tensor at each grid point.
If we only want a certain component of the tensor, we can pass indices to the function:

.. code-block:: python

   T00 = fs.Tuv(0, 0)

``T00`` is an *n* × *n* array containing *T*\ :sup:`00` at each grid point.
This is purely for syntactic convenience: ``fs.Tuv(0, 0)`` is equivalent to ``fs.Tuv()[:, :, 0, 0]``.

Energy density *e* and flow velocity *u*\ :sup:`μ`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   e = fs.energy_density()  # n x n
   u = fs.flow_velocity()  # n x n x 3

We can also extract the individual components of flow velocity:

.. code-block:: python

   u1 = fs.flow_velocity(1)  # n x n

Again, this is equivalent to ``fs.flow_velocity()[:, :, 1]``.

Shear tensor π\ :sup:`μν` and bulk pressure Π
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The shear pressure tensor π\ :sup:`μν` works just like *T*\ :sup:`μν`:

.. code-block:: python

   pi = fs.shear_tensor()  # n x n x 3 x 3
   pi01 = fs.shear_tensor(0, 1)  # n x n

The bulk viscous pressure Π depends on the equation of state *P(e)*.
By default, the ideal EoS *P(e)* = *e*/3 is used:

.. code-block:: python

   bulk = fs.bulk_pressure()

The bulk pressure is in fact zero with the ideal EoS, but there will be small nonzero values due to numerical precision.

To use another EoS, pass a callable object to ``bulk_pressure()``:

.. code-block:: python

   bulk = fs.bulk_pressure(eos)

For example, suppose we have a table of pressure and energy density we want to interpolate.
We can use ``scipy.interpolate`` to construct a spline and pass it to ``bulk_pressure()``:

.. code-block:: python

   import scipy.interpolate as interp

   eos_spline = interp.InterpolatedUnivariateSpline(energy_density, pressure)
   bulk = fs.bulk_pressure(eos_spline)

Other notes
~~~~~~~~~~~
The code should run in a few seconds, depending on the grid size.
Computation time is proportional to the number of grid cells (i.e. *n*\ :sup:`2`).

Ensure that the grid is large enough to accommodate radial expansion.
The code does not check for overflow.

``FreeStreamer`` returns references to its internal arrays, so do not modify them in place—make copies!

Testing and internals
---------------------
``FreeStreamer`` uses a two-dimensional cubic spline (`scipy.interpolate.RectBivariateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html>`_) to construct a continuous initial condition profile from a discrete grid.
This is very precise provided the grid spacing is small enough.
The spline sometimes goes very slightly negative around sharp boundaries; ``FreeStreamer`` coerces these negative values to zero.

The script ``test.py`` contains unit tests and generates visualizations for qualitative inspection.
To run the tests, install nose and run::

   nosetests -v test.py

There are two unit tests:

- Comparison against an analytic solution for a symmetric Gaussian initial state (computed in Mathematica).
- Comparison against a randomly-generated initial condition without interpolation.

These tests occasionally fail since there is a random component and the tolerance is somewhat stringent (every grid point must agree within 0.1%).
When a test fails, it will print out a list of ratios (observed/expected).
Typically the failures occur at the outermost grid cell where the system is very dilute, and even there it will only miss by ~0.2%.

To generate visualizations, execute ``test.py`` as a script with two arguments, the test case to visualize and a PDF output file.
There are three test cases:

- ``gaussian1``, a narrow symmetric Gaussian centered at the origin.
- ``gaussian2``, a wider asymmetric Gaussian offset from the origin.
- ``random``, a randomly-generated initial condition (this is not in any way realistic, it's only for visualization).

For example::

   python test.py gaussian1 freestream.pdf

will run the ``gaussian1`` test case and save results in ``freestream.pdf``.
The PDF contains visualizations of the initial state and everything that ``FreeStreamer`` computes.
In each visualization, red colors indicate positive values, blue means negative, and the maximum absolute value of the array is annotated in the upper left.

Animations
----------
The included script ``animate.py`` generates animations (like the one at the top of this page) from initial conditions saved in HDF5 format (e.g. `trento <https://github.com/Duke-QCD/trento>`_ events).
It requires python3 with matplotlib and h5py, and of course ``freestream`` must be installed.
To animate a trento event, first generate some events in HDF5 format then run the script::

   trento Pb Pb 10 -o events.hdf
   ./animate.py events.hdf event_0 freestream.mp4

The first argument is the HDF5 filename, the second is the dataset to animate, and the last is the animation filename.
Run ``./animate.py --help`` for more information including options for the animation duration, framerate, colormap, etc.
