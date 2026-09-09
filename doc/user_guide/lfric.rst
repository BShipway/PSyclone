.. -----------------------------------------------------------------------------
.. BSD 3-Clause License
..
.. Copyright (c) 2017-2026, Science and Technology Facilities Council
.. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are met:
..
.. * Redistributions of source code must retain the above copyright notice, this
..   list of conditions and the following disclaimer.
..
.. * Redistributions in binary form must reproduce the above copyright notice,
..   this list of conditions and the following disclaimer in the documentation
..   and/or other materials provided with the distribution.
..
.. * Neither the name of the copyright holder nor the names of its
..   contributors may be used to endorse or promote products derived from
..   this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
.. FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
.. COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
.. INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
.. BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
.. LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
.. CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
.. LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
.. ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. POSSIBILITY OF SUCH DAMAGE.
.. -----------------------------------------------------------------------------
.. Written by R. W. Ford and A. R. Porter, STFC Daresbury Lab
.. Modified by I. Kavcic, A. Coughtrie, O. Brunt and L. Turner, Met Office

.. highlight:: fortran

.. _lfric-api:

The LFRic DSL
=============

This section describes the LFRic application programming
interface (API). This API explains what a user needs to write in order
to make use of the LFRic API in PSyclone.

As with the majority of PSyclone APIs, the LFRic API specifies
how a user needs to write the algorithm layer and the kernel layer to
allow PSyclone to generate the PSy layer. These algorithm and kernel
APIs are discussed separately in the following sections.

The LFRic API supports the Met Office's finite element (hereafter FEM)
based 'GungHo' dynamical core.
This dynamical core with atmospheric physics parameterisation
schemes is a part of the Met Office LFRic modelling system :footcite:t:`lfric-2019`,
currently being developed in preparation for exascale computing in the 2020s.
The LFRic repository and the associated wiki are hosted at the `Met Office
Science Repository Service <https://code.metoffice.gov.uk/trac/home>`_.
The code is BSD-licensed, however browsing the `LFRic wiki
<https://code.metoffice.gov.uk/trac/lfric/wiki>`_ and
`code repository <https://code.metoffice.gov.uk/trac/lfric/browser>`_
requires login access to MOSRS. For more technical details on the
implementation of LFRic, please see the `LFRic documentation
<https://code.metoffice.gov.uk/trac/lfric/attachment/wiki/LFRicDocumentationPapers/lfric_documentation.pdf>`_.

.. note::
   The following sections assume that the reader is familiar
   with various concepts relating to the LFRic mesh and how it is decomposed
   for distributed-memory parallel computing. For a detailed description of
   these things, please see the :ref:`LFRic <lfric-developers>` section of the
   Developer Guide.

.. _lfric-api-algorithm:

Algorithm
---------

The general requirements for the structure of an Algorithm are explained
in the :ref:`algorithm-layer` section. This section explains the
LFRic-API-specific specialisations and extensions.

The LFRic API defines a set of objects, with specific meanings and
data-structures, that can be provided as arguments to Kernels within
invoke calls. These are: :ref:`scalar <lfric-scalar>`,
:ref:`field <lfric-field>`, :ref:`field vector <lfric-field-vector>`,
:ref:`operator <lfric-operator>`,
:ref:`column-wise operator <lfric-cma-operator>`,
:ref:`Quadrature <lfric-quadrature>`,
:ref:`Halo Depth <lfric-halo-depth>` and
:ref:`Stencil Extents <lfric-alg-stencil>`. The example below showcases the
use of each of these arguments:

::


  real(kind=r_def)                         :: rscalar
  integer(kind=i_def)                      :: iscalar, halo_depth
  logical(kind=l_def)                      :: lscalar
  real(kind=r_def),    dimension(50, 100)  :: real_array
  integer(kind=i_def), dimension(10)       :: integer_array
  logical(kind=l_def), dimension(2,5,10,8) :: logical_array
  integer(kind=i_def)                      :: stencil_extent
  type(field_type)                         :: field1, field2, field3
  type(field_type)                         :: field5(3), field6(3)
  type(integer_field_type)                 :: field7
  type(quadrature_type)                    :: qr
  type(operator_type)                      :: operator1
  type(columnwise_operator_type)           :: cma_op1
  ...
  call invoke( kernel1(field1, field2, operator1, real_array, qr),  &
               builtin1(rscalar, field2, field3),                   &
               int_builtin2(iscalar, field7),                       &
               kernel2(field1, stencil_extent, field3, lscalar),    &
               assembly_kernel(cma_op1, operator1),                 &
	             kernel3(field1, halo_depth),                         &
               kernel4(field3, integer_array, logical_array),       &
               name="some_calculation"                              &
             )


Each of these argument types is described in more detail in
the next :ref:`section <lfric-alg-arg-types>`.

The LFRic API has support for inter-grid kernels (those that
map fields between grids of different resolution). At the Algorithm
layer, an ``invoke`` of such kernels looks much like an
``invoke`` containing general-purpose kernels. The only restrictions to be
aware of are that inter-grid kernels accept only field or field-vectors
as arguments and that an ``invoke`` may not mix inter-grid kernels with
any other kernel type.

.. _lfric-alg-arg-types:

Algorithm Argument Types
------------------------

.. _lfric-scalar:

Scalar
++++++

In the LFRic API a scalar is a single-valued argument that is identified
with ``GH_SCALAR`` metadata. Scalar arguments can have ``real``,
``integer`` or ``logical`` data type in :ref:`user-defined Kernels
<lfric-kernel-valid-data-type>` (``logical`` data type is not supported
in the :ref:`LFRic Built-ins <lfric-built-ins-dtype-access>`).

.. _lfric-array:

Scalar Array
++++++++++++

In the LFRic API a scalar array represents a Fortran array of scalars, of at
least rank (number of dimensions) one. Scalar arrays are identified with
``GH_SCALAR_ARRAY`` metadata. As with scalars, array arguments can have
``real``, ``integer`` or ``logical`` data type in
:ref:`user-defined Kernels <lfric-kernel-valid-data-type>`.

.. _lfric-field:

Field
+++++

LFRic API fields, identified with ``GH_FIELD`` metadata, represent
FEM discretisations of various dynamical core prognostic and diagnostic
variables. In FEM, variables are discretised by placing them into a
function space (see :ref:`lfric-function-space`) from which they
inherit a polynomial expansion via the basis functions of that space.
Field values at points within a cell are evaluated as the sum of a set
of basis functions multiplied by coefficients which are the data points.
Points of evaluation are determined by a quadrature object
(:ref:`lfric-quadrature`) and are independent of the function space
the field is on. Placement of field data points, also called degrees of
freedom (hereafter "DoFs"), is determined by the function space the field
is on.
LFRic fields passed as arguments to any :ref:`LFRic kernel
<lfric-kernel-valid-data-type>` can be of ``real`` or ``integer``
primitive type. In the LFRic infrastructure, these fields are
represented by instances of the ``field_type`` and ``integer_field_type``
classes, respectively.

.. _lfric-field-vector:

Field Vector
++++++++++++

Depending on the function space a field lives on, the field data value at
a point can be a scalar or a vector (see :ref:`lfric-function-space`
for the list of scalar and vector function spaces). There is an
additional option, called a *field vector*, to represent a bundle of
either scalar- or vector-valued fields.
Field vectors are represented as ``GH_FIELD*N`` where ``N`` is the
size of the vector. The 3D coordinate field, for example, has
``(x, y, z)`` scalar values at the nodes and therefore has a
vector size of 3.

.. _lfric-operator:

Operator
++++++++

Represents a matrix constructed on a per-cell basis using Local
Matrix Assembly (LMA) and is identified with ``GH_OPERATOR``
metadata. In the LFRic infrastructure, operators are represented by
instances of the ``operator_type`` class. LFRic operators can only
have ``real``-valued data in :ref:`user-defined Kernels
<lfric-kernel-valid-data-type>` (:ref:`LFRic Built-ins <lfric-built-ins>`
do not currently support operators).

.. _lfric-cma-operator:

Column-wise Operator
++++++++++++++++++++

The LFRic API has support for the construction and use of
column-wise/Column Matrix Assembly (CMA) operators whose metadata
identifier is ``GH_COLUMNWISE_OPERATOR``. In the LFRic
infrastructure, column-wise operators are represented by instances
of the ``columnwise_operator_type`` class. As for the LMA operators
above, LFRic column-wise operators can only have ``real``-valued
:ref:`data <lfric-kernel-valid-data-type>`.

As the name suggests, these are operators constructed for a whole
column of the mesh. These are themselves constructed from the
Local Matrix Assembly (LMA) operators of each cell in the column.
The rules governing Kernels that have CMA operators as arguments
are given in the :ref:`lfric-kernel` section below.

There are three recognised Kernel types involving CMA operations;
construction, application (including inverse application) and
matrix-matrix. The following example sketches-out what the use
of such kernels might look like in the Algorithm layer::

  use field_mod, only: field_type
  use operator_mod, only : operator_type
  use columnwise_operator_mod, only : columnwise_operator_type
  type(field_type) :: field1, field2, field3
  type(operator_type) :: lma_op1, lma_op2
  type(columnwise_operator_type) :: cma_op1, cma_op2, cma_op3
  real(kind=r_def) :: alpha
  ...
  call invoke(                                                    &
          assembly_kernel(cma_op1, lma_op1, lma_op2),             &
          assembly_kernel2(cma_op2, lma_op1, lma_op2, field3),    &
          apply_kernel(field1, field2, cma_op1),                  &
          matrix_matrix_kernel(cma_op3, cma_op1, alpha, cma_op2), &
          apply_kernel(field3, field1, cma_op3),                  &
          name="cma_example")

The above invoke uses two LMA operators to construct the CMA operator
``cma_op1``.  A second CMA operator, ``cma_op2``, is assembled from
the same two LMA operators but also uses a field. The first of these
CMA operators is then applied to ``field2`` and the result stored in
``field1`` (assuming that the metadata for ``apply_kernel`` specifies
that it is the first field argument that is written to). The two CMA
operators are then combined to produce a third, ``cma_op3``. This is
then applied to ``field1`` and the result stored in ``field3``.

Note that PSyclone identifies the type of kernels performing
column-wise operations based on their arguments as described in
metadata (see :ref:`lfric-cma-mdata-rules` below). The names of the
kernels in the above example are purely illustrative and are not used
by PSyclone when determining kernel type.

A full example of CMA operator construction is available in
``examples/lfric/eg7``.

.. _lfric-quadrature:

Quadrature
++++++++++

Kernels conforming to the LFRic API may require quadrature
information (specified using e.g. ``gh_shape = gh_quadrature_XYoZ`` in
the kernel metadata - see Section :ref:`lfric-gh-shape`). This
information must be passed to the kernel from the Algorithm layer in
the form of one or more ``quadrature_type`` objects. These must be the
last arguments passed to the kernel (with the exception of ``halo_depth``
- :ref:`lfric-halo-depth` - if the kernel requires it) and must be
provided in the same
order that they are specified in the kernel metadata, e.g. if the
metadata for kernel ``pressure_gradient_kernel_type`` specified
``gh_shape = gh_quadrature_XYoZ`` and that for kernel
``geopotential_gradient_kernel`` had ``gh_shape(2) = (\
gh_quadrature_XYoZ, gh_quadrature_face \)`` then the corresponding
invoke would look something like::

      ...
      qr_xyoz = quadrature_xyoz_type(nqp_h_exact, nqp_h_exact, nqp_v_exact, rule)
      qr_face = quadrature_face_type(nqp_h_exact, nqp_v_exact, ..., rule)
      call invoke(pressure_gradient_kernel_type(rhs_tmp(igh_u), rho, theta, qr_xyoz), &
                  geopotential_gradient_kernel_type(rhs_tmp(igh_u), geopotential, &
                                                    qr_xyoz, qr_face))

These quadrature objects specify the set(s) of points at which the
basis/differential-basis functions required by the kernel are to be evaluated.

The arguments ``nqp_h_exact`` and ``nqp_v_exact`` are namelist options from the
finite-element configuration file, generated as a function of the
finite-element model (FEM) order in the horizontal and vertical directions (see
:ref:`lfric-function-space`). They specify the number of quadrature
points in the horizontal and vertical directions, respectively.

Here, ``qr_face`` has been passed the horizontal and vertical numbers of
quadrature points, meaning that the face in question is normal to a horizontal
plane. In general, ``qr_xyoz`` takes the number of quadrature points in each
direction (*x*, *y*, and *z*, in that order), whereas ``qr_face`` takes the
number of quadrature points in each direction tangential to the face in
question.

.. _lfric-halo-depth:

Halo Depth
++++++++++

If a Kernel is written such that it *must* iterate into the halo (has an
:ref:`OPERATES_ON <lfric-operates-on>` of ``HALO_CELL_COLUMN`` or
``OWNED_AND_HALO_CELL_COLUMN``) then the halo depth must be passed as a
final, ``integer`` argument to the Kernel.

.. _lfric-alg-stencil:

Stencil Extent
++++++++++++++

The metadata for a Kernel which operates on a cell-column may specify
that a Kernel performs a stencil operation on a field. Any such
metadata must provide a stencil type. See the
:ref:`lfric-api-meta-args` section for more details. The supported
stencil types are ``X1D``, ``Y1D``, ``XORY1D``, ``CROSS``, ``CROSS2D`` or
``REGION``.

If a stencil operation is specified by the Kernel metadata, the
Algorithm layer must provide the ``extent`` of the stencil (the
maximum distance from the central cell that the stencil extends). The
LFRic API expects this information to be added as an additional
``integer`` argument immediately after the relevant field when specifying
the Kernel via an ``invoke``.

For example::

  integer(kind=i_def) :: extent = 2
  call invoke(kernel(field1, field2, extent))

where ``field2`` has kernel metadata specifying that it has a stencil
access.

``extent``  may also be passed as a literal. For example::

  call invoke(kernel(field1, field2, 2))

where, again, ``field2`` has kernel metadata specifying that it has a
stencil access.

.. note:: The stencil extent specified in the Algorithm layer is not the
          same as the stencil size passed in to the Kernel. The latter
          contains the number of cells in the stencil which is dependent
          on both the stencil type and extent.

If the Kernel metadata specifies that the stencil is of type
``XORY1D`` (which means ``X1D`` or ``Y1D``) then the algorithm layer
must specify whether the stencil is ``X1D`` or ``Y1D`` for that
particular kernel call. The LFRic API expects this information to
be added as an additional argument immediately after the relevant
stencil extent argument. The argument should be an ``integer`` with
valid values being ``x_direction`` or ``y_direction``, both being
supplied by the ``LFRic`` infrastructure via the
``flux_direction_mod`` fortran module

For example::

  use flux_direction_mod, only : x_direction
  integer(kind=i_def) :: direction = x_direction
  integer(kind=i_def) :: extent = 2
  ! ...
  call invoke(kernel(field1, field2, extent, direction))

``direction`` may also be passed as a literal. For example::

  use flux_direction_mod, only : x_direction
  integer(kind=i_def) :: extent = 2
  ! ...
  call invoke(kernel(field1, field2, extent, x_direction))

If the stencil is of type ``CROSS2D`` then the arrays passed to the kernel are
of different dimensions to those of other stencils. The ``CROSS2D`` stencil is
designed for use when it is necessary for a kernel to know where the stencil
cells are, relative to the current cell. For this reason, the ``stencil_size``
passed to the kernel is an array of length 4 containing sizes for each branch
of the stencil. The ``stencil_size`` array is always ordered: West, South,
East, North. This branch dimension is also part of the ``stencil_dofmap`` array
making it possible to loop over each branch of the stencil individually. The
invoke call for the ``CROSS2D`` stencil remains of the same form as for other
stencils.

If certain fields use the same value of extent and/or direction then
the same variable, or literal value can be provided.

For example::

  call invoke(kernel1(field1, field2, extent,  field3, extent, direction), &
              kernel2(field1, field2, extent2, field4, extent, direction))

In the above example ``field2`` and ``field3`` in ``kernel1`` and
``field4`` in ``kernel2`` will have the same ``extent`` value but
``field2`` in ``kernel2`` may have a different value. Similarly,
``field3`` in ``kernel1`` and ``field4`` in ``kernel2`` will have the
same ``direction`` value.

An example of the use of stencils is available in ``examples/lfric/eg5``.

There is currently no attempt to perform type checking in PSyclone so
any errors in the type and/or position of arguments will not be picked
up until compile time. However, PSyclone does check for the correct
number of algorithm arguments. If the wrong number of arguments is
provided then an exception is raised.

For example, running test 19.2 from the LFRic API test suite gives:

.. code-block:: bash

  cd <PSYCLONEHOME>/src/psyclone/tests
  psyclone test_files/lfric/19.2_single_stencil_broken.f90
  "Generation Error: error: expected '5' arguments in the algorithm layer but found '4'.
  Expected '4' standard arguments, '1' stencil arguments and '0' qr_arguments'"

.. _lfric-mixed-precision:

Mixed Precision
---------------

The LFRic API supports the ability to specify the precision required
by the model via precision variables. To make use of this, the code
developer must declare scalars, arrays, fields and operators in the algorithm
layer with the required LFRic-supported precision. In the current
implementation there are two supported precisions for ``REAL`` data and
one each for ``INTEGER`` and ``LOGICAL`` data. The actual precision used in
the code can be set in a configuration file. For example, ``INTEGER`` data
could be set to be 32-bit precision. As ``REAL`` data has more than one
supported precision, different parts of the code can be configured to
have different precision.

The table below gives the currently supported datatypes, their
associated kernel metadata description and their precision:

.. tabularcolumns:: |l|l|l|

+--------------------------+---------------------------------------+-----------+
| Data Type                | Kernel Metadata                       | Precision |
+==========================+=======================================+===========+
| REAL(R_DEF)              | GH_SCALAR/GH_SCALAR_ARRAY, GH_REAL    | R_DEF     |
+--------------------------+---------------------------------------+-----------+
| REAL(R_BL)               | GH_SCALAR/GH_SCALAR_ARRAY, GH_REAL    | R_BL      |
+--------------------------+---------------------------------------+-----------+
| REAL(R_PHYS)             | GH_SCALAR/GH_SCALAR_ARRAY, GH_REAL    | R_PHYS    |
+--------------------------+---------------------------------------+-----------+
| REAL(R_SOLVER)           | GH_SCALAR/GH_SCALAR_ARRAY, GH_REAL    | R_SOLVER  |
+--------------------------+---------------------------------------+-----------+
| REAL(R_TRAN)             | GH_SCALAR/GH_SCALAR_ARRAY, GH_REAL    | R_TRAN    |
+--------------------------+---------------------------------------+-----------+
| INTEGER(I_DEF)           | GH_SCALAR/GH_SCALAR_ARRAY, GH_INTEGER | I_DEF     |
+--------------------------+---------------------------------------+-----------+
| LOGICAL(L_DEF)           | GH_SCALAR/GH_SCALAR_ARRAY, GH_LOGICAL | L_DEF     |
+--------------------------+---------------------------------------+-----------+
| FIELD_TYPE               | GH_FIELD, GH_REAL                     | R_DEF     |
+--------------------------+---------------------------------------+-----------+
| R_BL_FIELD_TYPE          | GH_FIELD, GH_REAL                     | R_BL      |
+--------------------------+---------------------------------------+-----------+
| R_SOLVER_FIELD_TYPE      | GH_FIELD, GH_REAL                     | R_SOLVER  |
+--------------------------+---------------------------------------+-----------+
| R_TRAN_FIELD_TYPE        | GH_FIELD, GH_REAL                     | R_TRAN    |
+--------------------------+---------------------------------------+-----------+
| INTEGER_FIELD_TYPE       | GH_FIELD, GH_INTEGER                  | I_DEF     |
+--------------------------+---------------------------------------+-----------+
| OPERATOR_TYPE            | GH_OPERATOR, GH_REAL                  | R_DEF     |
+--------------------------+---------------------------------------+-----------+
| R_SOLVER_OPERATOR_TYPE   | GH_OPERATOR, GH_REAL                  | R_SOLVER  |
+--------------------------+---------------------------------------+-----------+
| R_TRAN_OPERATOR_TYPE     | GH_OPERATOR, GH_REAL                  | R_TRAN    |
+--------------------------+---------------------------------------+-----------+
| COLUMNWISE_OPERATOR_TYPE | GH_COLUMNWISE_OPERATOR, GH_REAL       | R_SOLVER  |
+--------------------------+---------------------------------------+-----------+

As can be seen from the above table, the kernel metadata does not
capture all of the precision options. For example, from the metadata
it is not possible to determine whether a ``REAL`` scalar, ``REAL`` field
or ``REAL`` operator has precision ``R_DEF``, ``R_SOLVER`` or ``R_TRAN``.

If a scalar, array, field, or operator is specified with a particular
precision in the algorithm layer then any associated kernels that it
is passed to must have been written so that they support this
precision. If a kernel needs to support data that can be stored with
different precisions then appropriate precision-specific subroutines
should be written. These precision-specific subroutine should be
called via a generic interface (which lets Fortran choose the
appropriate subroutine based on the precision of its argument(s)).

Below is a simple example of an algorithm code calling the same
generic kernel twice with potentially different precision. The
implementation of the generic kernel such that it supports both 32-
and 64-bit precision is also shown. The use of LFRic names for
precision in the algorithm code allows precision to be controlled in a
simple way. For example, ``r_solver`` could be set to be 32-bits in
one configuration and 64-bits in another:

.. code-block:: fortran

  program test

    use constants_mod,      only : r_def, r_solver
    use field_mod,          only : field_type
    use r_solver_field_mod, only : r_solver_field_type
    use example_mod,        only : example_type

    type(field_type)          :: field_r_def
    type(r_solver_field_type) :: field_r_solver
    real(kind=r_def)          :: x_r_def
    real(kind=r_solver)       :: x_r_solver

    call invoke( example_type(field_r_def, x_r_def), &
                 example_type(field_r_solver, x_r_solver))

  end program test

  module example_mod

    use argument_mod
    use kernel_mod

    implicit none

    type, extends(kernel_type) :: example_type
      type(arg_type), dimension(2) :: meta_args = (/       &
           arg_type(gh_field,  gh_real, gh_readwrite, w3), &
           arg_type(gh_scalar, gh_real, gh_read )          &
           /)
       integer :: operates_on = cell_column
     contains
       procedure, nopass :: code => example_code
    end type example_type

    private
    public :: example_code

    interface example_code
      module procedure example_code_32
      module procedure example_code_64
    end interface example_code

  contains

    subroutine example_code_32(..., field1, x, ...)
      real*4, dimension(...), intent(inout) :: field1
      real*4, intent(in) :: x
      print *, "32-bit example called"
    end subroutine example_code_32

    subroutine example_code_64(..., field1, x, ...)
      real*8, dimension(...), intent(inout) :: field1
      real*8, intent(in) :: x
      print *, "64-bit example called"
    end subroutine example_code_64

  end module example_mod

In order to support mixed precision, PSyclone needs to know the
precision (as specified in the algorithm layer) of any kernel
arguments that are of a type that supports different precisions (e.g.
``GH_FIELD``). The reason for this is that PSyclone needs to be able
to declare data with the correct precision information within the
PSy-layer to ensure that the correct flavour of kernels are called.

PSyclone must therefore determine this information from the algorithm
layer. The rules for whether PSyclone requires information for
particular LFRic datatypes and what it does with or without this
information are given below:

.. _lfric-mixed-precision-fields:

Fields
++++++

PSyclone must be able to determine the datatype of a field from the
algorithm layer declarations. If it is not able to do this, PSyclone
will abort with a message that indicates the problem.

Supported field types, their Fortran datatype and precisions are
outlined in the table below:

.. tabularcolumns:: |l|l|l|

+-------------------------+------------------+--------------+
| Field Type              | Fortran Datatype | Precision    |
+=========================+==================+==============+
| ``field_type``          | ``real``         | ``r_def``    |
+-------------------------+------------------+--------------+
| ``r_bl_field_type``     | ``real``         | ``r_bl``     |
+-------------------------+------------------+--------------+
| ``r_solver_field_type`` | ``real``         | ``r_solver`` |
+-------------------------+------------------+--------------+
| ``r_tran_field_type``   | ``real``         | ``r_tran``   |
+-------------------------+------------------+--------------+
| ``integer_field_type``  | ``integer``      | ``i_def``    |
+-------------------------+------------------+--------------+

.. _lfric-mixed-precision-field-vectors:

Field Vectors
+++++++++++++

In addition to fields, LFRic supports an abstract vector type for fields,
used in the LFRic solver API. Please note that these structures are
different from the :ref:`field vector <lfric-field-vector>` implementation
of field bundles in the PSyclone LFRic API interface.

The LFRic abstract vector type has precision-specific implementations.
If PSyclone finds such a specifically declared field vector argument in the
algorithm layer, e.g. ``r_solver_field_vector_type``, it will assume that
the actual field being referenced is of the same datatype and precision
(see :ref:`above <lfric-mixed-precision-fields>` for details).
The correspondence between the available field types and their vector
implementations is given in the table below (note that only
``real``-valued fields have abstract vector implementations for now):

.. tabularcolumns:: |l|l|

+-------------------------+--------------------------------+
| Field Type              | Field Vector Type              |
+=========================+================================+
| ``field_type``          | ``field_vector_type``          |
+-------------------------+--------------------------------+
| ``r_bl_field_type``     | ``r_bl_field_vector_type``     |
+-------------------------+--------------------------------+
| ``r_solver_field_type`` | ``r_solver_field_vector_type`` |
+-------------------------+--------------------------------+
| ``r_tran_field_type``   | ``r_tran_field_vector_type``   |
+-------------------------+--------------------------------+

If PSyclone finds an argument that is declared as an
``abstract_field_type`` then it will not know the actual type of the
argument. For instance, the following algorithm layer code will cause
PSyclone to raise an exception::

    ! ...
    class (abstract_vector_type), intent(inout) :: x
    ! ...
    select type (x)
    type is (field_vector_type)
      call invoke(testkern_type(x%vector(1)))
    class default
      print *,"Error"
    end select
    ! ...

The suggested solution to this is to add a pointer variable to the
code that is of the required type. This pointer can then be associated
with the argument and passed into the routine::

    ! ...
    class (abstract_vector_type), target, intent(inout) :: x
    type(field_vector_type), pointer :: x_ptr
    ! ...
    select type (x)
    type is (field_vector_type)
      x_ptr => x
      call invoke(testkern_type(x_ptr%vector(1)))
    class default
      print *,"Error"
    end select
    ! ...

.. _lfric-mixed-precision-scalars:

Scalars
+++++++

It is not mandatory for PSyclone to be able to determine the datatype
of a scalar from the algorithm layer. This constraint was considered
to be too restrictive as PSyclone currently only examines the
declarations in the same source file as the ``invoke`` when
determining datatype. This means that if scalars are imported from
other modules (as is often the case) then their datatype cannot be
determined.

If the precision information for a scalar is found by PSyclone then
this is used. If the scalar declaration is found and it contains no
precision information then PSyclone will abort with a message that
indicates the problem (since this violates LFRic coding standards). If
no declaration information is found then default precision values are
used, as specified in the PSyclone config file (``r_def`` for ``real``,
``i_def`` for ``integer`` and ``l_def`` for ``logical``).

Supported precisions for scalars are outlined in the table below. If
an unsupported scalar precision is found then PSyclone will abort with
a message that indicates the problem.

.. tabularcolumns:: |l|l|

+------------------+--------------------------+
| Fortran Datatype | Supported Precision      |
+==================+==========================+
| ``real``         | ``r_def``, ``r_bl``,     |
|                  | ``r_solver``, ``r_tran`` |
+------------------+--------------------------+
| ``integer``      | ``i_def``                |
+------------------+--------------------------+
| ``logical``      | ``l_def``                |
+------------------+--------------------------+

.. _lfric-mixed-precision-lma-operators:

LMA Operators
+++++++++++++

PSyclone must be able to determine the datatype of an LMA operator.
If it is not able to do this, PSyclone will abort with a message that
indicates the problem.

Supported LMA operator types, their Fortran datatype and precisions are
outlined in the table below:

.. tabularcolumns:: |l|l|l|

+----------------------------+------------------+--------------+
| Operator Type              | Fortran Datatype | Precision    |
+============================+==================+==============+
| ``operator_type``          | ``real``         | ``r_def``    |
+----------------------------+------------------+--------------+
| ``r_solver_operator_type`` | ``real``         | ``r_solver`` |
+----------------------------+------------------+--------------+
| ``r_tran_operator_type``   | ``real``         | ``r_tran``   |
+----------------------------+------------------+--------------+

.. _lfric-mixed-precision-cma-operators:

Column-wise Operators
+++++++++++++++++++++

It is not mandatory for PSyclone to be able to determine the datatype
of a column-wise (CMA) operator. The reason for this is that only one
datatype is supported, a ``columnwise_operator_type`` which contains
``real``-valued data with precision ``r_solver``. PSyclone can therefore
simply add this datatype in the PSy-layer. However, if the datatype
information is found in the algorithm layer and it is not of the
expected type then PSyclone will abort with a message that indicates
the problem.

.. _lfric-mixed-precision-consistency:

Consistency
+++++++++++

If PSyclone is able to determine the datatype of an LFRic datatype
then PSyclone also checks that this datatype is consistent with the
associated kernel metadata. If it is not consistent then PSyclone will
abort with a message that indicates the problem.

.. _lfric-psy:

PSy-layer
---------

The general details of the PSy-layer are explained in the
:ref:`PSy-layer` section. This section describes any LFRic-specific
issues.

Module name
+++++++++++

The PSy-layer code is contained within a Fortran module. The name of
the module is determined from the algorithm-layer name with "_psy"
appended. The algorithm-layer name is the algorithm's module name if it
is a module, its subroutine name if it is a subroutine that is not
within a module, or the program name if it is a program.

So, for example, if the algorithm code is contained within a module
called "fred" then the PSy-layer module name will be "fred_psy".

.. _lfric-psy-arg-intents:

Argument Intents
################

LFRic :ref:`fields <lfric-field>`, :ref:`field vectors
<lfric-field-vector>`, :ref:`operators <lfric-operator>` and
:ref:`column-wise operators <lfric-cma-operator>` are objects that
contain pointers to data rather than data. The data are accessed by proxies
of these objects and modified in :ref:`kernels <lfric-kernel>`.
As the objects themselves are not modified in the PSy layer, their Fortran
intents there are always ``intent(in)``.

The Fortran intent of :ref:`scalars <lfric-scalar>` is still defined
by their :ref:`access metadata <lfric-kernel-valid-access>` as they are
actual data. This means ``intent(in)`` for ``GH_READ`` and ``intent(out)``
for ``GH_REDUCTION`` (more details in :ref:`meta_args <lfric-api-meta-args>`
section below).

The intent of other data structures is mandated by the relevant
LFRic API rules described in sections below.

.. _lfric-kernel:

Kernel
-------

The general requirements for the structure of a Kernel are explained
in the :ref:`kernel-layer` section. In the LFRic API there are six
different Kernel types; general purpose, CMA, inter-grid, domain, dof and
:ref:`lfric-built-ins`. In the case of built-ins, PSyclone generates
the source of the kernels.  This section explains the rules for the
other five user-supplied kernel types and then goes on to describe
their metadata and subroutine arguments.

Domain kernels are distinct from the other four user-supplied kernel
types because they must be passed data for the whole domain rather
than a single cell-column or dof. This permits the use of kernels that have
not been written to conform to the single-column/dof approach which
simplifies the integration with existing code. Obviously, any
parallelisation in the 'domain' kernel must be consistent with that
in the rest of the application. The motivation for such kernels in
LFRic is that they allow existing, "i-first" physics code to be called from
the PSy layer. Since those routines currently contain their own,
i-first looping structure (and associated OpenMP parallelisation), the
most efficient way to use them is to avoid enclosing them within a
loop in the PSy layer. This is a temporary measure and these
kernels will ultimately be replaced once the LFRic infrastructure has
support for i-first kernels
(https://code.metoffice.gov.uk/trac/lfric/ticket/2154). At that
point the looping (and associated parallelisation) will be put
back into the PSy layer.

.. _lfric-user-kernel-rules:

Rules for all User-Supplied Kernels that Operate on Cell-Columns
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In the following, 'operator' refers to both LMA and CMA operator
types.

1) A Kernel must have at least one argument that is a field, field
   vector, or operator. This rule reflects the fact that a Kernel
   operates on some subset of the whole domain (e.g. a cell-column)
   and is therefore designed to be called from within a loop that
   iterates over those subsets of the domain.

2) The continuity of the iteration space of the Kernel is determined
   from the function space of the modified argument (see Section
   :ref:`Supported Function Spaces <lfric-function-space>` below).
   If more than one argument is modified then the iteration space is taken
   to be the largest required by any of those arguments. E.g. if a Kernel
   writes to two fields, the first on ``W3`` (discontinuous) and the
   second on ``W1`` (continuous), then the iteration space of that Kernel
   will be determined by the field on the continuous space.

3) If any of the modified arguments are declared with the generic
   function space metadata (e.g. ``ANY_SPACE_<n>``, see
   :ref:`Supported Function Spaces <lfric-function-space>`)
   and their actual space cannot be determined statically then the
   iteration space is assumed to be

   1) discontinuous for ``ANY_DISCONTINUOUS_SPACE_<n>``;

   2) continuous for ``ANY_SPACE_<n>`` and ``ANY_W2``.  This assumption
      is always safe but leads to additional computation if the quantities
      being updated are actually on discontinuous function spaces.

4) Operators do not have halo operations operating on them as they
   are either cell- (LMA) or column-based (CMA) and therefore act
   like discontinuous fields.

5) Any Kernel that writes to an operator will have its iteration
   space expanded such that valid values for the operator are
   computed in the level-1 halo.

6) Any Kernel that reads from an operator must not access halos
   beyond level 1. In this case PSyclone will check that the Kernel
   does not require values beyond the level-1 halo. If it does then
   PSyclone will abort.

7) Any Kernel that takes an operator argument must not also take
   an ``integer``-valued field as an argument.

.. _lfric-no-cma-mdata-rules:

Rules specific to General-Purpose Kernels without CMA Operators
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1) General-purpose kernels that :ref:`operate on <lfric-operates-on>`
   any kind of ``CELL_COLUMN``
   accept arguments of any of the following types: field,
   field vector, LMA operator, scalar (``real``, ``integer`` or
   ``logical``).

2) A Kernel is permitted to write to more than one
   quantity (field or operator) and these quantities may be on the
   same or different function spaces.

3) A Kernel may not write to a scalar argument. (Only
   :ref:`built-ins <lfric-built-ins>` are permitted to do this.) Any
   scalar arguments must therefore be declared in the metadata as
   ``GH_READ`` - see :ref:`below <lfric-kernel-valid-access>`.

.. _lfric-cma-mdata-rules:

Rules for Kernels that work with CMA Operators
++++++++++++++++++++++++++++++++++++++++++++++

The LFRic API has support for kernels that assemble, apply (or
inverse-apply) column-wise/Column Matrix Assembly (CMA) operators.
Such operators may also be used by matrix-matrix kernels. There are
thus three types of CMA-related kernels.  Since, by definition, CMA
operators only act on data within a column, they have no horizontal
dependencies. Therefore, kernels that write to them may be
parallelised without colouring.

All three CMA-related kernel types must obey the following rules:

1) Since a CMA operator only acts within a single column of data,
   stencil operations are not permitted.

2) No vector quantities (e.g. ``GH_FIELD*3`` - see below) are
   permitted as arguments.

3) The kernel must operate on cell-columns.

There are then additional rules specific to each of the three
CMA kernel types. These are described below.

Assembly
########

CMA operators are themselves constructed from Local-Matrix-Assembly
(LMA) operators. Therefore, any kernel which assembles a CMA
operator must obey the following rules:

1) Have one or more LMA operators as read-only arguments.

2) Have exactly one CMA operator argument which must have write access.

3) Other types of argument (e.g. scalars or fields) are permitted but
   must be read-only.

Application and Inverse Application
###################################

Column-wise operators can only be applied to fields. CMA-Application
kernels must therefore:

1) Have a single CMA operator as a read-only argument.

2) Have exactly two field arguments, one read-only and one that is written to.

3) The function spaces of the read and written fields must match the
   from and to spaces, respectively, of the supplied CMA operator.

Matrix-Matrix
#############

A kernel that has just column-wise operators as arguments and zero or
more read-only scalars is identified as performing a matrix-matrix
operation. In this case:

1) Arguments must be CMA operators and, optionally, one or more scalars.

2) Exactly one of the CMA arguments must be written to while all other
   arguments must be read-only.

Rules for Inter-Grid Kernels
++++++++++++++++++++++++++++

1) An inter-grid kernel is identified by the presence of a field or
   field-vector argument with the optional ``mesh_arg`` metadata element (see
   :ref:`lfric-intergrid-mdata`).

2) An invoke that contains one or more inter-grid kernels must not contain
   any other kernel types. (This restriction is an implementation decision
   and could be lifted in future if there is a need.)

3) An inter-grid kernel is only permitted to have field or field-vector
   arguments.

4) All inter-grid kernel arguments must have the ``mesh_arg`` metadata entry.

5) An inter-grid kernel (and metadata) must have at least one field on
   each of the fine and coarse meshes. Specifying all fields as coarse or
   fine is forbidden.

6) Fields on different meshes must always live on different function spaces.

7) All fields on a given mesh must be on the same function space.

8) An inter-grid kernel must operate on cell-columns.

A consequence of Rules 5-7 is that an inter-grid kernel will
only involve two function spaces.

Rules for User-Supplied Kernels that Operate on the Domain
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The rules for kernels that have ``operates_on = DOMAIN`` are a subset
of :ref:`those <lfric-no-cma-mdata-rules>` for kernels that operate
on a ``CELL_COLUMN`` without CMA Operators. Specifically:

1) Only :ref:`scalar <lfric-scalar>`, :ref:`field <lfric-field>` and
   :ref:`field vector <lfric-field-vector>` arguments are permitted.

2) All fields must be on discontinuous function spaces.

3) Stencil accesses are not permitted.

.. _lfric-dof-kernel-rules:

Rules for all User-Supplied Kernels that Operate on DoFs (DoF Kernels)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Kernels that have an :ref:`operates_on <lfric-operates-on>` of ``DOF``
(or ``OWNED_DOF``) and :ref:`LFRic Built-ins<lfric-built-ins>` overlap
significantly in their scope, and the conventions that DoF Kernels
must follow are influenced by those for built-ins as a result. This
includes :ref:`metadata arguments <lfric-api-built-ins-metadata>` and
:ref:`valid data types and access
modes<lfric-built-ins-dtype-access>`. Naming conventions for DoF
Kernels should follow those for General-Purpose Kernels.

The list of rules for DoF Kernels is as follows:

1) A DoF Kernel must have at least one argument that is a field. This rule
   reflects that a Kernel operates on some subset of the whole domain
   and is therefore designed to be called from within a loop that iterates
   over those subsets of the domain. Fields (as opposed to e.g. operators)
   are accepted for DoF Kernels because only they have a single value at
   each DoF. Field vectors can be represented in a DoF kernel as a
   collection of field arguments, each one corresponding to an index in the
   field vector.

2) All Kernel arguments must be either fields or scalars (`real-` and/or
   `integer`-valued). DoF Kernels cannot accept operators.

3) All field arguments to a given DoF Kernel must be on the same function
   space so they have the same number of DoFs.

4) They must have at least one modified (i.e. written to) field argument. Unlike
   built-ins, this is not limited and more than one modified argument is
   allowed.

5) A Kernel may not write to a scalar argument. (Only built-ins are permitted
   to do this.) Any scalar arguments must therefore be declared in the metadata
   as `GH_READ` - see :ref:`below<lfric-kernel-valid-access>`

6) Kernels must be written to operate on a single DoF, such that field values
   at the same dof location/index can be provided to the Kernel within a loop
   over the DoFs of the function space of the field that is being updated.

.. _lfric-api-kernel-metadata:

Metadata
++++++++

The code below outlines the elements of the LFRic API Kernel
metadata, 1) 'meta_args_', 2) 'meta_funcs_', 3) 'meta_reference_element_',
4) 'meta_mesh_', 5) 'gh_shape' (`gh_shape and gh_evaluator_targets`_),
6) 'operates_on_' and 7) 'procedure_'::

  type, public, extends(kernel_type) :: my_kernel_type
    type(arg_type) :: meta_args(...) = (/ ... /)
    type(func_type) :: meta_funcs(...) = (/ ... /)
    type(reference_element_data_type) :: meta_reference_element(...) = (/ ... /)
    type(mesh_data_type) :: meta_mesh(...) = (/ ... /)
    integer :: gh_shape = gh_quadrature_XYoZ
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: my_kernel_code
  end type

These various metadata elements are discussed in order in the following
sections.

.. _lfric-api-meta-args:

meta_args
#########

The ``meta_args`` array specifies information about data that the
kernel code expects to be passed to it via its argument list. There is one
entry in the ``meta_args`` array for each **scalar**, **array**, **field**,
or **operator** passed into the Kernel and the order that these occur
in the ``meta_args`` array must be the same as they are expected in
the kernel code argument list. The entry must be of ``arg_type`` which
itself contains metadata about the associated argument. The size of the
``meta_args`` array must correspond to the number of **scalars**, **arrays**
**fields** and **operators** passed into the Kernel.

.. note:: It makes no sense for a Kernel to have only **scalar** or **array**
          arguments (because the PSy layer will call a Kernel for each point
          in the spatial domain) and PSyclone will reject such Kernels.

For example, if there are a total of 2 **scalar** / **array** / **field** /
**operator** entities being passed to the Kernel then the ``meta_args``
array will be of size 2 and there will be two ``arg_type`` entries::

  type(arg_type) :: meta_args(2) = (/                                  &
       arg_type( ... ),                                                &
       arg_type( ... )                                                 &
       /)

Argument metadata (information contained within the brackets of an
``arg_type`` entry), describes either a **scalar**, an **array**, a **field**
or an **operator** (either LMA or CMA).

The first argument-metadata entry describes whether the data that is
being passed is for a scalar (``GH_SCALAR``), an array (``GH_SCALAR_ARRAY``), a
field (``GH_FIELD``) or an operator (either ``GH_OPERATOR`` for LMA or
``GH_COLUMNWISE_OPERATOR`` for CMA). This information is mandatory.

Additionally, argument metadata can be used to describe a vector of
fields (see the :ref:`lfric-field-vector` section for more
details).

As an example, the following ``meta_args`` metadata describes 5
entries, the first is a scalar, the second is an array, the next two
are fields and the fifth is an operator. The fourth entry is a field vector
of size 3.

::

  type(arg_type) :: meta_args(5) = (/                                  &
       arg_type(GH_SCALAR, GH_REAL, ...),                              &
       arg_type(GH_SCALAR_ARRAY, GH_LOGICAL, ...),                     &
       arg_type(GH_FIELD, GH_INTEGER, ...),                            &
       arg_type(GH_FIELD*3, GH_REAL, ...),                             &
       arg_type(GH_OPERATOR, GH_REAL, ...)                             &
       /)

The second item in a metadata entry describes the Fortran primitive
(intrinsic) type of the data of a kernel argument. The currently supported
values are ``GH_REAL``, ``GH_INTEGER`` and ``GH_LOGICAL`` for ``real``,
``integer`` and ``logical`` data, respectively. This information is
mandatory. Valid data types for each LFRic API argument type are specified
later in this section (see :ref:`lfric-kernel-valid-data-type`).

The third component of argument metadata describes how the Kernel
makes use of the data being passed into it (the way it is accessed
within a Kernel). This information is mandatory. There are currently 6
possible values of this metadata ``GH_READ``, ``GH_WRITE``,
``GH_READWRITE``, ``GH_INC``, ``GH_READINC`` and ``GH_REDUCTION``. However,
not all combinations of metadata entries are valid and PSyclone will
raise an exception if an invalid combination is specified. Valid
combinations are specified later in this section (see
:ref:`lfric-kernel-valid-access`).

* ``GH_READ`` indicates that the data is read and is unmodified.

* ``GH_WRITE`` indicates the data is modified in the Kernel before
  (optionally) being read. If any shared DoFs are written to then
  different iterations of the Kernel must write the same value.

* ``GH_READWRITE`` indicates that different iterations of a Kernel
  update quantities which do not share DoFs, such as operators and
  fields over discontinuous function spaces. If a Kernel modifies only
  discontinuous fields and/or operators there is no need for
  synchronisation or colouring when running such Kernels in parallel.
  However, modifying another field with a ``GH_INC`` access in a
  Kernel means that synchronisation or colouring is required for
  parallel runs.

* ``GH_INC`` indicates that different iterations of a Kernel make
  contributions to shared values. For example, values at cell faces
  may receive contributions from cells on either side of the
  face. This means that such a Kernel needs appropriate
  synchronisation (or colouring) to run in parallel.

* ``GH_READINC`` indicates that the data is first read and then
  subsequently incremented. Therefore this is equivalent to a
  ``GH_READ`` followed by a ``GH_INC``.

* ``GH_REDUCTION`` indicates a reduction. Only Built-ins may perform
  reductions. The type of reduction (sum, maximum value, minimum value)
  is a property of the particular Built-in.

For example::

  type(arg_type) :: meta_args(7) = (/                                &
       arg_type(GH_OPERATOR,     GH_REAL,    GH_READ,      ... ),    &
       arg_type(GH_FIELD*3,      GH_REAL,    GH_WRITE,     ... ),    &
       arg_type(GH_FIELD,        GH_REAL,    GH_READWRITE, ... ),    &
       arg_type(GH_FIELD,        GH_INTEGER, GH_INC,       ... ),    &
       arg_type(GH_FIELD,        GH_REAL,    GH_READINC,   ... ),    &
       arg_type(GH_SCALAR_ARRAY, GH_LOGICAL, GH_READ,      ... ),    &
       arg_type(GH_SCALAR,       GH_REAL,    GH_REDUCTION)           &
       /)

.. warning:: It is important that ``GH_INC`` is not incorrectly used
             in place of a ``GH_READINC`` access as it could result in
             the reading of data from a dirty outermost halo when run
             in parallel, giving incorrect results. The reason for
             this is that PSyclone does not add a halo exchange for
             the outermost modified halo level of a field before a
             loop that contains a ``GH_INC`` access to that field,
             i.e. a loop iterating to the level-``n`` halo will result
             in a halo exchange to the level-(``n-1``) halo being
             added before the loop (which means no halo exchange is
             added when ``n==1``). The reason this can be performed is
             because any computation in the outermost halo will be
             incorrect (will only compute partial sums) and PSyclone
             therefore sets this halo level to dirty after the loop
             has completed. There is, therefore, no reason to make the
             values of the incremented field clean for the outermost
             modified halo. However, this optimisation does require
             that any (dirty) data in the outermost modified halo does
             not result in exceptions. With some compilers an
             exception can occur for a field that has not yet had its
             outermost halo data written to, i.e. if the uninitialised
             data is read. To avoid this potential problem in user
             code it is recommended that a redundant computation
             :ref:`transformation <lfric-api-transformations>`
             is added to compute all ``setval_c``, ``setval_x`` and
	     ``setval_random`` Built-in calls (see :ref:`lfric-built-ins`)
             to the same halo depth as the associated ``GH_INC``
             access - which is level-1 without any redundant
             computation transformations being applied to the
             associated loops. This will guarantee that all data has
             been initialised with a value before it is incremented
             and avoid any potential exceptions.

.. note:: In the LFRic API only :ref:`lfric-built-ins` are permitted
          to write to scalar arguments (and hence perform reductions).
          Furthermore, this permission is currently restricted to ``real``
          scalars (``GH_SCALAR, GH_REAL``) as the LFRic infrastructure
          does not yet support ``integer`` and ``logical`` reductions.

For a scalar, the argument metadata contains only these three entries.
However, fields, operators and scalar arrays require further entries specifying
function-space information or dimensionality. The meaning of these further
entries differs depending on whether a field, an operator or a scalar array is
being described.

In the case of a field the fourth argument specifies the function space that the
field lives on. In the case of an operator, the fourth and fifth arguments
describe the ``to`` and ``from`` function spaces respectively. In the case of a
scalar array, the fourth argument specifies the number of dimensions the array
has. More details about the supported function spaces are in subsection
:ref:`lfric-function-space`.

For example, the metadata for a kernel that applies a column-wise
operator to a field might look like::

  type(arg_type) :: meta_args(3) = (/                              &
       arg_type(GH_FIELD, GH_REAL, GH_INC, W1),                    &
       arg_type(GH_FIELD, GH_REAL, GH_READ, W2H),                  &
       arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_READ, W1, W2H) &
       /)

In some cases a Kernel may be written so that it works for fields and/or
operators from any type of a vector ``W2*`` space (all ``W2*`` spaces
except for the ``W2*trace`` spaces, see Section
:ref:`Supported Function Spaces <lfric-function-space>` below).
In this case the metadata should be specified as being ``ANY_W2``.

.. Warning:: In the current implementation it is assumed that all
             fields and/or operators specifying ``ANY_W2`` within a
             kernel will use the **same** function space. It is up to
             the user to ensure this is the case as otherwise invalid
             code would be generated.

It may be that a Kernel is written such that a field and/or operators
may be on/map-between any function space(s). In this case the metadata
should be specified as being one of ``ANY_SPACE_1``, ..., ``ANY_SPACE_<nmax>``
(see :ref:`Supported Function Spaces <lfric-function-space>`), with the
number of spaces, ``<nmax>``, being set in the :ref:`PSyclone configuration
file <configuration>` (see :ref:`here <lfric-num-any-spaces>` for more
details on this option).

If the generic function spaces are known to be discontinuous the metadata
may be specified as being one of ``ANY_DISCONTINUOUS_SPACE_1``, ...,
``ANY_DISCONTINUOUS_SPACE_<nmax>`` in order to avoid unnecessary computation
into the halos (see rules for
:ref:`user-supplied kernels <lfric-user-kernel-rules>` above).
The reason for having different names is that a Kernel might be written
to allow 2 or more arguments to be able to support any function space
but for a particular call the function spaces may have to be the same as
each other. Again, ``<nmax>`` is the :ref:`configurable
<lfric-num-any-spaces>` number of generalised discontinuous function spaces.

In the example below, the first field entry supports any function space but
it must be the same as the operator's ``to`` function space. Similarly,
the second field entry supports any function space but it must be the same
as the operator's ``from`` function space. Note, the metadata does not
forbid ``ANY_SPACE_1`` and ``ANY_SPACE_2`` from being the same.

::

  type(arg_type) :: meta_args(3) = (/                                    &
       arg_type(GH_FIELD,    GH_REAL, GH_INC,  ANY_SPACE_1),             &
       arg_type(GH_FIELD*3,  GH_REAL, GH_INC,  ANY_SPACE_2),             &
       arg_type(GH_OPERATOR, GH_REAL, GH_READ, ANY_SPACE_1, ANY_SPACE_2) &
       /)

Note also that the scope of this naming of any-space function spaces is
restricted to the argument list of individual kernels. I.e. if an
Invoke contains say, two kernel calls that each support arguments on
any function space, e.g. ``ANY_SPACE_1``, there is no requirement that
these two function spaces be the same. Put another way, if an Invoke
contained two calls of a kernel with arguments described by the above
metadata then the first field argument passed to each kernel call
need not be on the same space.

.. _lfric-kernel-valid-data-type:

Valid Data Types
^^^^^^^^^^^^^^^^

As mentioned earlier, the currently supported Fortran primitive
(intrinsic) types for kernel argument data are ``real``, ``integer``
and ``logical``, described by the ``GH_REAL``, ``GH_INTEGER`` and
``GH_LOGICAL`` metadata descriptors. Supported data types for each
argument type are given in the table below (please note that
:ref:`field vectors <lfric-field-vector>` follow the same rules as
the :ref:`LFRic fields <lfric-field>`):

.. tabularcolumns:: |l|l|

+------------------------+---------------------------------+
| Argument Type          | Data Type                       |
+========================+=================================+
| GH_SCALAR              | GH_REAL, GH_INTEGER, GH_LOGICAL |
+------------------------+---------------------------------+
| GH_SCALAR_ARRAY        | GH_REAL, GH_INTEGER, GH_LOGICAL |
+------------------------+---------------------------------+
| GH_FIELD               | GH_REAL, GH_INTEGER             |
+------------------------+---------------------------------+
| GH_OPERATOR            | GH_REAL                         |
+------------------------+---------------------------------+
| GH_COLUMNWISE_OPERATOR | GH_REAL                         |
+------------------------+---------------------------------+

.. _lfric-kernel-valid-access:

Valid Access Modes
^^^^^^^^^^^^^^^^^^

As mentioned earlier, not all combinations of metadata are
valid. Valid combinations for each argument type in
user-defined Kernels are summarised here. All argument types
(``GH_SCALAR``, ``GH_SCALAR_ARRAY``, ``GH_FIELD``, ``GH_OPERATOR`` and
``GH_COLUMNWISE_OPERATOR``) may be read within a Kernel and this
is specified in metadata using ``GH_READ``. At least one kernel
argument must be listed as being modified. When data is *modified*
in a user-supplied Kernel that operates on cell columns (see
:ref:`iteration space metadata <lfric-operates-on>`) then the permitted access
modes depend upon the argument type and the function space it is on:

.. tabularcolumns:: |l|l|l|

+------------------------+------------------------------+--------------------+
| Argument Type          | Function Space               | Access Type        |
+========================+==============================+====================+
| GH_SCALAR              | n/a                          | GH_READ            |
+------------------------+------------------------------+--------------------+
| GH_SCALAR_ARRAY        | n/a                          | GH_READ            |
+------------------------+------------------------------+--------------------+
| GH_FIELD               | Discontinuous                | GH_READ, GH_WRITE, |
|                        |                              | GH_READWRITE       |
+------------------------+------------------------------+--------------------+
| GH_FIELD               | Continuous                   | GH_READ, GH_WRITE, |
|                        |                              | GH_INC, GH_READINC |
+------------------------+------------------------------+--------------------+
| GH_OPERATOR            | Any for both 'to' and 'from' | GH_READ, GH_WRITE, |
|                        |                              | GH_READWRITE       |
+------------------------+------------------------------+--------------------+
| GH_COLUMNWISE_OPERATOR | Any for both 'to' and 'from' | GH_READ, GH_WRITE, |
|                        |                              | GH_READWRITE       |
+------------------------+------------------------------+--------------------+

Note that scalar arguments to user-defined Kernels must be read-only.
Only :ref:`Built-ins <lfric-built-ins>` are permitted to modify scalar
arguments. In practice this means that the only allowed access for scalar
arguments in user-defined Kernels is ``GH_READ`` (see the allowed accesses for
arguments in Built-ins in the :ref:`section below <lfric-built-ins-dtype-access>`).

Note also that a ``GH_FIELD`` argument that has ``GH_WRITE`` or
``GH_READWRITE`` as its access pattern must typically (see below) be
on a horizontally-discontinuous function space (see
:ref:`lfric-function-space` for the list of discontinuous function
spaces). Parallelisation of the loop over the horizontal domain for a
Kernel that updates such a field will not require colouring for either
of the above cases (since there are no shared entities).

There is however an exception to this - certain Kernels may write to
shared entities but each Kernel iteration is guaranteed to write the
*same value* to a given shared DoF. In this case, provided that the
first access to any such shared DoF is a write, the loop containing
such a Kernel may be parallelised without colouring. Therefore,
``GH_WRITE`` access is permitted for ``GH_FIELD`` arguments on
continuous function spaces. Obviously, care must be taken to ensure
that the Kernel implementation satisfies the constraints just
described as PSyclone cannot currently check this.

If a field is described as being on ``ANY_SPACE_*``, there is currently no
way to determine its continuity from the metadata (unless we can statically
determine the space of the field being passed in). At the moment this type
of a user-supplied Kernel is always treated as if it is updating a field
that is on a function space that is continuous in the horizontal, even if
it is not (see rules for :ref:`user-supplied kernels
<lfric-user-kernel-rules>` above).

There is no restriction on the number and function spaces of other
quantities that a general-purpose kernel can modify other than that it
must modify at least one. The rules for kernels involving CMA operators,
however, are stricter and only one argument may be modified (the CMA
operator itself for assembly, a field for CMA-application and a CMA
operator for matrix-matrix kernels). If a kernel writes to quantities
on different function spaces then PSyclone generates loop bounds
appropriate to the largest iteration space. This means that if a
single kernel updates one quantity on a continuous function space and
one on a discontinuous space then the resulting loop will include
cells in the level-1 halo since they are required for a quantity on a
continuous space. As a consequence, any quantities on a discontinuous
space will then be computed redundantly in the level-1 halo. Currently
PSyclone makes no attempt to take advantage of this (by e.g. setting
the appropriate level-1 halo to 'clean').

PSyclone ensures that both CMA and LMA operators are computed
(redundantly) out to the level-1 halo cells. This permits their use in
kernels which modify quantities on continuous function spaces and also
in subsequent redundant computation of other quantities on
discontinuous function spaces. In conjunction with this, PSyclone also
checks (when generating the PSy layer) that any kernels which read
operator values do not do so beyond the level-1 halo. If any such
accesses are found then PSyclone aborts.

.. _lfric-array-sizes:

Array sizes
^^^^^^^^^^^

The size of a :ref:`scalar array <lfric-array>` is described by ``<n>``,
where *n > 0* is the number of Fortran ranks representing the dimension of the
array, e.g. a logical, scalar array of rank three would be specified as:

::

  arg_type(GH_SCALAR_ARRAY, GH_LOGICAL, GH_READ, 3)

.. _lfric-function-space:

Supported Function Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in the :ref:`lfric-field` and :ref:`lfric-field-vector`
sections, the function space of an argument specifies how it maps
onto the underlying topology and, additionally, whether the data at a
point is a vector. In LFRic API the dimension of the basis function
set for the scalar function spaces is 1 and for the vector function spaces
is 3 (see the table in :ref:`lfric-stub-generation-rules` for the
dimensions of the basis and differential basis functions).

Function spaces can share DoFs between cells in the horizontal, vertical
or both directions. Depending on the function space and FEM order,
the shared DoFs can lie on one or more cell entities (faces, edges
and vertices) in each direction. This property is referred to as the
**continuity** of a function space (horizontal, vertical or full).
Alternatively, if there are no shared DoFs a function space is described
as **discontinuous** (fully or in a particular direction).

The mixed FEM formulation is built on a foundation set of four function
spaces described below.

* ``W0`` is the space of scalar functions with full continuity. The
  shared DoFs lie on cell vertices in the lowest order FEM and on
  all three entities in higher order FEM.

* ``W1`` is the space of vector functions with full continuity in the
  tangential direction only. In the lowest order FEM the shared DoFs
  lie on cell edges for each component, whereas in higher order they
  also lie on cell faces.

* ``W2`` is the space of vector functions with full continuity in the
  normal direction only. The shared DoFs lie on cell faces for each
  component.

* ``W3`` is the space of scalar functions with full discontinuity.
  All DoFs lie within the cell volume and are not shared across the
  cell boundaries.

Other spaces required for representation of scalar or component-wise
vector variables are:

* ``Wtheta`` is the space of scalar functions based on the vertical
  part of ``W2``, discontinuous in the horizontal and continuous
  in the vertical;

* ``W2H`` is the space of vector functions based on the horizontal
  part of ``W2``, continuous in the horizontal and discontinuous
  in the vertical;

* ``W2V`` is the space of vector functions based on the vertical
  part of ``W2``, discontinuous in the horizontal and continuous
  in the vertical;

* ``W2broken`` is the space of vector functions, locally identical
  to the ``W2`` space. However, DoFs are topologically discontinuous in
  all directions despite their placement on cell faces;

* ``W2trace`` is the space of scalar functions defined only on cell faces,
  resulting from taking the trace of a ``W2`` space. DoFs are shared between
  faces, hence making this space fully continuous;

* ``W2Htrace`` is the space of scalar functions defined only on cell faces
  in the horizontal, resulting from taking the trace of a ``W2H`` space.
  DoFs are shared between horizontal faces, hence making this space
  continuous in the horizontal and discontinuous in the vertical;

* ``W2Vtrace`` is the space of scalar functions defined only on cell faces
  in the vertical, resulting from taking the trace of a ``W2V`` space.
  DoFs are shared between vertical faces, hence making this space
  discontinuous in the horizontal and continuous in the vertical;

* ``Wchi`` is the space of scalar functions used to store coordinates
  in LFRic. It is fully discontinuous except for the coordinate order
  ``0`` when it becomes the ``W0`` space (i.e. fully continuous).
  Please see the next section for more details on this function space.

The previously mentioned FEM order can be specified in the horizontal
and vertical directions independently, through the ``element_order_h`` and
``element_order_v`` arguments in the function space initialisation. These
element orders dictate the polynomial order of the basis functions used to
represent a field. In most cases these element orders will be set to *0* in
LFRic, and increasing them will result in an increased number of DoFs per cell.
Increasing the element order in either direction will never change the
continuity of a space; however, it can increase the number of shared DoFs per
entity.

For example, the ``W3`` space has a single DoF per cell at the lowest order,
situated in the cell volume. The basis functions are constant over the
cell, and the space is discontinuous. Increasing to ``element_order_v=1``
(the so-called 'next-to-lowest order' in the vertical direction) will result
in a second volume DoF per cell, yielding linear basis functions vertically,
and constant basis functions horizontally. The space remains discontinuous
in both directions.

In addition to the specific function space metadata, there are also
three generic function space metadata descriptors mentioned in
sections above:

* ``ANY_SPACE_<n>``, *n = 1, 2, ... nmax*, for when the function
  space of the argument(s) cannot be determined and/or for when
  a Kernel has been written so that it works with fields on any
  of the available spaces (as mentioned in the
  :ref:`meta_args section <lfric-api-meta-args>`, the number of
  spaces, ``<nmax>``, is :ref:`configurable <lfric-num-any-spaces>`);

* ``ANY_DISCONTINUOUS_SPACE_<n>``, *n = 1, 2, ... nmax*, for when
  the function space of the argument(s) cannot be determined
  but is known to be discontinuous and/or for when a Kernel
  has been written so that it works with fields on any of the
  discontinuous spaces (again, the number of spaces, ``<nmax>``,
  is :ref:`configurable <lfric-num-any-spaces>`);

* ``ANY_W2`` for any type of a vector ``W2*`` function space, i.e. ``W2``,
  ``W2H``, ``W2V`` and ``W2broken`` but not ``W2*trace`` spaces.

As mentioned :ref:`previously <lfric-user-kernel-rules>` ,
``ANY_SPACE_<n>`` and ``ANY_W2`` function space types are treated as
continuous while ``ANY_DISCONTINUOUS_SPACE_<n>`` spaces are treated
as discontinuous.

.. note:: The name and use of ``ANY_W2`` metadata (e.g. continuity and
          vector or/and scalar basis of ``W2*`` spaces the metadata
          can represent) are being reviewed in PSyclone issue #540.

Since the LFRic API operates on columns of data, function spaces
are categorised as continuous or discontinuous with regard to their
**continuity in the horizontal**. For example, a ``GH_FIELD`` that
specifies ``GH_INC`` as its access pattern (see
:ref:lfric-kernel-valid-access: above) may be continuous in the vertical
(and discontinuous in the horizontal), continuous in the horizontal
(and discontinuous in the vertical), or continuous in both. In each
case the code is the same. This principle of horizontal continuity also
applies to the three generic ``ANY_*_*`` function space identifiers
above. The valid metadata values for continuous and discontinuous
function spaces are summarised in the table below.

.. tabularcolumns:: |l|l|

+---------------------------+--------------------------------------+
| Function Space Continuity | Function Space Name                  |
+===========================+======================================+
| Continuous                | W0, W1, W2, W2H, W2trace, W2Htrace,  |
|                           | ANY_W2, ANY_SPACE_<n>                |
+---------------------------+--------------------------------------+
| Discontinuous             | W2broken, W2V, W2Vtrace, W3, Wtheta, |
|                           | ANY_DISCONTINUOUS_SPACE_<n>          |
+---------------------------+--------------------------------------+

Horizontally discontinuous function spaces and fields over them will not
need colouring so PSyclone does not perform it. If such attempt is made,
PSyclone will raise a ``Generation Error`` in the **LFRicColourTrans**
transformation (see :ref:`lfric-api-transformations` for more details
on transformations). An example of fields iterating over a discontinuous
function space ``Wtheta`` is given in ``examples/lfric/eg9``, with the
``GH_READWRITE`` access descriptor denoting an update to the relevant
fields. This example also demonstrates how to only colour loops over
continuous function spaces when transformations are applied.

.. _lfric-ro-function-space:

Read-Only Function Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^

LFRic supports the concept of a **read-only function space**. A field
on such a function space must not be modified by any kernels contained
within ``invoke`` calls (i.e. within any code that PSyclone is
responsible for). Further, a field on a read-only function space must
contain clean halos in order to avoid any halo exchanges that would
occur if the field is read within a kernel where redundant
computation is performed.

The primary reason for including a read-only function space is that it
does not need any halo-exchange support e.g. it does not require a
routing table, which can reduce the memory footprint.

Currently ``Wchi`` is the only read-only function space in LFRic.

As a read-only function space is not modified, it does not matter
whether it is classified as continuous or discontinuous. LFRic
therefore treats read-only as a third category of function space.

Optional Field Metadata
^^^^^^^^^^^^^^^^^^^^^^^

A field entry in the meta_args array may have an optional fifth element.
This element describes either a stencil access or, for inter-grid kernels,
which mesh the field is on. Since an inter-grid kernel is not permitted
to have stencil accesses, these two options are mutually exclusive.
The metadata for each case is described in the following sections.

Stencil Metadata
________________


Stencil metadata specifies that the corresponding field argument is accessed
as a stencil operation within the Kernel.  Stencil metadata only makes sense
if the associated field is read within a Kernel i.e. it only makes
sense to specify stencil metadata if the first entry is ``GH_FIELD``
and the second entry is ``GH_READ``.

Stencil metadata is written in the following format::

  STENCIL(type)

where ``type`` may be one of ``X1D``, ``Y1D``, ``XORY1D``, ``CROSS``,
``CROSS2D`` or ``REGION``.  As the stencil ``extent`` (the maximum distance from
the central cell that the stencil extends) is not provided in the metadata,
it is expected to be provided by the algorithm writer as part of the
``invoke`` call (see Section :ref:`lfric-alg-stencil`). As there
is currently no way to specify a fixed extent value for stencils in the
Kernel metadata, Kernels must therefore be written to support
different values of extent (i.e. stencils with a variable number of
cells).

The ``XORY1D`` stencil type indicates that the Kernel can accept
either ``X1D`` or ``Y1D`` stencils. In this case it is up to the
algorithm developer to specify which of these it is from the algorithm
layer as part of the ``invoke`` call (see Section
:ref:`lfric-alg-stencil`).

For example, the following stencil (with ``extent=2``):

.. code-block:: none

  | 3 | 2 | 1 | 4 | 5 |

would be declared as::

  STENCIL(X1D)

and the following stencil (with ``extent=2``):

.. code-block:: none

  |   |   | 9 |   |   |
  |   |   | 8 |   |   |
  | 3 | 2 | 1 | 6 | 7 |
  |   |   | 4 |   |   |
  |   |   | 5 |   |   |

would be declared as::

  STENCIL(CROSS)

The ``REGION`` stencil references a block of cells:

.. code-block:: none

  | 9 | 8 | 7 |
  | 2 | 1 | 6 |
  | 3 | 4 | 5 |


and would be declared as::

  STENCIL(REGION)

Below is an example of stencil information within the full kernel metadata::

  type(arg_type) :: meta_args(3) = (/                                  &
       arg_type(GH_FIELD,    GH_REAL, GH_INC,  W1),                    &
       arg_type(GH_FIELD,    GH_REAL, GH_READ, W2H, STENCIL(CROSS)),   &
       arg_type(GH_OPERATOR, GH_REAL, GH_READ, W1, W2H)                &
       /)

There is a full example of this distributed with PSyclone. It may
be found in ``examples/lfric/eg5``.

.. _lfric-intergrid-mdata:

Inter-Grid Metadata
___________________


The alternative form of the optional fifth metadata argument for a
field specifies which mesh the associated field is on.  This is
required for inter-grid kernels which perform prolongation or
restriction operations on fields (or field vectors) existing on grids
of different resolutions.

Mesh metadata is written in the following format::

  mesh_arg=type

where ``type`` may be one of ``GH_COARSE`` or ``GH_FINE``. Any kernel
having a field argument with this metadata is assumed to be an
inter-grid kernel and, as such, all of its other arguments (which
must also be fields) must have it specified too. An example of the
metadata for such a kernel is given below::

  type(arg_type) :: meta_args(2) = (/                                      &
      arg_type(GH_FIELD, GH_REAL, GH_READWRITE, ANY_DISCONTINUOUS_SPACE_1, &
                                                      mesh_arg=GH_COARSE), &
      arg_type(GH_FIELD, GH_REAL, GH_READ,      ANY_DISCONTINUOUS_SPACE_2, &
                                                      mesh_arg=GH_FINE  )  &
      /)

Note that an inter-grid kernel must have at least one field (or field-
vector) argument on each mesh type. Fields that are on different
meshes cannot be on the same function space while those on the same
mesh must also be on the same function space.


Column-wise Operators (CMA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we provide example metadata for each of the three
recognised kernel types involving CMA operators.

Column-wise operators are constructed from cell-wise (local) operators.
Therefore, in order to **assemble** a CMA operator, a kernel must have at
least one read-only LMA operator, e.g.::

   type(arg_type) :: meta_args(2) = (/                                                 &
        arg_type(GH_OPERATOR,            GH_REAL, GH_READ,  ANY_SPACE_1, ANY_SPACE_2), &
        arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_WRITE, ANY_SPACE_1, ANY_SPACE_2)  &
        /)

CMA operators (and their inverse) are **applied** to fields. Therefore any
kernel of this type must have one read-only CMA operator, one read-only
field and a field that is updated, e.g.::

   type(arg_type) :: meta_args(3) = (/                                               &
        arg_type(GH_FIELD,               GH_REAL, GH_INC,  ANY_SPACE_1),             &
        arg_type(GH_FIELD,               GH_REAL, GH_READ, ANY_SPACE_2),             &
        arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_READ, ANY_SPACE_1, ANY_SPACE_2) &
        /)

**Matrix-matrix** kernels compute the product/linear combination of CMA
operators. They must therefore have one such operator that is updated while
the rest are read-only. They may also have read-only scalar arguments, e.g.::

   type(arg_type) :: meta_args(3) = (/                                                 &
        arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_WRITE, ANY_SPACE_1, ANY_SPACE_2), &
        arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_READ,  ANY_SPACE_1, ANY_SPACE_2), &
        arg_type(GH_COLUMNWISE_OPERATOR, GH_REAL, GH_READ,  ANY_SPACE_1, ANY_SPACE_2), &
        arg_type(GH_SCALAR,              GH_REAL, GH_READ) /)

.. note:: The order with which arguments are specified in metadata for CMA
          kernels does not affect the process of identifying the type of
          kernel (whether it is assembly, matrix-matrix etc.)

.. _lfric-meta-funcs:

meta_funcs
##########

The (optional) second component of kernel metadata specifies
whether any quadrature or evaluator data is required for a given
function space. (If no quadrature or evaluator data is required then
this metadata should be omitted.) Consider the
following kernel metadata::

    type, extends(kernel_type) :: testkern_operator_type
      type(arg_type), dimension(3) :: meta_args =                 &
          (/ arg_type(gh_operator, gh_real,    gh_write, w0, w0), &
             arg_type(gh_field*3,  gh_real,    gh_read,  w1),     &
             arg_type(gh_scalar,   gh_integer, gh_read)           &
          /)
      type(func_type) :: meta_funcs(2) =                          &
          (/ func_type(w0, gh_basis, gh_diff_basis)               &
             func_type(w1, gh_basis)                              &
          /)
      integer :: gh_shape = gh_quadrature_XYoZ
      integer :: operates_on = cell_column
    contains
      procedure, nopass :: code => testkern_operator_code
    end type testkern_operator_type

The ``arg_type`` component of this metadata describes a kernel that
takes three arguments (an operator, a field and an ``integer``
scalar). Following the ``meta_args`` array we now have a
``meta_funcs`` array. This allows the user to specify that the kernel
requires basis functions (``gh_basis``) and/or the differential of the
basis functions (``gh_diff_basis``) on one or more of the function
spaces associated with the arguments listed in ``meta_args``.  In this
case we require both for the W0 function space but only basis
functions for W1.

.. note:: Basis and differential basis functions for both ``real``- and
          ``integer``-valued field arguments have ``real`` values on the
          points on which these functions are :ref:`required
          <lfric-gh-shape>`.

meta_reference_element
######################

A kernel that requires properties of the reference element in LFRic
specifies those properties through the ``meta_reference_element``
metadata entry.  (If no reference element properties are required then
this metadata should be omitted.)  Consider the following example
kernel metadata::

  type, extends(kernel_type) :: testkern_type
    type(arg_type), dimension(2) :: meta_args =      &
        (/ arg_type(gh_field, gh_real, gh_read, w1), &
           arg_type(gh_field, gh_real, gh_inc,  w0) /)
    type(reference_element_data_type), dimension(2) ::               &
      meta_reference_element =                                       &
        (/ reference_element_data_type(normals_to_horizontal_faces), &
           reference_element_data_type(normals_to_vertical_faces) /)
  contains
    procedure, nopass :: code => testkern_code
  end type testkern_type

This metadata specifies that the ``testkern_type`` kernel requires two
properties of the reference element. The supported properties are
listed below:

.. tabularcolumns:: |p{5.5cm}|p{8.5cm}|

===================================  ===========================================
Name                                 Description
===================================  ===========================================
normals_to_horizontal_faces          Array of normals pointing in the positive
                                     (x, y, z) axis direction for each
                                     horizontal face indexed as (component,
                                     face).
normals_to_vertical_faces            Array of normals pointing in the positive
                                     (x, y, z) axis direction for each vertical
                                     face indexed as (component, face).
normals_to_faces                     Array of normals pointing in the positive
                                     (x, y, z) axis direction for each face
                                     indexed as (component, face).
outward_normals_to_horizontal_faces  Array of outward-pointing normals for each
                                     horizontal face indexed as (component,
                                     face).
outward_normals_to_vertical_faces    Array of outward-pointing normals for each
                                     vertical face indexed as (component, face).
outward_normals_to_faces             Array of outward-pointing normals for each
                                     face indexed as (component, face).
===================================  ===========================================

meta_mesh
#########

A kernel that requires properties of the LFRic mesh object specifies
those properties through the ``meta_mesh`` metadata entry. (If no
mesh properties are required then this metadata should be omitted.)
Consider the following example kernel metadata::

  type, extends(kernel_type) :: testkern_type
    type(arg_type), dimension(2) :: meta_args =      &
        (/ arg_type(gh_field, gh_real, gh_read, w1), &
           arg_type(gh_field, gh_real, gh_inc,  w0) /)
    type(mesh_data_type), dimension(1) ::            &
      meta_mesh =                                    &
        (/ mesh_data_type(adjacent_face) /)
  contains
    procedure, nopass :: code => testkern_code
  end type testkern_type

This metadata specifies that the ``testkern_type`` kernel requires one
property of the mesh. There is currently one supported property:

======================= ==================================================
Name                    Description
======================= ==================================================
adjacent_face           Local ID of a neighbouring face in each
                        horizontally-adjacent cell indexed as (face).
======================= ==================================================

.. _lfric-gh-shape:

gh_shape and gh_evaluator_targets
#################################

If a kernel requires basis or differential-basis functions then the
metadata must also specify the set of points on which these functions
are required. This information is provided by the ``gh_shape``
component of the metadata. Currently PSyclone supports four shapes;
``gh_quadrature_XYoZ`` for Gaussian quadrature points,
``gh_quadrature_face`` for quadrature points on cell faces,
``gh_quadrature_edge`` for quadrature points on cell edges and
``gh_evaluator`` for evaluation at nodal points. If a kernel requires
just one of these then ``gh_shape`` is an ``integer`` scalar. However, if
more than one is required then ``gh_shape`` becomes a one-dimensional,
``integer`` array, e.g.::

    integer :: gh_shape(2) = (/ gh_quadrature_face, gh_quadrature_edge /)

If a kernel requires an evaluator then there are two options: if an
evaluator is required for multiple function spaces then these can be
specified using the additional ``gh_evaluator_targets`` metadata
entry. This entry is a one-dimensional, ``integer`` array containing the
desired function spaces. For example, to request
basis/differential-basis functions evaluated on both W0 and W1, the
metadata would be::

    integer :: gh_shape = gh_evaluator
    integer :: gh_evaluator_targets(2) = (/W0, W1/)

The kernel must have an argument (field or operator) on each of the
function spaces listed in ``gh_evaluator_targets``.
The default behaviour if ``gh_evaluator_targets`` is not specified is
to provide evaluators for each function space associated with the
quantities that the kernel is updating. All necessary data is
extracted in the PSy layer and passed to the kernel(s) as required -
nothing is required from the Algorithm layer. If a kernel requires
quadrature on the other hand, the Algorithm writer must supply a
``quadrature_type`` object for each specified quadrature as the last
argument(s) to the kernel (see Section :ref:`lfric-quadrature`).

Note that it is an error for kernel metadata to specify a value for
``gh_shape`` if no basis or differential-basis functions are required.
It is also an error to specify ``gh_evaluator_targets`` if the kernel
does not require an evaluator (i.e. ``gh_shape != gh_evaluator``).

.. _lfric-operates-on:

operates_on
###########

The fourth type of metadata provided is ``OPERATES_ON``. This
specifies that the Kernel has been written with the assumption that it
is supplied with the specified data for each field/operator argument.
The possible values for ``OPERATES_ON`` and their interpretation are
summarised in the following table:

.. tabularcolumns:: |p{4.5cm}|p{3.0cm}|p{6.5cm}|

+--------------------------------+--------------------------------------+--------------------------------------------+
| operates_on                    | Data passed for each field/operator  | Iteration space                            |
|                                | argument                             |                                            |
+================================+======================================+============================================+
| ``cell_column``                | Single column of cells.              | Conceptually, all columns in the global    |
|                                |                                      | mesh. For each MPI                         |
|                                |                                      | process this will operate on all owned     |
|                                |                                      | columns and may be extended into the halo  |
|                                |                                      | to perform redundant computation.          |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``owned_cell_column``          | Single column of cells.              | Restricted to owned columns. Prevents      |
|                                |                                      | extending into the halos to perform        |
|                                |                                      | redundant computation.                     |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``halo_cell_column``           | Single column of cells.              | Restricted to columns from the halo region |
|                                |                                      | (to a specified depth).                    |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``owned_and_halo_cell_column`` | Single column of cells.              | Iteration space must include both owned    |
|                                |                                      | and halo columns (to a specified depth).   |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``dof``                        | Single DoF.                          | Defaults to owned DoFs but may be extended |
|                                |                                      | to annexed and halo DoFs.                  |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``owned_dof``                  | Single DoF.                          | Restricted to owned DoFs only. Prevents    |
|                                |                                      | extending into the halos to perform        |
|                                |                                      | redundant computation.                     |
+--------------------------------+--------------------------------------+--------------------------------------------+
| ``domain``                     | All columns of cells in the (sub-)   | None.                                      |
|                                | domain.                              |                                            |
+--------------------------------+--------------------------------------+--------------------------------------------+

(For a description of the concepts of 'owned' and 'halo' cells and 'annexed' DoFs
please see the :ref:`LFRic section <lfric-developers>` of the Developer Guide.)

The ``owned_cell_column`` and ``owned_dof`` values of ``OPERATES_ON`` are intended for
use only with special cases where the kernel concerned cannot be used to perform
redundant computation (e.g. when filling a field with pseudo-random numbers without
regard to cell location). Processing an application that makes use of such a kernel
requires that the ``COMPUTE_ANNEXED_DOFS`` configuration option (see
:ref:`lfric-annexed_dofs`) be set to ``False`` as PSyclone can no longer guarantee that
annexed DoFs are always clean between different ``invoke`` calls.

procedure
#########

The fifth and final type of metadata is ``procedure`` metadata. This
specifies the name of the Kernel subroutine that this metadata
describes.

For example::

  procedure, nopass :: my_kernel_subroutine

.. _lfric-kern-subroutine:

Subroutine
++++++++++

.. _lfric-stub-generation-rules:

Rules for General-Purpose Kernels
#################################

The arguments to general-purpose kernels (those that do not involve
either CMA operators or prolongation/restriction operations) that
operate on cell-columns follow a set of rules
which have been specified for the LFRic API. These rules are encoded
in the ``generate()`` method within the ``ArgOrdering`` abstract class
in the ``lfric.py`` file. The rules, along with PSyclone's naming
conventions, are:

1) If an LMA operator is passed then include the ``cells`` argument.
   ``cells`` is an ``integer`` of kind ``i_def`` and has intent ``in``.
2) Include ``nlayers``, the number of layers in a column. ``nlayers``
   is an ``integer`` of kind ``i_def`` and has intent ``in``. PSyclone
   will obtain the value of ``nlayers`` to use for a particular kernel
   from the first field or operator in the argument list.
3) For each scalar/field/vector_field/operator/ScalarArray in the order specified by
   the meta_args metadata:

   1) If the current entry is a scalar quantity then include the Fortran
      variable in the argument list. The intent is determined from the
      metadata (see :ref:`lfric-api-meta-args` for an explanation).
   2) If the current entry is a field then include the field
      array. The field array name is currently specified as being
      ``"field_"<argument_position>"_"<field_function_space>``. A
      field array is a rank-1, ``real`` array with extent equal to the
      number of unique degrees of freedom for the space that the field
      is on. Its precision (kind) depends on how it is defined in the
      algorithm layer, see the :ref:`lfric-mixed-precision` section
      for more details. This value is passed in separately. Again, the
      intent is determined from the metadata (see
      :ref:`lfric-api-meta-args`).

      1) If the field entry has a stencil access then add an ``integer`` (or
         if the stencil is of type ``CROSS2D``, an ``integer`` rank-1 array of
         extent 4 and kind ``i_def``) stencil-size argument with intent ``in``.
         This will supply the number of cells in the stencil or, in the case
         of the ``CROSS2D`` stencil, the number of cells in each branch of
         the stencil.
      2) If the stencil is of type ``CROSS2D`` then an ``integer`` of kind
         ``i_def`` and intent ``in`` for the max branch length is needed.
         This is used in defining the dimensions of the stencil dofmap array
         and is required due to the varying length of the branches of the
         stencil when used on planar meshes.
      3) Also needed is a stencil dofmap array of type ``integer``, kind
         ``i_def`` and intent ``in`` in either 2 or 3 dimensions. For a
         ``CROSS2D`` stencil the array needs dimensions of
         (number-of-dofs-in-cell, max-branch-length, 4).
         All other stencils need dimensions of (number-of-dofs-in-cell,
         stencil-size).
      4) If the field entry stencil access is of type ``XORY1D`` then
         add an additional ``integer`` direction argument of kind
         ``i_def`` and with intent ``in``.

   3) If the current entry is a field vector then for each dimension
      of the vector, include a field array. The field array name is
      specified as
      ``"field_"<argument_position>"_"<field_function_space>"_v"<vector_position>``.
      A field array in a field vector is declared in the same way as a
      field array (described in the previous step).
   4) If the current entry is an operator then first include an
      ``integer`` extent of kind ``i_def``. The name of this extent is
      ``<operator_name>"_ncell_3d"``. Next include the operator.  This
      is a rank-3, ``real`` array. Its precision (kind) depends on how
      it is defined in the algorithm layer, see the
      :ref:`lfric-mixed-precision` section for more details. The
      extent of the first dimension is ``<operator_name>"_ncell_3d"``,
      and of the second and third dimension are the local degrees of
      freedom for the ``to`` and ``from`` function spaces,
      respectively. Again the intent is determined
      from the metadata (see :ref:`lfric-api-meta-args`).
   5) If the current entry is a ScalarArray then first include a rank-1
      ``integer`` array of kind ``i_def`` and size ``nranks_<array_name>``
      containing the upper bounds for each rank, ``dims_<array_name>``
      (the lower bound is assumed to be 1 as this is how Fortran passes
      array slices to subroutines by default). Then pass the array of
      the data type and kind specified in the metadata. The ScalarArray
      must be denoted with intent ``in`` to match its read-only nature.

4) For each function space in the order they appear in the metadata arguments
   (the ``to`` function space of an operator is considered to be before the
   ``from`` function space of the same operator as it appears first in
   lexicographic order)

   1) Include the number of local degrees of freedom (i.e. number per-cell)
      for the function space. This is an ``integer`` of kind ``i_def`` and
      has intent ``in``. The name of this argument is
      ``"ndf_"<field_function_space>``.
   2) If there is a field on this space

      1) Include the unique number of degrees of freedom for the function
         space. This is an ``integer`` of kind ``i_def`` and has intent ``in``.
         The name of this argument is ``"undf_"<field_function_space>``.
      2) Include the **dofmap** for this function space. This is an ``integer``
         array of kind ``i_def`` with intent ``in``. It has one dimension
         sized by the local degrees of freedom for the function space.

   3) For each operation on the function space (``basis``, ``diff_basis``),
      in the order specified in the metadata, pass ``real`` arrays of kind
      ``r_def`` with intent ``in``. For each shape specified in the
      ``gh_shape`` metadata entry:

      1) If shape is ``gh_quadrature_*`` then the arrays are of rank four
         and are named
         ``"basis_"<field_function_space>_<quadrature_arg_name>`` or
         ``"diff_basis_"<field_function_space>_<quadrature_arg_name>``,
         as appropriate:

         1) If shape is ``gh_quadrature_xyoz`` then the arrays have extent
            (``dimension``, ``number_of_dofs``, ``np_xy``, ``np_z``).

         2) If shape is ``gh_quadrature_face`` or ``gh_quadrature_edge``
            then the  arrays have extent
            (``dimension``, ``number_of_dofs``, ``np_xyz``, ``nfaces`` or
            ``nedges``).

      2) If shape is ``gh_evaluator`` then we pass one array for
         each target function space (i.e. as specified by
         ``gh_evaluator_targets``). Each of these arrays are of rank three
         with extent (``dimension``, ``number_of_dofs``,
         ``ndf_<target_function_space>``). The name of the argument is
         ``"basis_"<field_function_space>"_on_"<target_function_space>`` or
         ``"diff_basis_"<field_function_space>"_on_"<target_function_space>``,
         as appropriate.

      Here ``<quadrature_arg_name>`` is the name of the corresponding
      quadrature object being passed to the Invoke.
      ``dimension`` is 1 or 3 and depends upon the function space
      (see :ref:`lfric-function-space` above for more information) and
      whether or not it is a basis or a differential basis function (see
      the table below). ``number_of_dofs`` is the number of degrees of
      freedom (DoFs) associated with the function space and ``np_*`` are
      the number of points to be evaluated: i) ``*_xyz`` in
      all directions (3D); ii) ``*_xy`` in the horizontal plane (2D);
      iii) ``*_x, *_y`` in the horizontal (1D); and iv) ``*_z`` in the
      vertical (1D). ``nfaces`` and ``nedges`` are the number of horizontal
      faces/edges obtained from the appropriate quadrature object supplied
      to the Invoke.

      .. tabularcolumns:: |l|c|l|

      +---------------+-----------+------------------------------------+
      | Function Type | Dimension | Function Space Name                |
      +===============+===========+====================================+
      | Basis         |    1      | W0, W2trace, W2Htrace, W2Vtrace,   |
      |               |           | W3, Wtheta, Wchi                   |
      |               +-----------+------------------------------------+
      |               |    3      | W1, W2, W2H, W2V, W2broken, ANY_W2 |
      +---------------+-----------+------------------------------------+
      | Differential  |    1      | W2, W2H, W2V, W2broken, ANY_W2     |
      | Basis         +-----------+------------------------------------+
      |               |    3      | W0, W1, W2trace, W2Htrace,         |
      |               |           | W2Vtrace, W3, Wtheta, Wchi         |
      +---------------+-----------+------------------------------------+

5) If either the ``normals_to_horizontal_faces`` or
   ``outward_normals_to_horizontal_faces`` properties of the reference
   element are required then pass the number of horizontal faces of the
   reference element (``nfaces_re_h``). Similarly, if either the
   ``normals_to_vertical_faces`` or ``outward_normals_to_vertical_faces`` are
   required then pass the number of vertical faces (``nfaces_re_v``). This
   also holds for the ``normals_to_faces`` and ``outward_normals_to_faces``
   where the number of all faces of the reference element (``nfaces_re``)
   is passed to the kernel. (All of these quantities are integers of kind
   ``i_def``.) Then, in the order specified in the
   ``meta_reference_element`` metadata:

   1) For the ``normals_to_horizontal/vertical_faces``, pass a rank-2
      ``integer`` array of kind ``i_def`` with dimensions
      ``(3, nfaces_re_h/v)``.
   2) For the ``outward_normals_to_horizontal/vertical_faces``, pass a rank-2
      ``integer`` array of kind ``i_def`` with dimensions
      ``(3, nfaces_re_h/v)``.
   3) For ``normals_to_faces`` or ``outward_normals_to_faces`` pass
      a rank-2 ``integer`` array of kind ``i_def`` with dimensions
      ``(3, nfaces_re)``.

6) If the ``adjacent_face`` mesh property is required then:

   1) If the number of horizontal cell faces obtained from the reference
      element (``nfaces_re_h``) is not already being passed to the kernel (due
      to rule 5 above) then supply it here. This is an ``integer`` of kind
      ``i_def``.
   2) Pass a rank-1, ``integer`` array of kind ``i_def`` and extent
      ``nfaces_re_h``.

7) If Quadrature is required (``gh_shape = gh_quadrature_*``) then, for
   each shape in the order specified in the ``gh_shape`` metadata:

   1) Include ``integer``, scalar arguments of kind ``i_def`` with intent
      ``in`` that specify the extent of the basis/diff-basis arrays:

      1) If ``gh_shape`` is ``gh_quadrature_XYoZ`` then pass
         ``np_xy_<quadrature_arg_name>`` and ``np_z_<quadrature_arg_name>``.
      2) If ``gh_shape`` is ``gh_quadrature_face``/``_edge`` then pass
         ``nfaces``/``nedges_<quadrature_arg_name>`` and
         ``np_xyz_<quadrature_arg_name>``.

   2) Include weights which are ``real`` arrays of kind ``r_def``:

      1) If ``gh_quadrature_XYoZ`` pass in
         ``weights_xz_<quadrature_arg_name>`` (rank one, extent
         ``np_xy_<quadrature_arg_name>``)
         and ``weights_z_<quadrature_arg_name>`` (rank one, extent
         ``np_z_<quadrature_arg_name>``).
      2) If ``gh_quadrature_face``/``_edge`` pass in
         ``weights_xyz_<quadrature_arg_name>`` (rank two with extents
         [``np_xyz_<quadrature_arg_name>``,
         ``nfaces/nedges_<quadrature_arg_name>``]).

Examples
^^^^^^^^

For instance, if a kernel has only one written argument and requires an
evaluator then its metadata might be::

  type, extends(kernel_type) :: testkern_operator_type
     type(arg_type), dimension(2) :: meta_args =               &
          (/ arg_type(gh_operator, gh_real, gh_write, w0, w1), &
             arg_type(gh_field*3,  gh_real, gh_read,  w0) /)
     type(func_type) :: meta_funcs(1) =                        &
          (/ func_type(w0, gh_basis) /)
     integer :: operates_on = cell_column
     integer :: gh_shape = gh_evaluator
   contains
     procedure, nopass :: code => testkern_operator_code
  end type testkern_operator_type

then we only pass the basis functions evaluated on ``W0`` (the space of
the written kernel argument). The subroutine arguments will therefore
be::

  subroutine testkern_operator_code(cell, nlayers, ncell_3d,        &
       local_stencil, xdata, ydata, zdata, ndf_w0, undf_w0, map_w0, &
       basis_w0_on_w0, ndf_w1)

where ``local_stencil`` is the operator, ``xdata``, ``ydata``
etc\. are the three components of the field vector and ``map_w0`` is
the dofmap for the ``W0`` function space.

If instead, ``gh_evaluator_targets`` is specified in the metadata::

  type, extends(kernel_type) :: testkern_operator_type
     type(arg_type), dimension(2) :: meta_args =               &
          (/ arg_type(gh_operator, gh_real, gh_write, w0, w1), &
             arg_type(gh_field*3,  gh_real, gh_read,  w0) /)
     type(func_type) :: meta_funcs(1) =               &
          (/ func_type(w0, gh_basis) /)
     integer :: operates_on = cell_column
     integer :: gh_shape = gh_evaluator
     integer :: gh_evaluator_targets(2) = (/W0, W1/)
   contains
     procedure, nopass :: code => testkern_operator_code
  end type testkern_operator_type

then we will need to pass two sets of basis functions (evaluated at ``W0``
and at ``W1``)::

  subroutine testkern_operator_code(cell, nlayers, ncell_3d,        &
       local_stencil, xdata, ydata, zdata, ndf_w0, undf_w0, map_w0, &
       basis_w0_on_w0, basis_w0_on_w1, ndf_w1)

If the metadata specifies that a kernel requires both an evaluator
and quadrature::

  type, extends(kernel_type) :: testkern_operator_type
     type(arg_type), dimension(2) :: meta_args =               &
          (/ arg_type(gh_operator, gh_real, gh_write, w0, w1), &
             arg_type(gh_field*3,  gh_real, gh_read,  w0) /)
     type(func_type) :: meta_funcs(1) =                        &
          (/ func_type(w0, gh_basis) /)
     integer :: operates_on = cell_column
     integer :: gh_shape(2) = (/ gh_evaluator, gh_quadrature_face /)
   contains
     procedure, nopass :: code => testkern_operator_code
  end type testkern_operator_type

then we will need to pass basis functions for both the evaluator and the
quadrature (where ``qr_face`` is the name of the face-quadrature object
passed to the Invoke)::

  subroutine testkern_operator_code(cell, nlayers, ncell_3d,              &
       local_stencil, xdata, ydata, zdata, ndf_w0, undf_w0, map_w0,       &
       basis_w0_on_w0, basis_w0_qr_face, ndf_w1,                          &
       np_xyz_qr_face, nfaces_qr_face, weights_xyz_qr_face)

If the metadata specifies that the kernel requires a property of the
reference element::

  type, extends(kernel_type) :: testkern_operator_type
     type(arg_type), dimension(2) :: meta_args =               &
          (/ arg_type(gh_operator, gh_real, gh_write, w0, w1), &
             arg_type(gh_field*3,  gh_real, gh_read,  w0) /)
     type(reference_element_data_type) :: meta_reference_element(1) =  &
          (/ reference_element_data_type(normals_to_horizontal_faces) /)
     integer :: operates_on = cell_column
   contains
     procedure, nopass :: code => testkern_operator_code
  end type testkern_operator_type

then the kernel must be passed the number of faces of the reference element
and the array of face normals in the specified direction (here horizontal)::

  subroutine testkern_operator_code(cell, nlayers, ncell_3d,        &
       local_stencil, xdata, ydata, zdata, ndf_w0, undf_w0, map_w0, &
       nfaces_re_h, normals_face_h)


Rules for CMA Kernels
#####################

Kernels involving CMA operators are restricted to just three types;
assembly, application/inverse-application and matrix-matrix.
We give the rules for each of these in the sections below.

Assembly
^^^^^^^^

An assembly kernel requires the column-banded dofmap for both the to-
and from-function spaces of the CMA operator being assembled as well
as the number of DoFs for each of the dofmaps. The full set of rules is:

1) Include the ``cell`` argument. ``cell`` is an ``integer`` of kind
   ``i_def``and has intent ``in``.

2) Include ``nlayers``, the number of layers in a column. ``nlayers``
   is an ``integer`` of kind ``i_def`` and has intent ``in``.

3) Include the total number of cells in the 2D mesh (including halos),
   ``ncell_2d``, which is an ``integer`` of kind ``i_def`` with
   intent ``in``.

4) Include the total number of cells, ``ncell_3d``, which is an ``integer``
   of kind ``i_def`` with intent ``in``.

5) For each argument in the ``meta_args`` metadata array:

   1) If it is a LMA operator, include a ``real``, 3-dimensional
      array. The first dimension is ``ncell_3d``. The second and third
      dimension are the local degrees of freedom for the ``to`` and
      ``from`` spaces, respectively.  The precision of the array depends
      on how it is defined in the algorithm layer, see the
      :ref:`lfric-mixed-precision` section for more details;

   2) If it is a CMA operator, include a ``real``, 3-dimensional array
      of kind ``r_solver``. The first dimension is
      ``"bandwidth_"<operator_name>``, the second is
      ``"nrow_"<operator_name>``, and the third is ``ncell_2d``.

      1) Include the number of rows in the banded matrix.  This is
         an ``integer`` of kind ``i_def`` with intent ``in`` and is named
         as ``"nrow_"<operator_name>``.

      2) If the from-space of the operator is *not* the same as the
         to-space then include the number of columns in the banded
         matrix.  This is an ``integer`` of kind ``i_def`` with intent
         ``in`` and is named as ``"ncol_"<operator_name>``.

      3) Include the bandwidth of the banded matrix. This is an
         ``integer`` of kind ``i_def`` with intent ``in`` and is named as
         ``"bandwidth_"<operator_name>``.

      4) Include banded-matrix parameter ``alpha``. This is an ``integer``
         of kind ``i_def`` with intent ``in`` and is named as
         ``"alpha_"<operator_name>``.

      5) Include banded-matrix parameter ``beta``. This is an ``integer``
         of kind ``i_def`` with intent ``in`` and is named as
         ``"beta_"<operator_name>``.

      6) Include banded-matrix parameter ``gamma_m``. This is an ``integer``
         of kind ``i_def`` with intent ``in`` and is named as
         ``"gamma_m_"<operator_name>``.

      7) Include banded-matrix parameter ``gamma_p``. This is an ``integer``
         of kind ``i_def`` with intent ``in`` and is named as
         ``"gamma_p_"<operator_name>``.

   3) If it is a field or scalar argument then include arguments following
      the same rules as for general-purpose kernels.

6) For each unique function space in the order they appear in the
   metadata arguments (the ``to`` function space of an operator is
   considered to be before the ``from`` function space of the same
   operator as it appears first in lexicographic order):

   1) Include the number of degrees of freedom per cell for the space.
      This is an ``integer`` of kind ``i_def`` with intent ``in``. The name
      of this argument is ``"ndf_"<arg_function_space>``.

   2) If there is a field on this space then:

      1) Include the unique number of degrees of freedom for the
         function space. This is an ``integer`` of kind ``i_def`` and has
         intent ``in``. The name of this argument is
         ``"undf_"<field_function_space>``.

      2) Include the dofmap for this space. This is an ``integer`` array
         of kind ``i_def`` with intent ``in``. It has one dimension
         sized by the local degrees of freedom for the function space.

   3) If the CMA operator has this space as its to/from space then
      include the column-banded dofmap, the list of offsets for the
      to/from-space. This is an ``integer`` array of rank 2 and kind
      ``i_def``. The first dimension is ``"ndf_"<arg_function_space>``
      and the second is ``nlayers``.


Application/Inverse-Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A kernel applying a CMA operator requires the column-indirection
dofmap for both the to- and from-function spaces of the CMA
operator. Since it does not have any LMA operator arguments it does
not require the ``ncell_3d`` and ``nlayers`` scalar arguments. (Since a
column-wise operator is, by definition, assembled for a whole column,
there is no loop over levels when applying it.)
The full set of rules is then:

1) Include the ``cell`` argument. ``cell`` is an ``integer`` of kind
   ``i_def`` and has intent ``in``.

2) Include the total number of cells in the 2D mesh (including halos),
   ``ncell_2d``, which is an ``integer`` of kind ``i_def`` with
   intent ``in``.

3) For each argument in the ``meta_args`` metadata array:

   1) If it is a field, include the field array. This is a ``real``
      array of rank 1. Its precision (kind) depends on how it is
      defined in the algorithm layer, see the
      :ref:`lfric-mixed-precision`. The field array name is
      currently specified as being
      ``"field_"<argument_position>"_"<field_function_space>``. The
      extent of the array is the number of unique degrees of freedom
      for the function space that the field is on.  This value is
      passed in separately. The intent of the argument is determined
      from the metadata (see :ref:`lfric-api-meta-args`);

   2) If it is a CMA operator, include it and its associated
      parameters (see Rule 5 of CMA Assembly kernels).

4) For each of the unique function spaces encountered in the
   metadata arguments (the ``to`` function space of an operator
   is considered to be before the ``from`` function space of the
   same operator as it appears first in lexicographic order):

   1) Include the number of degrees of freedom per cell for the associated
      function space. This is an ``integer`` of kind ``i_def`` with intent
      ``in``. The name of this argument is ``"ndf_"<field_function_space>``;

   2) Include the number of unique degrees of freedom for the associated
      function space. This is an ``integer`` of kind ``i_def`` with intent
      ``in``. The name of this argument is ``"undf_"<field_function_space>``;

   3) Include the dofmap for this function space. This is a rank-1 ``integer``
      array of kind ``i_def`` with extent equal to the number of degrees of
      freedom of the space (``"ndf_"<field_function_space>``).

5) Include the indirection map for the to-space of the CMA operator.
   This is a rank-1 ``integer`` array of kind ``i_def`` with extent ``nrow``.

6) If the from-space of the operator is *not* the same as the to-space
   then include the indirection map for the from-space of the CMA operator.
   This is a rank-1 ``integer`` array of kind ``i_def`` with extent ``ncol``.

Matrix-Matrix
^^^^^^^^^^^^^

Does not require any dofmaps and also does not require the ``nlayers``
and ``ncell_3d`` scalar arguments. The full set of rules are then:

1) Include the ``cell`` argument. ``cell`` is an ``integer`` of kind
   ``i_def`` and has intent ``in``.

2) Include the total number of cells in the 2D mesh (including halos),
   ``ncell_2d``, which is an ``integer`` of kind ``i_def`` with intent ``in``.

3) For each CMA operator or scalar argument specified in metadata:

   1) If it is a CMA operator, include it and its associated
      parameters (see Rule 5 of CMA Assembly kernels);

   2) If it is a scalar argument include the corresponding Fortran
      variable in the argument list with intent ``in``.

Rules for Inter-Grid Kernels
############################

As already specified, inter-grid kernels are only permitted to take
fields and/or field-vectors as arguments. Fields (and field-vectors)
that are on different meshes must be on different function
spaces. Fields on the same mesh must also be on the same function
space.

Argument ordering follows the general pattern used for 'normal'
kernels with field data being followed by dofmap data. The rules for
arguments to inter-grid kernels are as follows:

1) Include ``nlayers``, the number of layers in a column. ``nlayers``
   is an ``integer`` of kind ``i_def`` and has intent ``in``.

2) Include the ``cell_map`` for the current cell (column). This is
   an ``integer`` array of rank two, kind ``i_def`` and intent ``in``
   which provides the mapping from the coarse to the fine mesh. It
   has extent ``(ncell_f_per_c_x, ncell_f_per_c_y)``.

3) Include ``ncell_f_per_c_x``, and ``ncell_f_per_c_y``, the numbers of
   fine cells per coarse cell in the ``x`` and ``y`` directions,
   respectively. These are integers of kind ``i_def`` and have intent
   ``in``.

4) Include ``ncell_f``, the number of cells (columns) in the fine mesh.
   This is an ``integer`` of kind ``i_def`` and has intent ``in``.

5) For each argument in the ``meta_args`` metadata array (which must be
   a field or field-vector):

   1) Pass in field data as done for a regular kernel.

6) For each unique function space (of which there will currently be two)
   in the order in which they are encountered in the ``meta_args``
   metadata array, include dofmap information:

   If the dofmap is associated with an argument on the fine mesh:

   1) Include ``ndf_fine``, the number of DoFs per cell for the FS of
      the field on the fine mesh;

   2) Include ``undf_fine``, the number of unique DoFs per cell for the FS
      of the field on the fine mesh;

   3) Include ``dofmap_fine``, the *whole* dofmap for the fine mesh. This
      is an ``integer`` array of rank two and kind ``i_def`` with intent
      ``in``. The extent of the first dimension is ``ndf_fine`` and that of
      the second is ``ncell_f``.

   else, the dofmap is associated with an argument on the coarse mesh:

   1) Include ``undf_coarse``, the number of unique DoFs for the coarse
      field. This is an ``integer`` of kind ``i_def`` with intent ``in``;

   2) Include ``dofmap_coarse``, the dofmap for the current cell (column)
      in the coarse mesh. This is an ``integer`` array of rank one, kind
      ``i_def``and has intent ``in``.

Rules for Domain Kernels
########################

The rules for kernels that have ``operates_on = DOMAIN`` are almost
identical to those for general-purpose kernels (described :ref:`above
<lfric-stub-generation-rules>`), allowing for the fact that they
are not permitted any type of operator argument or any argument with a
stencil access. The only difference is that, since the kernel operates
on the whole domain, the number of columns in the mesh excluding those
in the halo (``ncell_2d_no_halos``), must be passed in. This is provided
as the second argument to the kernel (after ``nlayers``).
``ncell_2d_no_halos`` is an ``integer`` of kind ``i_def`` with intent ``in``.

Rules for DoF Kernels
#####################

The rules for kernels that have ``operates_on = DOF`` are similar to those for
general-purpose kernels but, due to the restriction that only fields and
scalars can be passed to them, are much fewer. The full set of rules, along
with PSyclone's naming conventions, are:

   1) Include `df`, the index of the single dof to be operated on. This is an
      ``integer`` of of kind ``i_def`` with intent ``in``.

   2) For each scalar/field in the order specified by the meta_args metadata:

      1) If the current entry is a scalar quantity then include the Fortran
         variable in the argument list. The intent is determined from the
         metadata (see meta_args for an explanation).

      2) If the current entry is a field then include the field array. The
         field array name is currently specified as being ``"field_"
         <argument_position>``. A field array is a rank-1, real array with
         extent equal to the number of unique degrees of freedom for the space
         that the field is on. Its precision (kind) depends on how it is
         defined in the algorithm layer, see the :ref:`Mixed Precision
         <lfric-mixed-precision>` section for more details. This value is
         passed in separately. Again, the intent is determined from the
         metadata (see :ref:`meta_args <lfric-api-meta-args>`).

   3) For each field vector in the order specified by the meta_args metadata,
      there needs to be an equivalent number of arguments in the kernel as
      the dimension of the field vector. The dimension is specified in the
      metadata. The arguments must be ordered following the indexing of the
      field vector.

.. _lfric-kernel-arg-intents:

Argument Intents
################

As described :ref:`above <lfric-psy-arg-intents>`, LFRic kernels read
and/or update the data pointed to by objects such as
:ref:`fields <lfric-field>` or :ref:`operators <lfric-operator>`.
This data is passed to the kernels as :ref:`subroutine arguments
<lfric-kern-subroutine>` and their Fortran intents usually follow the
logic determined by their :ref:`access modes <lfric-kernel-valid-access>`.

* ``GH_READ`` indicates ``intent(in)`` as the argument is only ever read from.

* ``GH_WRITE`` (for discontinuous function spaces) indicates that the argument
  is only written to in a kernel. The field and operator arguments' data in
  LFRic are always defined outside of a kernel so the argument intent for
  this access type is ``intent(inout)``.

* ``GH_INC``, ``GH_READINC`` and ``GH_READWRITE`` indicate
  ``intent(inout)`` as the arguments are updated (albeit in a
  different way due to different access to DoFs, see
  :ref:`lfric-api-meta-args` for more details).


Kernel Naming Conventions
+++++++++++++++++++++++++

LFRic development uses strict naming conventions related to kernels.
While they are not a requirement for PSyclone itself, any LFRic
development should follow these conventions (see e.g.
:ref:`LFRic examples <examples_lfric>` in PSyclone):

Module name:
    ``<base_name>_kernel_mod``
Kernel type name:
    ``<base_name>_kernel_type``
Subroutine name:
    ``<base_name>_code``

The latest version of the LFRic coding style guidelines are available in this
`LFRic wiki page
<https://code.metoffice.gov.uk/trac/lfric/wiki/LFRicTechnical/FortranCodingStandards>`_
(requires login access to MOSRS, see the above :ref:`introduction <lfric-api>`
to the LFRic API).

.. _lfric-built-ins:

Built-ins
---------

The basic concept of a PSyclone Built-in is described in the
:ref:`psykal-built-ins` section.  In the LFRic API, calls to
Built-ins generally follow a convention that the field/scalar written
to comes first in the argument list. LFRic Built-ins must conform to the
following rules:

1) They must have one and only one modified (i.e. written to) argument.

2) They must operate on a DoF (``operates_on = DOF`` metadata).

3) There must be at least one field in the argument list. This is so
   that we know the number of DoFs to iterate over in the PSy layer.

4) Kernel arguments must be either fields or scalars (``real``- and/or
   ``integer``-valued).

5) All field arguments to a given Built-in must be on the same
   function space. This is because all current Built-ins operate on
   DoFs and therefore all fields should have the same number. It also
   means that we can determine the number of DoFs uniquely when a
   scalar is written to;

6) Built-ins that update ``real``-valued fields can, in general, only
   read from other ``real``-valued fields, but they can take both ``real``
   and ``integer`` scalar arguments (see rule 8 for exceptions);

7) Built-ins that update ``integer``-valued fields can, in general, only
   read from other ``integer``-valued fields and take ``integer`` scalar
   arguments (see rule 8 for exceptions);

8) The only two exceptions from the rules 6) and 7) above regarding the
   same data type of "write" and "read" field arguments are Built-ins
   that convert field data from ``real`` to ``integer``, ``real_to_int_X``,
   and from ``integer`` to ``real``, ``int_to_real_X``.

The Built-ins supported for the LFRic API are listed in the related
subsections, grouped first by the data type of fields they operate on
(:ref:`real-valued <lfric-built-ins-real>` and
:ref:`integer-valued <lfric-built-ins-int>`) and then by the mathematical
operation they perform.

The field arguments in Built-ins are the derived types that represent the
:ref:`LFRic fields <lfric-field>`, however mathematical operations are
actually performed on the data of the *field proxies* (e.g.
``field1_proxy%data(:)``). For instance, ``X_plus_Y`` Built-in adds the
values of two fields accessed via their proxies in a loop over DoFs:

.. code-block:: fortran

  DO df=loop0_start,loop0_stop
     field3_proxy%data(df) = field1_proxy%data(df) + field2_proxy%data(df)

where the precise values of the loop limits depend on the use of
:ref:`distributed memory <psykal_usage>`,
:ref:`annexed DoFs <lfric-annexed_dofs>` or both.

As described in the PSy-layer :ref:`Argument Intents
<lfric-psy-arg-intents>` section, the Fortran intent of LFRic
:ref:`field <lfric-field>` objects is always ``in`` (because it is only
the data pointed to from within the object that is modified). The field
or scalar that has its data modified by a Built-in is marked in **bold**.

For clarity, the calculation performed by each Built-in is described using
Fortran array syntax without the details about field proxies. The actual
implementation of the Built-in may change in future (*e.g.* it could be
implemented by PSyclone generating a call to an optimised Maths library).

.. _lfric-api-built-ins-metadata:

Metadata
++++++++

The code below outlines the elements of the LFRic API Built-in
metadata for the Built-ins that update a ``real``-valued field,
1) 'meta_args', 2) 'operates_on' and 3) 'procedure'::

  type, public, extends(kernel_type) :: aX_plus_bY
     private
     type(arg_type) :: meta_args(5) = (/                              &
          arg_type(GH_FIELD,  GH_REAL, GH_WRITE, ANY_SPACE_1),        &
          arg_type(GH_SCALAR, GH_REAL, GH_READ              ),        &
          arg_type(GH_FIELD,  GH_REAL, GH_READ,  ANY_SPACE_1),        &
          arg_type(GH_SCALAR, GH_REAL, GH_READ              ),        &
          arg_type(GH_FIELD,  GH_REAL, GH_READ,  ANY_SPACE_1)         &
          /)
     integer :: operates_on = DOF
   contains
     procedure, nopass :: aX_plus_bY_code
  end type aX_plus_bY

As can be seen, the metadata for a Built-in kernel is a subset of that
for a :ref:`user-defined Kernel <lfric-api-kernel-metadata>` with the
exception that ``operates_on`` must be ``DOF`` instead of ``CELL_COLUMN``.

The metadata for the LFRic Built-ins that update an ``integer``-valued
field is similar::

  !> ifield3 = ifield1 + ifield2
  type, public, extends(kernel_type) :: int_X_plus_Y
     private
     type(arg_type) :: meta_args(3) = (/                              &
          arg_type(GH_FIELD, GH_INTEGER, GH_WRITE, ANY_SPACE_1),      &
          arg_type(GH_FIELD, GH_INTEGER, GH_READ,  ANY_SPACE_1),      &
          arg_type(GH_FIELD, GH_INTEGER, GH_READ,  ANY_SPACE_1)       &
          /)
     integer :: operates_on = DOF
   contains
     procedure, nopass :: int_X_plus_Y_code
  end type int_X_plus_Y

.. _lfric-built-ins-dtype-access:

Valid Data Types and Access Modes
#################################

The allowed data types and accesses for arguments in LFRic Built-in
kernels are a bit different than for the
:ref:`user-defined Kernels <lfric-kernel-valid-access>` and
are listed in the table below.

.. tabularcolumns:: |l|l|l|l|

+---------------+---------------------+----------------+--------------------+
| Argument Type | Data Type           | Function Space | Access Type        |
+===============+=====================+================+====================+
| GH_SCALAR     | GH_INTEGER          | n/a            | GH_READ            |
+---------------+---------------------+----------------+--------------------+
| GH_SCALAR     | GH_REAL             | n/a            | GH_READ,           |
|               |                     |                | GH_REDUCTION       |
+---------------+---------------------+----------------+--------------------+
| GH_FIELD      | GH_REAL, GH_INTEGER | ANY_SPACE_<n>  | GH_READ, GH_WRITE, |
|               |                     |                | GH_READWRITE       |
+---------------+---------------------+----------------+--------------------+

.. note:: Since the LFRic infrastructure does not currently support
          ``integer`` reductions, ``integer`` scalar arguments in Built-ins
          are restricted to having read-only access. Also, ``logical``
          scalar arguments are not permitted.

.. _lfric-built-ins-names:

Naming scheme
+++++++++++++

The supported Built-ins in the LFRic API are named according to the
scheme presented below. Any new Built-in needs to comply with these rules.

1) Ordering of arguments in Built-ins calls follows
   *LHS (result) <- RHS (operation on arguments)*
   direction, except where a Built-in returns the *LHS* result to one of
   the *RHS* arguments. In that case ordering of arguments remains as in
   the *RHS* expression, with the returning *RHS* argument written as close
   to the *LHS* as it can be without affecting the mathematical expression.

2) Field names begin with upper case in short form (e.g. **X**, **Y**,
   **Z**) and any case in long form (e.g. **Field1**, **field**).

3) Scalar names begin with lower case:  e.g. **a**, **b**, are **scalar1**,
   **scalar2**. Special names for scalars are: **constant** (or **c**),
   **innprod** (inner/scalar product of two fields) and **sumfld**
   (sum of a field).

4) Arguments in Built-ins variable declarations and constructs (PSyclone
   Fortran and Python definitions):

   1) Are always  written in long form and lower case (e.g. **field1**,
      **field2**, **scalar1**, **scalar2**);

   2) *LHS* result arguments are always listed first;

   3) *RHS* arguments are listed in order of appearance in the mathematical
      expression, except when one of them is the *LHS* result.

5) Built-ins names in Fortran consist of:

   1) *RHS* arguments in short form (e.g. **X**, **Y**, **a**, **b**) only;

   2) Descriptive name of mathematical operation on *RHS* arguments in the
      form  ``<operationname>_<RHSargs>`` or
      ``<RHSargs>_<operationname>_<RHSargs>``;

   3) Prefix ``"inc_"`` where the result is returned to one of the *RHS*
      arguments (i.e. ``"inc_"<RHSargs>_<operationname>_<RHSargs>``);

   4) Prefix ``"int_"`` for the Built-in operations on the ``integer``-valued
      field arguments (i.e. ``"int_inc_"<RHSargs>_<operationname>_<RHSargs>``).

6) Built-ins names in Python definitions are similar to their Fortran
   counterparts, with a few differences:

   1) Operators and *RHS* arguments are all in upper case (e.g. **X**,
      **Y**, **A**, **B**, **Plus**, **Minus**);

   2) There are no underscores;

   4) Common suffix is ``"Kern"``;

   3) Common prefix is ``"LFRic"`` for the Built-in operations on the
      ``real``-valued arguments and ``"LFRicInt"`` for the Built-in
      operations on the ``integer``-valued fields.

Querying Built-in Operations
++++++++++++++++++++++++++++

Within a Python script, the (lowercase) names of all available
Built-ins in the LFRIc API can be queried using the ``BUILTIN_MAP``
dictionary object from the ``psyclone.domain.lfric.lfric_builtins``
module.

Example code:

.. highlight:: python
.. testcode::

    from psyclone.domain.lfric.lfric_builtins import BUILTIN_MAP
    
    kernel_name = "setval_x"    # example only
    if kernel_name.lower() in BUILTIN_MAP:
        print(f"Name '{kernel_name}' is a Built-in kernel.")
    else:
        print(f"Name '{kernel_name}' is not a Built-in.")

.. testoutput::

  Name 'setval_x' is a Built-in kernel.

.. _lfric-built-ins-real:

Built-in operations on ``real``-valued fields
+++++++++++++++++++++++++++++++++++++++++++++

As described :ref:`above <lfric-built-ins-dtype-access>`, Built-ins that
operate on ``real``-valued fields mandate ``GH_REAL`` as the kernel
metadata for fields and scalars.

The precision of fields and scalars, however, is determined by the
algorithm layer via precision variables as described in the :ref:`Mixed
Precision <lfric-mixed-precision>` section (see subsections on
:ref:`fields <lfric-mixed-precision-fields>` and
:ref:`scalars <lfric-mixed-precision-scalars>`).

For instance, field and scalar declarations for the ``aX_plus_Y``
Built-in that operates on ``r_solver_field_type`` and uses ``r_solver``
scalar will be::

  real(kind=r_solver), intent(in) :: ascalar
  type(r_solver_field_type), intent(in) :: zfield, xfield, yfield

Mixing precisions is not explicitly forbidden, so we may have e.g.
``X_divideby_a`` Built-in where::

  real(kind=r_def), intent(in) :: ascalar
  type(r_tran_field_type), intent(in) :: yfield, xfield

Certain Built-ins are currently restricted in the precision of the
arguments that they accept. Those that calculate the inner product
and sum of a field are restricted to ``r_def`` precision because the
scalar global reductions in the LFRic infrastructure are currently
only able to support ``field_type`` and hence have ``r_def`` precision.
In addition, all integer arguments to Built-ins are currently restricted
to ``i_def`` precision.

Addition
########

Built-ins that add (scaled) ``real``-valued fields and return the result
as a ``real``-valued field are denoted with the keyword **plus**.

X_plus_Y
^^^^^^^^

**X_plus_Y** (**field3**, *field1*, *field2*)

Sums two fields and stores the result in the third field (``Z = X + Y``)::

  field3(:) = field1(:) + field2(:)

inc_X_plus_Y
^^^^^^^^^^^^

**inc_X_plus_Y** (**field1**, *field2*)

Adds the second field to the first and returns it (``X = X + Y``)::

  field1(:) = field1(:) + field2(:)

a_plus_X
^^^^^^^^

**a_plus_X** (**field2**, *rscalar*, *field1*)

Adds a ``real`` scalar value to all elements of a field and stores
the result in another field (``Y = a + X``)::

  field2(:) = rscalar + field1(:)

inc_a_plus_X
^^^^^^^^^^^^

**inc_a_plus_X** (*rscalar*, **field**)

Adds a ``real`` scalar value to all elements of a field and returns
the field (``X = a + X``)::

  field(:) = rscalar + field(:)

aX_plus_Y
^^^^^^^^^

**aX_plus_Y** (**field3**, *rscalar*, *field1*, *field2*)

Performs ``Z = aX + Y``::

  field3(:) = rscalar*field1(:) + field2(:)

inc_aX_plus_Y
^^^^^^^^^^^^^

**inc_aX_plus_Y** (*rscalar*, **field1**, *field2*)

Performs ``X = aX + Y`` (increments the first field)::

  field1(:) = rscalar*field1(:) + field2(:)

inc_X_plus_bY
^^^^^^^^^^^^^

**inc_X_plus_bY** (**field1**, *rscalar*, *field2*)

Performs ``X = X + bY`` (increments the first field)::

  field1(:) = field1(:) + rscalar*field2(:)

aX_plus_bY
^^^^^^^^^^

**aX_plus_bY** (**field3**, *rscalar1*, *field1*, *rscalar2*, *field2*)

Performs ``Z = aX + bY``::

  field3(:) = rscalar1*field1(:) + rscalar2*field2(:)

inc_aX_plus_bY
^^^^^^^^^^^^^^

**inc_aX_plus_bY** (*rscalar1*, **field1**, *rscalar2*, *field2*)

Performs ``X = aX + bY`` (increments the first field)::

  field1(:) = rscalar1*field1(:) + rscalar2*field2(:)

aX_plus_aY
^^^^^^^^^^

**aX_plus_aY** (**field3**, *rscalar*, *field1*, *field2*)

Performs ``Z = aX + aY = a(X + Y)``::

  field3(:) = rscalar*(field1(:) + field2(:))

Subtraction
###########

Built-ins which subtract (scaled) ``real``-valued  fields and return the
result as a ``real``-valued field are denoted with the keyword **minus**.

X_minus_Y
^^^^^^^^^

**X_minus_Y** (**field3**, *field1*, *field2*)

Subtracts the second field from the first and returns the result in the
third field (``Z = X - Y``)::

  field3(:) = field1(:) - field2(:)

inc_X_minus_Y
^^^^^^^^^^^^^

**inc_X_minus_Y** (**field1**, *field2*)

Subtracts the second field from the first and returns it (``X = X - Y``)::

  field1(:) = field1(:) - field2(:)

a_minus_X
^^^^^^^^^

**a_minus_X** (**field2**, *rscalar*, *field1*)

Subtracts all elements of a field from a ``real`` scalar value and
stores the result in another field (``Y = a - X``)::

  field2(:) = rscalar - field1(:)

inc_a_minus_X
^^^^^^^^^^^^^

**inc_a_minus_X** (*rscalar*, **field**)

Subtracts all elements of a field from a ``real`` scalar value and
returns the field (``X = a - X``)::

  field(:) = rscalar - field(:)

X_minus_a
^^^^^^^^^

**X_minus_a** (**field2**, *field1*, *rscalar*)

Subtracts a ``real`` scalar value from all elements of a field and
stores the result in another field (``Y = X - a``)::

  field2(:) = field1(:) - rscalar

inc_X_minus_a
^^^^^^^^^^^^^

**inc_X_minus_a** (**field**, *rscalar*)

Subtracts a ``real`` scalar value from all elements of a field and
returns the field (``X = X - a``)::

  field(:) = field(:) - rscalar

aX_minus_Y
^^^^^^^^^^

**aX_minus_Y** (**field3**, *rscalar*, *field1*, *field2*)

Performs ``Z = aX - Y``::

  field3(:) = rscalar*field1(:) - field2(:)

X_minus_bY
^^^^^^^^^^

**X_minus_bY** (**field3**, *field1*, *rscalar*, *field2*)

Performs ``Z = X - bY``::

  field3(:) = field1(:) - rscalar*field2(:)

inc_X_minus_bY
^^^^^^^^^^^^^^

**inc_X_minus_bY** (**field1**, *rscalar*, *field2*)

Performs ``X = X - bY`` (decrements the first field)::

  field1(:) = field1(:) - rscalar*field2(:)

aX_minus_bY
^^^^^^^^^^^

**aX_minus_bY** (**field3**, *rscalar1*, *field1*, *rscalar2*, *field2*)

Performs ``Z = aX - bY``::

  field3(:) = rscalar1*field1(:) - rscalar2*field2(:)

Multiplication
##############

Built-ins which multiply (scaled) ``real``-valued fields and return the
result as a ``real``-valued field are denoted with the keyword **times**.

X_times_Y
^^^^^^^^^

**X_times_Y** (**field3**, *field1*, *field2*)

Multiplies two fields DoF by DoF and returns the result in a
third field (``Z = X*Y``)::

  field3(:) = field1(:)*field2(:)

inc_X_times_Y
^^^^^^^^^^^^^

**inc_X_times_Y** (**field1**, *field2*)

Multiplies the first field by the second and returns it (``X = X*Y``)::

  field1(:) = field1(:)*field2(:)

inc_aX_times_Y
^^^^^^^^^^^^^^

**inc_aX_times_Y** (*rscalar*, **field1**, *field2*)

Performs ``X = a*X*Y`` (increments the first field)::

  field1(:) = rscalar*field1(:)*field2(:)

Scaling
#######

Built-ins which scale ``real``-valued fields are technically cases of
multiplying a ``real``-valued field by a ``real`` scalar and are hence
also denoted with the keyword **times**.

a_times_X
^^^^^^^^^

**a_times_X** (**field2**, *rscalar*, *field1*)

Multiplies a field by a ``real`` scalar value and stores the result
in another field (``Y = a*X``)::

  field2(:) = rscalar*field1(:)

inc_a_times_X
^^^^^^^^^^^^^

**inc_a_times_X** (*rscalar*, **field**)

Multiplies a field by a ``real`` scalar value and returns the
field (``X = a*X``)::

  field(:) = rscalar*field(:)

Division
########

Built-ins which divide ``real``-valued fields and return the result
as a ``real``-valued field are denoted with the keyword **divideby**.

X_divideby_Y
^^^^^^^^^^^^

**X_divideby_Y** (**field3**, *field1*, *field2*)

Divides the first field by the second field, DoF by DoF, and stores the
result in the third field (``Z = X/Y``)::

  field3(:) = field1(:)/field2(:)

inc_X_divideby_Y
^^^^^^^^^^^^^^^^

**inc_X_divideby_Y** (**field1**, *field2*)

Divides the first field by the second and returns it (``X = X/Y``)::

  field1(:) = field1(:)/field2(:)

X_divideby_a
^^^^^^^^^^^^

**X_divideby_a** (**field2**, *field1*, *rscalar*)

Divides each field element by a ``real`` scalar value and stores
the result in another field (``Y = X/a``)::

  field2(:) = field1(:)/rscalar

inc_X_divideby_a
^^^^^^^^^^^^^^^^

**inc_X_divideby_a** (**field**, *rscalar*)

Divides each field element by a ``real`` scalar value and returns
the field (``X = X/a``)::

  field(:) = field(:)/rscalar

Inverse scaling
###############

Built-ins which perform inverse scaling of ``real``-valued fields are
also denoted with the keyword **divideby** as they divide a ``real``
scalar by elements of a ``real``-valued field.

a_divideby_X
^^^^^^^^^^^^

**a_divideby_X** (**field2**, *rscalar*, *field1*)

Divides a ``real`` scalar value by each field element and stores the
result in another field (``Y = a/X``)::

  field2(:) = rscalar/field1(:)

inc_a_divideby_X
^^^^^^^^^^^^^^^^

**inc_a_divideby_X** (*rscalar*, **field**)

Divides a ``real`` scalar value by each field element and returns
the field (``X = a/X``)::

  field(:) = rscalar/field(:)

Setting to a value
##################

Built-ins which set ``real``-valued field elements to some ``real``
value are denoted with the keyword **setval**.

setval_c
^^^^^^^^

**setval_c** (**field**, *constant*)

Sets all elements of a field *field* to a ``real`` scalar
*constant* (``X = c``)::

  field(:) = constant

setval_X
^^^^^^^^

**setval_X** (**field2**, *field1*)

Sets a field *field2* equal (DoF per DoF) to another field
*field1* (``Y = X``)::

  field2(:) = field1(:)

setval_random
^^^^^^^^^^^^^

**setval_random** (**field**)

Fills all elements of a field *field* using a sequence of ``real``,
pseudo-random numbers in the interval ``0 <= x < 1``::

  do df = 1, ndofs
    field(df) = RAND()
  end do

where ``RAND()`` is some function that returns a new pseudo-random number
each time it is called.

Due to different parallel elements using independent random-number generator
streams, this built-in has ``OPERATES_ON=owned_dof``. This will prevent
optimisations such as redundant computation (including the global
``COMPUTE_ANNEXED_DOFS`` option).


.. warning:: This Built-in is implemented using the Fortran ``random_number``
	     intrinsic. Therefore no guarantee is made as to the quality of
	     the sequence of pseudo-random numbers, especially when running
	     in parallel.

Raising to power
################

Built-ins which raise ``real``-valued field elements to an exponent are
denoted with the keyword **powreal** for a ``real`` exponent or **powint**
for an ``integer`` exponent.

inc_X_powreal_a
^^^^^^^^^^^^^^^

**inc_X_powreal_a** (**field**, *rscalar*)

Raises a field to a ``real`` scalar value and returns the
field (``X = X**a``)::

  field(:) = field(:)**rscalar

inc_X_powint_n
^^^^^^^^^^^^^^

**inc_X_powint_n** (**field**, *iscalar*)

Raises a field to an ``integer`` scalar value and returns
the field (``X = X**n``)::

  field(:) = field(:)**iscalar

where ``iscalar`` is an ``integer`` scalar of ``i_def`` precision.

Inner product
#############

Built-ins which calculate the inner product of two ``real``-valued fields
or of a ``real``-valued field with itself and return the result as a
``real`` scalar are denoted with the keyword **innerproduct**.

.. note:: When used with distributed memory these Built-ins will
          trigger the addition of a global sum which may affect the
          performance and/or scalability of the code.
          Also, whilst the fields in these Built-ins can be of any
          supported ``real`` :ref:`precision <lfric-mixed-precision>`,
          the only currently supported precision for the global
          reductions in the LFRic infrastructure is ``r_def``, hence
          the result will be converted accordingly.

X_innerproduct_Y
^^^^^^^^^^^^^^^^

**X_innerproduct_Y** (**innprod**, *field1*, *field2*)

Computes the inner product of two fields, *field1*
and *field2*, *i.e.*::

  innprod = SUM(field1(:)*field2(:))

where **innprod** is a ``real`` scalar of ``r_def`` precision.

X_innerproduct_X
^^^^^^^^^^^^^^^^

**X_innerproduct_X** (**innprod**, *field*)

Computes the inner product of the field *field1* by itself, *i.e.*::

  innprod = SUM(field(:)*field(:))

where **innprod** is a ``real`` scalar of ``r_def`` precision.

Sum of elements
###############

A Built-in which sums the elements of a ``real``-valued field and returns
the result as a ``real`` scalar is denoted with the keyword **sum**.

.. note:: When used with distributed memory this Built-in will trigger
          the addition of a global sum which may affect the
          performance and/or scalability of the code.
          Also, whilst the fields in these Built-ins can be of any
          supported ``real`` :ref:`precision <lfric-mixed-precision>`,
          the only currently supported precision for the global
          reductions in the LFRic infrastructure is ``r_def``, hence
          the result will be converted accordingly.

sum_X
^^^^^

**sum_X** (**sumfld**, *field*)

Sums all of the elements of the field *field* and returns the result
in the ``real`` scalar variable *sumfld*::

  sumfld = SUM(field(:))

where **sumfld** is a ``real`` scalar of ``r_def`` precision.

Sign of elements
################

A Built-in which returns the sign of a ``real``-valued field is denoted
with the keyword **sign**.

sign_X
^^^^^^

**sign_X** (**field2**, *rscalar*, *field1*)

Returns the sign of a ``real``-valued field, e.g. in Fortran:
``Y = sign(a, X)``. Here ``a`` is a ``real`` scalar and ``Y`` and ``X``
are ``real``-valued fields. The results are ``a`` for ``X >= 0`` and
``-a`` for ``X < 0``::

  field2(:) = SIGN(rscalar, field1(:))

DoF-wise maximum of elements
############################

Built-ins which return the DoF-wise maximum of a ``real`` scalar and
a ``real``-valued field are denoted with the keyword **max**.

max_aX
^^^^^^

**max_aX** (**field2**, *rscalar*, *field1*)

Returns maximum of *rscalar* and each element of the field *field1* as
the second field **field2** (``Y = max(a, X)``)::

  field2(:) = MAX(rscalar, field1(:))

inc_max_aX
^^^^^^^^^^

**inc_max_aX** (*rscalar*, **field**)

Returns maximum of *rscalar* and each element of the field *field* in
the same field (``X = max(a, X)``)::

  field(:) = MAX(rscalar, field(:))

DoF-wise minimum of elements
############################

Built-ins which return the DoF-wise minimum of a ``real`` scalar and
a ``real``-valued field are denoted with the keyword **min**.

min_aX
^^^^^^

**min_aX** (**field2**, *rscalar*, *field1*)

Returns minimum of *rscalar* and each element of the field *field1* as
the second field **field2** (``Y = min(a, X)``)::

  field2(:) = MIN(rscalar, field1(:))

inc_min_aX
^^^^^^^^^^

**inc_min_aX** (*rscalar*, **field**)

Returns minimum of *rscalar* and each element of the field *field* in
the same field (``X = min(a, X)``)::

  field(:) = MIN(rscalar, field(:))

Global minimum and maximum field-element values
###############################################

Built-ins which scan through all elements of a field and return its
maximum or minimum value.

minval_X
^^^^^^^^

**minval_X** (*rscalar*, **field**)

Returns the minimum value held in the field *field*::

  rscalar = MINVAL(field(:))

maxval_X
^^^^^^^^

**maxval_X** (*rscalar*, **field**)

Returns the maximum value held in the field *field*::

  rscalar = MAXVAL(field(:))

Conversion of ``real`` field elements
#####################################

Built-ins which take a ``real`` field for conversion to a field of
a different datatype or precision are denoted by the datatype that the
input ``real`` field will be converted to. A Built-in that converts a
``real`` to an ``integer`` field is denoted by the phrase **to_int**.
Likewise, a Built-in that converts a ``real`` to a ``real`` field is
denoted by the phrase **to_real**.

.. _real-to-int-built-in:

real_to_int_X
^^^^^^^^^^^^^

**real_to_int_X** (**ifield2**, *field1*)

Converts ``real``-valued field elements to ``integer``-valued field
elements, e.g. in Fortran this would be: ``Y = INT(X, kind=i_<prec>)``.
Here ``Y`` is an ``integer``-valued field and ``X`` is the
``real``-valued field being converted::

  ifield2(:) = INT(field1(:), kind=i_<prec>)

where **ifield2** is currently the only supported ``integer``-valued field
type in LFRic (``integer_field_type`` of ``i_def`` precision) and a ``real``
-valued field *field1* can be of any :ref:`supported precisions
<lfric-mixed-precision>` for ``GH_REAL`` fields (e.g. ``r_tran`` for
``r_tran_field_type``).

.. _real-to-real-built-in:

real_to_real_X
^^^^^^^^^^^^^^

**real_to_real_X** (**field2**, *field1*)

Converts ``real``-valued field elements from a precision ``r_<prec>``
to ``real``-valued field elements of a differing precision ``r_<prec>``,
e.g. in Fortran this would be: ``Y = REAL(X, kind=r_<prec>)``. Here ``Y``
and ``X`` are both ``real``-valued fields, with ``X`` being converted
to the precision of ``Y``::

  field2(:) = REAL(field1(:), kind=r_<prec>)

**field2** and *field1* are ``real``-valued fields of any :ref:`supported
precisions <lfric-mixed-precision>` for ``GH_REAL`` fields (e.g. ``r_tran``
for ``r_tran_field_type``).

.. _lfric-built-ins-int:

Built-in operations on ``integer``-valued fields
++++++++++++++++++++++++++++++++++++++++++++++++

The number of supported Built-in operations on the ``integer``-valued
fields is not as large as for their ``real`` counterparts as not all
mathematical operations on ``integer``-valued fields make sense.

As described :ref:`above <lfric-built-ins-dtype-access>`, Built-ins that
operate on ``integer``-valued fields mandate ``GH_INTEGER`` as the kernel
metadata for fields and scalars. Both ``integer`` scalar arguments and
``integer``-valued fields can only currently have ``i_def`` precision,
as described in the :ref:`Mixed Precision <lfric-mixed-precision>` section.

For instance, field and scalar declarations for the ``X_minus_a``
Built-in will be::

  integer(kind=i_def), intent(in) :: ascalar
  type(integer_field_type), intent(in) :: yfield, xfield

Addition
########

Built-ins that add ``integer``-valued fields and return the result as
an ``integer``-valued field are denoted with the keyword **plus** and
the prefix **int**.

int_X_plus_Y
^^^^^^^^^^^^

**int_X_plus_Y** (**ifield3**, *ifield1*, *ifield2*)

Sums two fields and stores the result in the third field (``Z = X + Y``)::

  ifield3(:) = ifield1(:) + ifield2(:)

int_inc_X_plus_Y
^^^^^^^^^^^^^^^^

**int_inc_X_plus_Y** (**ifield1**, *ifield2*)

Adds the second field to the first and returns it (``X = X + Y``)::

  ifield1(:) = ifield1(:) + ifield2(:)

int_a_plus_X
^^^^^^^^^^^^

**int_a_plus_X** (**ifield2**, *iscalar*, *ifield1*)

Adds an ``integer`` scalar value to all elements of a field and stores
the result in another field (``Y = a + X``)::

  ifield2(:) = iscalar + ifield1(:)

int_inc_a_plus_X
^^^^^^^^^^^^^^^^

**int_inc_a_plus_X** (*iscalar*, **ifield**)

Adds an ``integer`` scalar value to all elements of a field and returns
the field (``X = a + X``)::

  ifield(:) = iscalar + ifield(:)

Subtraction
###########

Built-ins which subtract ``integer``-valued fields and return the result
as an ``integer``-valued field are denoted with the keyword **minus**
and the prefix **int**.

int_X_minus_Y
^^^^^^^^^^^^^

**int_X_minus_Y** (**ifield3**, *ifield1*, *ifield2*)

Subtracts the second field from the first and returns the result in the
third field (``Z = X - Y``)::

  ifield3(:) = ifield1(:) - ifield2(:)

int_inc_X_minus_Y
^^^^^^^^^^^^^^^^^

**int_inc_X_minus_Y** (**ifield1**, *ifield2*)

Subtracts the second field from the first and returns it (``X = X - Y``)::

  ifield1(:) = ifield1(:) - ifield2(:)

int_a_minus_X
^^^^^^^^^^^^^

**int_a_minus_X** (**ifield2**, *iscalar*, *ifield1*)

Subtracts all elements of a field from an ``integer`` scalar value and
stores the result in another field (``Y = a - X``)::

  ifield2(:) = iscalar - ifield1(:)

int_inc_a_minus_X
^^^^^^^^^^^^^^^^^

**int_inc_a_minus_X** (*iscalar*, **ifield**)

Subtracts all elements of a field from an ``integer`` scalar value and
returns the field (``X = a - X``)::

  ifield(:) = iscalar - ifield(:)

int_X_minus_a
^^^^^^^^^^^^^

**int_X_minus_a** (**ifield2**, *ifield1*, *iscalar*)

Subtracts an ``integer`` scalar value from all elements of a field and
stores the result in another field (``Y = X - a``)::

  ifield2(:) =  ifield1(:) - iscalar

int_inc_X_minus_a
^^^^^^^^^^^^^^^^^

**int_inc_X_minus_a** (**ifield**, *iscalar*)

Subtracts an ``integer`` scalar value from all elements of a field and
returns the field (``X = X - a``)::

  ifield(:) =  ifield(:) - iscalar

Multiplication
##############

Built-ins which multiply ``integer``-valued fields and return the result
as an ``integer``-valued field are denoted with the keyword **times**
and the prefix **int**.

int_X_times_Y
^^^^^^^^^^^^^

**int_X_times_Y** (**ifield3**, *ifield1*, *ifield2*)

Multiplies two fields DoF by DoF and returns the result in a
third field (``Z = X*Y``)::

  ifield3(:) = ifield1(:)*ifield2(:)

int_inc_X_times_Y
^^^^^^^^^^^^^^^^^

**int_inc_X_times_Y** (**ifield1**, *ifield2*)

Multiplies the first field by the second and returns it (``X = X*Y``)::

  ifield1(:) = ifield1(:)*ifield2(:)

Scaling
#######

Built-ins which scale ``integer``-valued fields are denoted with the keyword
**times** and prefixed by the keyword **int**.

int_a_times_X
^^^^^^^^^^^^^

**int_a_times_X** (**ifield2**, *iscalar*, *ifield1*)

Multiplies a field by an ``integer`` scalar and stores the result
in another field (``Y = a*X``)::

  ifield2(:) = iscalar*ifield1(:)

int_inc_a_times_X
^^^^^^^^^^^^^^^^^

**int_inc_a_times_X** (*iscalar*, **ifield**)

Multiplies a field by an ``integer`` scalar value and returns the
field (``X = a*X``)::

  ifield(:) = iscalar*ifield(:)

Setting to a value
##################

Built-ins which set ``integer``-valued field elements to some ``integer``
value are denoted with the keyword **setval** and the prefix **int**.

int_setval_c
^^^^^^^^^^^^

**int_setval_c** (**ifield**, *constant*)

Sets all elements of a field *ifield* to an ``integer`` scalar
*constant* (``X = c``)::

  ifield(:) = constant

int_setval_X
^^^^^^^^^^^^

**int_setval_X** (**ifield2**, *ifield1*)

Sets a field *ifield2* equal (DoF per DoF) to another field
*ifield1* (``Y = X``)::

  ifield2(:) = ifield1(:)

Sign of elements
################

A Built-in which returns the sign of an ``integer``-valued field
is denoted with the keyword **sign** and the prefix **int**.

int_sign_X
^^^^^^^^^^

**int_sign_X** (**ifield2**, *iscalar*, *ifield1*)

Returns the sign of an ``integer``-valued field, e.g. in Fortran:
``Y = sign(a, X)``. Here ``a`` is an ``integer`` scalar and ``Y``
and ``X`` are ``integer``-valued fields.
The results are ``a`` for ``X >= 0`` and ``-a`` for ``a < 0``::

  ifield2(:) = SIGN(iscalar, ifield1(:))

DoF-wise maximum of elements
############################

Built-ins which return the DoF-wise maximum of an ``integer`` scalar
and an ``integer``-valued field are denoted with the keyword **max**.

int_max_aX
^^^^^^^^^^

**int_max_aX** (**ifield2**, *iscalar*, *ifield1*)

Returns maximum of *iscalar* and each element of the field *ifield1* as
the second field **ifield2** (``Y = max(a, X)``)::

  ifield2(:) = MAX(iscalar, ifield1(:))

int_inc_max_aX
^^^^^^^^^^^^^^

**int_inc_max_aX** (*iscalar*, **ifield**)

Returns maximum of *iscalar* and each element of the field *ifield* in
the same field (``X = max(a, X)``)::

  ifield(:) = MAX(iscalar, ifield(:))

DoF-wise minimum of elements
############################

Built-ins which return the DoF-wise minimum of an ``integer`` scalar
and an ``integer``-valued field are denoted with the keyword **min**.

int_min_aX
^^^^^^^^^^

**int_min_aX** (**ifield2**, *iscalar*, *ifield1*)

Returns minimum of *iscalar* and each element of the field *ifield1* as
the second field **ifield2** (``Y = min(a, X)``)::

  ifield2(:) = MIN(iscalar, ifield1(:))

int_inc_min_aX
^^^^^^^^^^^^^^

**int_inc_min_aX** (*iscalar*, **ifield**)

Returns minimum of *iscalar* and each element of the field *ifield* in
the same field (``X = min(a, X)``)::

  ifield(:) = MIN(iscalar, ifield(:))

Conversion of ``integer`` to ``real`` field elements
####################################################

A Built-in which takes an ``integer`` field and converts it to
a ``real`` field is denoted by the phrase **to_real**.

.. _int-to-real-built-in:

int_to_real_X
^^^^^^^^^^^^^

**int_to_real_X** (**field2**, *ifield1*)

Converts ``integer``-valued field elements to ``real``-valued field
elements, e.g. in Fortran this would be ``Y = REAL(X, kind=r_<prec>)``.
Here ``Y`` is a ``real``-valued field and ``X`` is the
``integer``-valued field being converted::

  field2(:) = REAL(ifield1(:), kind=r_<prec>)

where **ifield1** is currently the only supported ``integer``-valued
field type in LFRic (``integer_field_type`` of ``i_def`` precision).
The ``real``-valued **field1** can be of any :ref:`supported
precisions <lfric-mixed-precision>` for ``GH_REAL`` fields, hence
``r_<prec>`` is determined from the algorithm layer (e.g.
``r_solver`` for ``r_solver_field_type``).

Boundary Conditions
-------------------

In the LFRic API, boundary conditions for a field or LMA operator can
be enforced by the algorithm developer by calling the Kernels
``enforce_bc_type`` or ``enforce_operator_bc_type``,
respectively. These kernels take a field or operator as input and apply
boundary conditions. For example::

  call invoke( kernel_type(field1, field2),      &
               enforce_bc_type(field1),          &
               kernel_with_op_type(field1, op1), &
               enforce_operator_bc_type(op1)     &
             )

The particular boundary conditions that are applied are not known by
PSyclone, PSyclone simply recognises these kernels by their names and passes
pre-specified dofmap and boundary_value arrays into the kernel
implementations, the contents of which are set by the LFRic
infrastructure.

Up to and including version 1.4.0 of PSyclone, boundary conditions
were applied automatically after a call to ``matrix_vector_type`` if
the field arguments were on a vector function space (one of ``W1``,
``W2``, ``W2H``, ``W2V`` or ``W2broken``). With the subsequent introduction
of the ability to apply boundary conditions to operators this functionality
is no longer required and has been removed.

Example ``eg4`` in the ``examples/lfric`` directory includes a call
to ``enforce_bc_kernel_type`` so can be used to see the boundary condition
code that is added by PSyclone. See the ``README`` in the
``examples/lfric`` directory for instructions on how to run this
example.

An example of applying boundary conditions to an operator is the kernel
``enforce_operator_bc_kernel_mod.F90`` in the
``<PSYCLONEHOME>/src/psyclone/tests/test_files/lfric`` directory.
Since operators are discontinuous quantities, updating their values can
be safely performed in parallel (see Section :ref:`lfric-kernel`).
The ``GH_READWRITE`` access is used for updating discontinuous operators
(see subsection :ref:`lfric-kernel-valid-access` for more details).

.. _lfric-conventions:

Conventions
-----------

The naming of LFRic API kernels and associated entities (types,
subroutines and modules) follows the convention that the kernel file is
named ``<name>_mod.[fF90]``, the module inside the kernel file is
``<name>_mod``, the name of the kernel metadata in the module is
``<name>_type`` and the name of the kernel subroutine in the module is
``<name>_code``.
However, PSyclone does not need
this convention to be followed apart from the stub generator (see the
:ref:`stub-generation` Section ) where the name of the metadata to be
parsed is determined from the module name.

The contents of the metadata is also usually declared private but this
does not affect PSyclone.

Finally, the ``procedure`` metadata (located within the kernel
metadata) usually has ``nopass`` specified but again this is ignored
by PSyclone.

.. _lfric-api-configuration:

Configuration
-------------

The general and the LFRic-API-specific configuration options are described
in the :ref:`Configuration <configuration>` section.

.. _lfric-annexed_dofs:

Annexed DoFs
++++++++++++

When a kernel operates on DoFs (rather than cell-columns) for a continuous
field using distributed memory, PSyclone need only ensure that DoFs owned by a
processor are computed. However, for continuous fields, shared DoFs at
the boundary between processors must be replicated (as different cells
share the same DoF). Only one processor can own a DoF, therefore
processors will have continuous fields which contain DoFs that the
processor does not own. These unowned DoFs are called `annexed` in the
LFRic API and are a separate, but related, concept to field halos.

When a kernel that operates on a cell-column needs to read a
continuous field then the annexed DoFs must be up-to-date on all
processors. If they are not then a halo exchange must be
added. Currently PSyclone defaults, for kernels which iterate over
DoFs, to iterating over only owned DoFs. This behaviour can be changed
by setting `COMPUTE_ANNEXED_DOFS` to ``true`` in the `lfric`
section of the configuration file (see the :ref:`configuration`
section). PSyclone will then generate code to iterate over both owned
and annexed DoFs, thereby reducing the number of halo exchanges
required (at the expense of redundantly computing annexed DoFs). For
more details please refer to the :ref:`lfric-developers`
developers section.

.. _lfric-run-time-checks:

Run-time Checks
+++++++++++++++

PSyclone performs static consistency checks where possible. When this
is not possible PSyclone can generate run-time checks. As there may be
performance costs associated with run-time checks they may be switched
on or off by the `RUN_TIME_CHECKS` option in the configuration file
(or by using the ``--config-opts`` command line option to overwrite
the setting in the configuration file). The value of `RUN_TIME_CHECKS`
must be one of:

- `none` No runtime checks will be added (default)
- `warn` Runtime checks will be added, and violations will cause a warning
  message to be logged.
- `error` Runtime checks will be added, and violations will cause an error
  message to be logged. The application will then abort.

Currently run-time checks can be generated to:

1) Check that a field with a read-only function space (see section
   :ref:`lfric-ro-function-space`) is not modified by a kernel. This is
   enforced by checking that all fields that are marked (in kernel
   metadata) as being updated by a kernel are not on a read-only function
   space. A second check that is required for fields on read-only
   function spaces is to ensure that the halo is clean before it is accessed.
   This check is currently implemented within the LFRic
   infrastructure halo exchange call (that the PSyclone LFRic API places
   at appropriate locations). If the halo is clean then the halo exchange
   will not be called. However, if the halo is not clean then the
   resulting halo exchange call will cause the infrastructure to raise an
   error (because the field is on a read-only space).

2) Check that the function space of a field is consistent with the
   kernel function space metadata that the field's data is passed
   into. For example, if kernel metadata specifies that a field is on
   the ``W2`` function space then a run-time check is added to ensure that
   the field object passed into the PSy layer is indeed on that space.
   For more general kernel function space metadata, such as
   `ANY_DISCONTINUOUS_SPACE_*` then a run-time check is added to
   ensure that the field is on one of the discontinuous function
   spaces supported in the LFRic API.

.. _lfric-datatype-kind:

Supported Data Types and Default Kind
+++++++++++++++++++++++++++++++++++++

The LFRic API supports three Fortran primitive (intrinsic) data
types, ``real``, ``integer`` and ``logical`` (listed in the
`supported_fortran_datatypes` section of the :ref:`PSyclone
configuration file <configuration>`). All three data types are used
for :ref:`scalars <lfric-scalar>`. :ref:`Fields <lfric-field>` and
:ref:`field vectors <lfric-field-vector>` are allowed to have ``real``
and ``integer`` data. :ref:`Operators <lfric-operator>` and
:ref:`column-wise operators <lfric-cma-operator>` are only allowed to
have ``real`` data. These supported primitive types are linked to the
respective :ref:`kernel data type <lfric-kernel-valid-data-type>`
metadata descriptors, ``GH_REAL`` and ``GH_INTEGER``.

The default kind (precision) for these supported data types is
set to ``r_def``, ``i_def`` and ``l_def``, respectively, in the
``default_kind`` dictionary in the configuration file. These default
values are defined in the LFRic infrastructure code.

.. note:: Whilst the ``logical`` Fortran primitive (intrinsic) data
          type is supported in the LFRic API for scalar arguments, it is
          not yet available for fields and operators. This will be added
          as required in future releases.

.. _lfric-precision-map:

Precision Map
+++++++++++++

This gives the amount of storage (in bytes) associated with a
particular LFRic precision. The values for 'r_tran', 'r_solver',
'r_def', and 'r_bl' are set within LFRic infrastructure
according to CPP ifdefs. The values given in the configuration file
are the defaults. 'l_def' is included in the dictionary so that it
contains a complete record of the various precision symbols used in
LFRic.

.. note:: Storing the precision map in the LFRic API within PSyclone is a
          temporary measure which will yield to the LFRic infrastructure
          as the single source of precisions, as discussed in PSyclone
          issue #1941.

.. _lfric-num-any-spaces:

Number of Generalised ``ANY_*_SPACE`` Function Spaces
+++++++++++++++++++++++++++++++++++++++++++++++++++++

As outlined in the :ref:`meta_args <lfric-api-meta-args>` and the
:ref:`Supported Function Spaces <lfric-function-space>` sections
above, the number of generalised ``ANY_SPACE_<n>`` and
``ANY_DISCONTINUOUS_SPACE_<n>`` function spaces can be set in the
:ref:`PSyclone configuration file <configuration>`.

The relevant parameters are ``NUM_ANY_SPACE`` and
``NUM_ANY_DISCONTINUOUS_SPACE``, respectively. Their default values in
the configuration file are 10 and their allowed values are positive
non-zero integers. PSyclone will raise a ``ConfigurationError`` if a
supplied value is invalid.

.. _lfric-api-transformations:

Transformations
---------------

This section describes the LFRic API-specific transformations. In
cases, excepting **LFRicRedundantComputationTrans**,
**LFRicAsyncHaloExchangeTrans**, **LFRicKernelConstTrans** and
**LFRicKokkosTrans**,
these transformations are specialisations of generic transformations
described in the :ref:`transformations` section. The difference
between these transformations and the generic ones is that these
perform LFRic API-specific checks to make sure the transformations
are valid. In practice these transformations perform the required
checks then call the generic ones internally.

The use of the LFRic API-specific transformations is exactly the
same as the equivalent generic ones in all cases excepting
**LFRicLoopFuseTrans**. In this case an additional optional argument
**same_space** can be set when applying the transformation.
The reason for this is to allow loop fusion when one or more of the
iteration spaces is determined by a function space that is unknown by
PSyclone at compile time. This is the case when the ``ANY_SPACE_<n>``
function space is specified in the Kernel metadata. Adding
``{"same_space": True}`` as option when applying the transformation allows
the user to specify that the spaces are the same (see
:ref:`sec_transformations_available` for using options in transformations).
This option should therefore be used with caution. PSyclone will
raise an error if **same_space** is used when at least one of the function
spaces is not ``ANY_SPACE_<n>`` or both spaces are not the same. In general,
PSyclone will not allow loop fusion if it does not know the spaces
are the same. The exception are loops over discontinuous spaces (see
:ref:`lfric-function-space` for list of discontinuous function spaces)
for which loop fusion is allowed (unless the loop bounds become different
due to a prior transformation).

The **LFRicRedundantComputationTrans** and
**LFRicAsyncHaloExchange** transformations are only valid for the
LFRic API. This is because this API is currently the only one
that supports distributed memory.  An example of redundant computation
can be found in ``examples/lfric/eg8`` and an example of asynchronous
halo exchanges can be found in ``examples/lfric/eg11``.

The **LFRicKernelConstTrans** transformation is only valid for the
LFRic API. This is because the properties that it makes constant
are API specific.

The **LFRicKokkosTrans** transformation is also only valid for the LFRic
API. It replaces a single loop -- over cell columns or over degrees of
freedom -- with a call to a C++ region
generated by the Kokkos back-end (``psyclone.psyir.backend.kokkos``), which
the PSy layer reaches through a ``bind(C)`` interface. It is not a
specialisation of a generic transformation: it recognises a kernel shape
rather than a named kernel, and refuses anything it cannot express in the
narrow C ABI that the back-end emits. That ABI is a fixed set of C types,
but not a fixed precision: the width a kind reaches C++ at is the width
LFRic's precision map gives it, so an ``r_solver`` kernel is generated in
single or double precision according to how LFRic was configured. The same
precision map decides which implementation of a kind-polymorphic kernel is
captured: a kernel written as a generic interface over specific procedures
that differ only in precision is resolved to the one the algorithm layer's
arguments select, and the generated region is named after that procedure
rather than after the interface. Choosing between the members asks
PSyclone's own argument matcher to build the interface each member's
metadata implies, and that matcher could not build one for a kernel asking
for an evaluator, which is part of issue #928; rather than compare the
arguments here and cite the issue, the matcher was taught the shape -- one
rank-3 basis per target space, no point count and no weights -- so the
question is asked in the one place that asks it. The same change stops a
basis on an ``ANY_SPACE_<n>``, whose first extent metadata cannot fix
(issue #461), being read as the members matching no precision at all: that
extent is named with a variable, as the PSy layer names it, and is not one
of the things the matcher compares. A field's data may be real or integer:
LFRic's ``integer_field_type`` has ``field_type``'s proxy shape with
``integer(i_def)`` data, and the element type of the View a field crosses
on follows that argument's own declaration, so a ``gh_integer`` field is a
``View<int*>`` where a ``gh_real`` one is a ``View<double*>``. Nothing else
about the field moves with the intrinsic -- the proxy member the PSy layer
reads, the ``ndf`` and ``undf`` formals, the dofmap and the launch belong
to the function space -- so a kernel taking both intrinsics is one region
carrying both widths, each formal at its own. Arithmetic between two
integer field values stays integer, the writer typing its expressions from
their operands, so a quotient of two integer Views truncates as the Fortran
it was generated from does. A field at a kind the ABI has no C type for is
refused naming that kind, as any other formal at such a kind is, rather
than admitted because its intrinsic was. A kernel whose body holds a loop that
PSyclone's dependence analysis accepts -- in practice a loop over a column's
levels whose iterations touch disjoint elements, rather than one sweeping a
recurrence -- is generated over a Kokkos ``TeamPolicy`` with one team on each
cell, and that loop is spread across the team's members as a
``TeamVectorRange``; everything outside it runs on every member, with an
array write made by one under ``Kokkos::single`` and published by a team
barrier. A kernel holding an automatic array -- a column temporary whose
extent is a runtime value -- is generated over a ``TeamPolicy`` too, with
that array placed in team scratch; where it has no loop to spread, the launch
is the flat one and the scratch is private to the rank running the cell,
and where it has one, the team runs the cell and shares the scratch. A kernel
with neither keeps the ``RangePolicy``. A loop over LFRic's ``dof`` or
``owned_dof`` iteration space is launched over dofs rather than over cell
columns: a flat ``RangePolicy`` over the loop's own dof count, with no
dofmap reached and no cell index in the body, each iteration writing one
dof and no two writing the same one. That shape has no team, so a kernel
over dofs holding an automatic array or a loop to spread is refused; an
LFRic builtin is refused too, PSyclone generating its body rather than
reading it from a kernel file. A loop over the halo cells alone begins
where the owned cells end, and the cell it begins at crosses as a second
scalar formal beside the count, filled by the PSy layer from the loop's own
lower bound. The optional ``team_size`` argument
of ``apply`` fixes how many members a team has; without it the policy asks
for ``Kokkos::AUTO``, which is one member on the OpenMP back-end, so a host
build reaches the team-level concurrency only by naming a size. An extent --
of an array argument or of such a
temporary -- is read as a declared bound rather than as a name, so
``dimension(max_length,4)`` and ``dimension(nlayers+1)`` are accepted as
readily as ``dimension(nlayers)``; what is required is an integer expression
over the kernel's own arguments and literals, built with ``+``, ``-``, ``*``
and ``/``. A quotient is carried rather than refused, because Fortran and C++
both truncate an integer quotient toward zero, so
``dimension((stencil_size+1)/2)`` is sized as the Fortran declaring it meant;
a launch whose scratch extent divides also states that rule and stops an
extent that has come out negative, which is an allocation neither language
defines. A name a ``parameter`` beside the array gives a value to is resolved
to that value rather than asked of the launch, so
``integer(kind=i_def), parameter :: nfaces = 4`` sizing
``dimension(nfaces)`` needs nothing passed. An ``allocatable`` local is
sized by an ``ALLOCATE`` in the body, and where that statement's bounds obey the same grammar the region
reads the shape from it, rewrites the declaration and removes both the
``ALLOCATE`` and its ``DEALLOCATE``, after which the array is the scratch
case written the other way round. An allocation the launch cannot evaluate
before it enters the region -- one sized from the kernel's own data, one made
inside a loop, one carrying ``stat``, ``errmsg``, ``source`` or ``mold``, or
a second allocation of the same array -- is refused by name instead. An
assumed-shape *local*, which has no caller to be sized by, states no shape
anywhere the region can read and stays refused, as does a shape with
no C form, such as ``dimension(MAX(nlayers-n,1))``. An assumed-shape
*formal* is measured instead of refused: a boundary-condition kernel writing
``real(kind=r_def), intent(in) :: normals(:,:)`` lets Fortran take the
extents from the actual, and the PSy layer holds that actual, so the region
carries one integer formal per dimension the declaration left out and the
call passes ``SIZE(actual, dim=n)`` for each. The View is sized by that
formal and every shape enquiry the body makes about the array resolves to
it, as it would to a declared extent. The lower bound of such a formal is 1
whatever the actual was declared from -- Fortran's own rule for an
assumed-shape dummy -- so only the size comes from the caller; one that
states a bound and leaves the other, ``dimension(0:)``, would take its
origin from the kernel and its extent from the caller and is refused by
name. The declared *lower*
bound is read the same way and need not be 1: ``dimension(0:nlayers-1)`` is generated
as a View of ``nlayers`` elements with ``0`` subtracted from every subscript
of that array, so an origin the Fortran chose is carried through rather than
refused. A body that asks an array for
its shape -- ``LBOUND``, ``UBOUND`` or ``SIZE`` -- is answered from that same
declaration rather than at run time, so the region carries the declared bound
instead of querying a View. Most such calls are written by PSyclone rather
than by the kernel author: lowering a whole-array assignment to an explicit
loop puts a pair of them into its bounds. That lowering is what admits an
assignment over a whole array or a section of one -- ``a(i:j) = ...``, which
the finite-volume kernels use to write a column as a unit, and
``vector = 0.0_r_def`` over an array declared with a shape -- since C++ has
no whole-array assignment and the region has to say the loop instead. What
the lowering declines is refused here with its reason quoted: an assignment
reading the array it writes, for which no order of loops means what the
Fortran meant. An array-valued intrinsic on the right-hand side is generated
rather than refused -- ``MATMUL``, ``DOT_PRODUCT``, ``SUM``, ``MINVAL``,
``MAXVAL`` and ``TRANSPOSE`` become loop nests over the destination, with no
array temporary between them, because one element of a contraction is a
scalar reduction and one element of a transpose is an index swap. What is
refused of these is a fold the shape cannot express: a ``dim`` that is not a
literal or names no dimension the operand has, a ``mask``, a reduction
directly inside another, or an operand that is neither a whole array nor a
section of one. ``RESHAPE`` is generated only from a rank-1 source with a
literal shape. An intrinsic no writer in the chain can spell is refused when
the transformation is asked rather than when it generates, because
``validate`` puts each of the body's intrinsics to the writer itself.
A section that is not in an assignment is judged by where it is
instead: as an actual argument it is left to the call, since inlining the
callee takes the argument away with it; in the bounds of an ``ALLOCATE`` it
is read as the shape that statement states; and anywhere else it is refused.
A body that calls a subroutine or a function is inlined rather than
refused. The region is a C++ function with no Fortran to call into, so
``InlineTrans`` makes the callee's statements the kernel's own before any
other rule looks at the body, and a callee that itself calls is inlined in
its turn, one call at a time until none is left; a callee brings its own
loops, locals and sections with it and each is then judged like the
kernel's own. The repetition is bounded at eight calls into one body, and
the bound is load-bearing rather than defensive: ``InlineTrans`` has no
recursion check, so a routine that calls itself would be substituted into
itself for as long as it was asked, and reaching the bound is instead a
refusal naming the routine still to be inlined. In scope are a procedure of
the kernel's own module and a procedure of a module the kernel ``use``\ s
whose source PSyclone can read, the second brought into the kernel's
container by ``KernelModuleInlineTrans`` first -- and what that container
gains it keeps, so a callee reading a named constant from a third module
brings that constant with it. A function used this way is in scope whether
or not the frontend could tell it from an array: ``selector(face)`` in an
expression leaves the name an unspecialised symbol, and a ``Call``'s callee
is treated as the routine it is. A callee whose module is not on the search
path is out of scope, PSyclone having a name for it and no body. A procedure the
callee itself calls is in scope on the same terms, and a call it makes to a
sibling of its own module is settled before it travels: the sibling's
statements are put in the call's place in the module the two share, so
``crosses_panel_edge``'s call to ``rotated_panel_neighbour`` reaches the
kernel as the arithmetic it stood for rather than as a name the kernel's
container has nothing to bind. A sibling reached through a *generic
interface* of that module is settled the same way and by the same rule as a
kernel's own generic call, which is the arguments rather than the order the
interface lists its specifics in:
``pointwise_coordinate_jacobian_r_single``'s call to ``jacobian_abr2XYZ``
becomes a call to the specific its arguments select, and only that specific
travels. A sibling that cannot be substituted -- one declaring a static
local, say, or one reached through an interface whose specifics differ in a
kind PSyclone cannot reduce to a value, so that the arguments settle nothing
-- leaves its call where the file put it, and the refusal that follows names
it. Data of the callee's module is not in
scope and is not brought into it by this: ``chi2xyz``, which reads a
rotation matrix its module keeps at run time, is still refused for that
datum, a named constant being the only thing that crosses a container
boundary. Being in
scope is not being inlinable, and the rest of the judgement is PSyclone's
rather than this transformation's: a callee reading data private to its own
module, one whose declarations depend on an argument the call site writes to
before calling, and one whose actual and formal types do not agree are each
refused in ``InlineTrans``'s own words with the call named, followed by
``KernelModuleInlineTrans``'s own where the callee could not be brought in
either. A formal declared ``TARGET`` is none of those: the attribute
constrains what a pointer elsewhere may be aimed at, which substituting a
body neither creates nor breaks, so such a formal is given the type the
frontend did parse from its declaration and the callee is then inlined like
any other. A formal carrying any other attribute PSyclone does not model --
``POINTER``, ``ALLOCATABLE``, ``VALUE`` -- is refused as before, in
``InlineTrans``'s words. A callee's own ``POINTER`` local is relaxed too
where it only ever aims at a whole array, and becomes a ``View`` handle in
the generated region; every other use of such a pointer is refused by name. A name the two
scopes hold differently -- an import in one and, in the other, a name that
scope cannot say the origin of, which is how LFRic's FFSL kernels and their
support routines reach ``reference_element_mod``'s ``S`` -- is given the
module both name before the two tables are merged, and refused as it was
where that module cannot be read. An imported name a later pass needs the
type of is read from its module too, so an array section bounded by a
module's parameter is lowered by that parameter's value. Types agreeing is a scored judgement rather than
an identity: a literal actual stating no kind, an actual whose type PSyclone cannot
resolve, and an array section against a formal argument PSyclone holds only
a partial type for are each a weaker match than an exact one rather than no
match, and a call to a generic interface takes the strongest match among its
candidates. Two candidates matching a relaxed call equally well is not
settled by guessing: that call is refused as ambiguous, naming both. Not
every call is a call, either: an indexed name whose
meaning the kernel's own file does not settle -- ``blending_weights(index)``,
where the array comes from a ``use`` -- is read by the frontend as a call,
and resolving it reaches a datum rather than a routine. PSyclone reports
that by failing to specialise the symbol rather than by refusing the
transformation, and it is refused here with the rest, so that a kernel which
cannot be captured is declined rather than raising out of ``validate``.
A body may also fill an array from a
constructor -- ``v_dot_n = (/ -1.0, 1.0, 1.0, -1.0 /)``, or one full-extent
dimension of an array as ``vert_vec(:,qp1,qp2) = (/ ... /)`` -- which is
generated as one assignment per element, into the array the kernel has
already declared and from the origin its declaration gives. A constructor
used as a value rather than as a whole right-hand side -- an actual argument,
an operand, or one nested inside another -- is refused by naming the
position, because C has no array-valued expression and the region creates no
temporary to hold one. A ``DO WHILE`` loop in the body is generated as a C
``while``, and is never spread across the team. An unlabelled ``EXIT`` is
generated as a C ``break``, which leaves the same loop the Fortran leaves;
the loop it leaves is never spread across the team either, since a lambda
cannot break the loop it was launched over, and an ``EXIT`` that names the
construct it leaves is refused with the rest of the Fortran the PSyIR does
not model. A name the body reads that is not one of
its arguments reaches the region one of four ways. A ``parameter`` declared
in the kernel's own module is written in as its value, since a kernel module
is ``private`` by default and the PSy layer could not import the name; the
value need not be a literal, one written as an arithmetic over other
parameters being folded to what they state. An array ``parameter`` is indexed
by something the loop computes rather than used whole, so there is no single
value to write in: its elements are declared instead inside the generated
launch body, as a ``const`` array beside the body's other locals rather than
at file scope, because a namespace-scope array is host data and a device
compiler will not read one. A constant imported from elsewhere is passed by
value, which needs the
source of its module on PSyclone's search path so that its kind can be read
rather than guessed. A *variable* of the kernel's own module -- the profile
``real(kind=r_def), public :: profile_heights(100)`` that
``profile_interp_kernel_mod`` interpolates from is the shape -- is state
rather than a value, so the region takes it as a formal of its own and the
PSy layer ``use``\ s the kernel module to pass it, a scalar by value and an
array as a read-only View of the extents its declaration states; what the
module holds where the launch is made is what the region sees, so a body that
assigns to one is refused rather than losing the assignment, as is one the
module keeps ``private``, one sized by a name the region has no argument for,
one with no declared extents at all, and one declared inside the routine
rather than the module, which Fortran gives the ``SAVE`` attribute and
nothing outside the routine names. Last, a name appearing only as an
intrinsic's ``kind`` argument, as the ``r_def`` of ``real(x, r_def)``, is
none of these, being a type rather than data -- it is the width the cast is
generated at, so it joins the precisions the region records rather than its
arguments. A ``logical`` scalar -- an argument or an
imported constant -- is the one thing on that ABI whose width is never
consulted: it is declared ``logical(c_bool), value`` on the interface and the
call site wraps the actual in ``LOGICAL(..., c_bool)``, so the compiler
converts it whatever kind LFRic gave it, and no width assertion is generated
for it because there is none to make. A ``logical`` *array* is still refused,
because an array crosses by reference and would be reinterpreted rather than
converted. A formal or constant declared with no kind at all -- ``logical,
intent(in) :: include_surface``, or the ``integer`` housekeeping arguments
LFRic's argument ordering supplies -- is accepted where it is a logical or an
integer, and refused where it is a real. The logical needs no width, by the
conversion just described. The default integer does cross at a width, and
that width is stated in no kind parameter the precision map could be asked
about, so it is measured instead of assumed: the generated interface carries
``storage_size(1) == storage_size(1_c_int)`` as a compile-time assertion, and
a build whose default integer is not ``c_int`` fails to compile. A real is
refused because LFRic names a kind on every real it means -- ``r_def``,
``r_solver``, ``r_single`` and ``r_tran`` are all in use and all different --
so a real that names none has said nothing about its width rather than having
chosen a default. A declaration stating a width in place of a kind name,
``integer*8`` or ``real(kind=8)``, is refused for every intrinsic: it did say
which width it wanted, and reading it as the default would narrow it in
silence. A kernel that reads a
field through a stencil is accepted when the stencil is ``cross``, ``cross2d``
or ``region``, and refused by name otherwise. A 2-D stencil gives the kernel a
sliced dofmap and a sliced size array, both of which become Views with the cell
index appended exactly as the ordinary dofmap does. A 1-D or region stencil
gives the size as a per-cell scalar instead, fed from
``field_stencil_size(cell)``; because a region runs every cell at once, that
scalar crosses the ABI as the whole rank-1 array and the generated body
subscripts it by the launch's own cell, which makes it the same per-cell View
the 2-D shape's size array already is. The dofmap beside it is declared over
that per-cell size, while LFRic allocates it uniformly, and a View's extents
fix its strides; the region therefore also carries the dofmap's storage extent
as a scalar, which the PSy layer supplies as
``SIZE(field_stencil_dofmap, dim=2)``, and sizes both that View and any
automatic array declared over the stencil from it. ``xory1d`` is refused: it
carries a direction argument on top of the 1-D size, chosen per cell in the
algorithm layer, which the ABI does not describe. A stencil also makes the PSy layer emit a halo exchange in front of the loop; that
exchange is lowered before the loop is replaced, because it computes its own
depth by walking forward to the accesses that read the field and the
replacement would have removed them first. A kernel taking an LMA operator --
a ``gh_operator`` argument -- is accepted, and needs nothing added to the ABI:
such an operator reaches the kernel as an integer ``ncell_3d`` and a rank-3
array over ``(ncell_3d, ndf1, ndf2)``, with every one of those extents a
formal of its own, so the ordinary View description covers it whole. A
*columnwise* operator -- ``gh_columnwise_operator``, the CMA form -- is
refused: it is a banded matrix carrying its own bandwidth and indexing
arguments, none of which the region has an argument kind for. Taking an
operator also gives the kernel a leading ``cell`` argument, which the PSy
layer fills with the cell-column loop's own counter; that counter is the
region's launch index, so the cell position is *declared* in the launch body
from the index the launch already has rather than passed across the interface.
The generated interface is therefore one argument shorter than the kernel's
own signature, while the arithmetic the kernel writes over it --
``ik = (cell - 1) * nlayers + 1``, by which a kernel finds its column's slice
of the operator -- is generated unchanged. If the PSy layer ever supplied
something other than the loop variable there, the transformation refuses
rather than binding the wrong index. A kernel that asks for a basis is
accepted when every shape it names is ``gh_quadrature_XYoZ`` or
``gh_evaluator``, and refused by name otherwise. Neither adds anything to the
ABI: XYoZ quadrature gives the kernel two point counts, two weight arrays and
one basis array per function space that asked for one, over
``(dim, ndf, np_xy, np_z)``, while an evaluator gives it no rule at all,
tabulating the basis at the nodal points of a target space to give
``(dim, ndf, ndf of the target)`` and no weights. Every one of those is an
argument the PSy layer has computed before the loop and every extent of it is
a formal of the same kernel, so the ordinary scalar and View descriptions
cover them whole; a kernel may name both shapes, and then carries a basis
array per shape on each of its spaces. Face and edge quadrature are refused by
name, because they replace the XYoZ pair with a face or edge count and a
single point count and nothing here has been measured against them. An
inter-grid kernel -- one whose fields carry ``mesh_arg`` and so live on a
coarse mesh and a fine one -- is accepted, and it too adds nothing to the
ABI. LFRic runs such a kernel once per *coarse* cell and the body reaches
the fine cells itself, through the cell map ``cell_map(:,:,cell)`` naming
the ``ncell_f_per_c_x`` by ``ncell_f_per_c_y`` fine cells that coarse cell
refines into, so the launch is the flat one over cells that a single-mesh
kernel already has. The cell map is an actual the PSy layer slices by cell,
which makes it a rank-3 View exactly as a sliced dofmap is; the three counts
beside it are scalars; and a field on the fine mesh brings its dofmap whole,
``map_f(:,:)``, which is what the PSy layer passes and what the region
therefore declares. The second mesh reaches the region as data and never as
an iteration space: ``ncell_f`` is the extent the fine dofmap is strided by,
not a bound anything is launched over. A loop that iterates into the
halo is accepted, and the halo exchange stays where the PSy layer put it.
The launch's upper bound crosses the interface as a scalar formal filled with
the loop's own stop expression, so what that bound means is settled in the PSy
layer and only its value reaches the region: a literal depth, whose bound the
PSy layer writes as ``mesh%get_last_halo_cell(1)``; a dof loop over the
annexed dofs, written
``field_proxy%vspace%get_last_dof_annexed()``; and a depth computed at run
time, ``mesh%get_last_halo_cell(depth)``, whose depth expression is evaluated
where it already was and never crosses. Every per-cell View is sliced to that
same formal, so a region running into the halo describes the cells it runs
over rather than the owned ones. What such bounds have in common is that each
counts consecutively from the first cell or dof; a loop whose *lower* bound is
shifted into the halo -- ``halo_cell_column``, which runs the halo alone --
is still refused, there being no lower-bound formal to fill. Nothing is
exchanged inside a region: the exchange the PSy layer emits in front of the
loop is lowered in front of the call to the region.
A field written at a dof two cells share -- a ``gh_inc`` or ``gh_readinc``
argument, where the two cells each contribute to it, or a ``gh_write`` to a
continuous function space, where they each replace it -- has two answers
here, and both are generated. The default is an atomic: a contribution is
emitted as ``Kokkos::atomic_add``, or ``_sub``, ``_mul`` or ``_div`` by the
operator it carries, and a replacement as ``Kokkos::atomic_store``, on the
View element. It is decided per argument rather than per region, so a
``gh_write`` to a *discontinuous* space in the same kernel stays a plain
assignment; and it is the update and not the read that is made atomic, so a
``gh_readinc`` reads its element plainly. Each component of a field vector
is a field of its own and is made safe separately. The question is asked of
cell iteration alone: a dof-iterating kernel writes each dof once, so its
stores stay plain assignments whatever their function space, and the
formals of such a kernel -- which are its fields and scalars, with no
``nlayers``, ``ndf``, ``undf`` or dofmap among them -- are never compared
against a cell kernel's argument order. A caller who knows the
loop's stores reach no shared dof may say so through the
``disjoint_writes`` option, which is an assertion about the kernel and is
refused where it contradicts the loop's colouring, the ``atomics`` option
or the metadata. The other answer is colouring, which is what LFRic's own
OpenMP path takes: a loop
:py:class:`~psyclone.domain.lfric.transformations.LFRicColourTrans` has
already rewritten is accepted, the inner ``cells_in_colour`` loop being the
one captured while the outer loop over colours stays in the PSy layer and
enters the region once per colour. No atomic is generated there, the cells
of one colour meeting at no dof. Such a region carries three arguments an
uncoloured one does not -- LFRic's colour map, the colour being launched and
the number of colours -- and declares its cell from them rather than from
its launch index; the number of colours is not redundant beside the map,
because the map crosses the ABI as bare storage and is rebuilt inside the
region as a rank-2 View, where that number is the ``LayoutLeft`` stride.
Which answer is taken follows the loop the transformation is given, unless
the ``atomics`` option of ``apply`` says otherwise. Asking for both
answers at once is refused -- ``True`` on a coloured loop guards data no
other cell of the launch reaches -- and so is asking for neither, ``False``
on an uncoloured loop whose kernel writes a shared field being the one
combination that would generate a race. The two are alternatives
and not a ranking. Both are correct, and which is faster is a measurement on
a GPU that has not been taken here, so the default is the one that needs
nothing of the algorithm layer. They do differ in one property that is not a
measurement: floating-point addition is not associative, so the order the
contributions to a shared dof arrive in is part of the answer. An atomic
launch adds them in the order its threads reach the dof, which is cell order
on one thread and an order that varies between runs on more; a coloured
launch adds them in colour order, which is fixed whatever the concurrency.
So an atomic region reproduces a serial Fortran run bit for bit on one
thread and not on several, and a coloured region reproduces itself on any
thread count and a serial Fortran run on none.
An access that is neither of those and
neither safe nor read-only -- a reduction, of which GungHo has none in a
coded kernel today -- is refused by naming it.
A kernel symbol whose name Fortran allows and C++ reserves is refused,
whether it is a formal or a local. Fortran reserves no words, so ``const``,
``new`` and ``operator`` are ordinary variable names, and one written out as
a C++ identifier gives a region no compiler will take. The
contract it accepts, and the reasons it refuses, are given in its
documentation below.

The LFRic API-specific transformations currently available
are given below. Early transformations include "Dynamo0p3" or "Dynamo"
in their name to indicate that these transformations are only valid
for this particular API. More recent transformations typically include
"LFRic" in their name to indicate the same restriction. However, more
importantly, transformations that are specific to LFRic reside in the
LFRic-specific "psyclone.domain/lfric/transformations"
directory. Note, the early LFRic API-specific
transformations have not yet been migrated to this directory.

.. note:: Only the loop-colouring, OpenMP and **LFRicKokkosTrans**
          transformations are currently supported for loops that contain
          inter-grid kernels. Attempting to apply other transformation types
          will result in PSyclone raising an error.

.. autoclass:: psyclone.domain.lfric.transformations.LFRicExtractTrans
    :members:
    :noindex:

.. autoclass:: psyclone.domain.lfric.transformations.LFRicKokkosTrans
    :members:
    :noindex:

.. _lfric-kokkos-capture-contract:

.. rubric:: The LFRicKokkosTrans capture contract

**Precision is carried, not chosen.** A kind is placed on the ABI by the
width LFRic's precision map gives it, so an ``r_solver`` kernel reaches
C++ as ``float`` in a single-precision build and as ``double`` in a
double-precision one, from the same source and with no option to set. The
region computes at that width throughout: its locals are declared at their
own kind and its literals are suffixed, because neither crosses an
interface and both would otherwise be generated at the C writer's default
of ``double``, silently promoting the kernel. The generated ``bind(C)``
interface then asserts, at compile time, that each kind really has the
width the C++ was generated for, so a rebuild at another precision is a
compile error naming the kind rather than a wrong answer.

**A kind-polymorphic kernel is resolved, not refused.** LFRic writes such
a kernel as a generic interface over specific procedures differing only in
the precision of their real arguments, and PSyclone presents one schedule
per procedure. The one captured is the one the algorithm layer's
precisions select, which is
:py:meth:`~psyclone.domain.lfric.LFRicKern.validate_kernel_code_args`'s
question rather than this transformation's -- it exists to identify the
right subroutine of a mixed-precision kernel, and matches in byte widths
through the same precision map. The region takes the name of the selected
procedure rather than of the interface, so two invokes of one kernel at
different precisions generate two regions instead of colliding on one.

Matching in widths is what makes the ABI right and the choice sometimes
impossible. ``r_single`` and ``r_solver`` are both 4 bytes, so an
interface offering both is refused rather than resolved by coincidence,
as is one no algorithm precision selects at all. Metadata the matcher
cannot model -- a stencil or a CMA kernel, which PSyclone's issue #928
leaves unbuilt -- is a third refusal, kept distinct from
finding no match because a question that cannot be asked has not been
answered "no".

An evaluator shape was one of those and is no longer. Rather than reach
past the matcher and compare the actual arguments here, citing #928 as the
reason, the matcher itself was taught what an evaluator implies: one
rank-3 basis per target space, extents of ``(dim, ndf, ndf of the
target)``, with no point count and no weights because an evaluator carries
no quadrature rule. That is a smaller answer than it sounds -- an
evaluator asks for no callback the class does not already have -- and it
keeps the question in the one place that asks it, so a kernel is selected
between on its precisions whatever shape its basis has.

It also removes a refusal that was a misreading rather than a mismatch.
Metadata cannot fix a basis's first extent for an ``any_space`` or an
``any_discontinuous_space``, which is PSyclone's issue #461, and a matcher
that could not produce that extent reported the members as matching no
algorithm precision at all -- an answer about precisions it had never
compared. The extent is instead named with a variable, exactly as the PSy
layer names it, and it is not one of the things the matcher compares.

An inter-grid shape was one of those too, and was taught the same way:
the cell map with its two per-cell counts and the fine mesh's cell count,
and a whole dofmap for a field on the fine mesh where a per-cell one
serves a field on the coarse. Both are what the PSy layer already passes,
so what the matcher gained is a description rather than a decision.

**A field's data may be real or integer.** LFRic's ``integer_field_type``
has ``field_type``'s proxy shape with ``integer(i_def)`` data, and the
region's element type follows the argument's own declaration rather than
being fixed when the field ABI is written, so a ``gh_integer`` field
crosses as a ``View<int*>`` where a ``gh_real`` one crosses as a
``View<double*>``. Nothing else about the field moves with the intrinsic:
the proxy member the PSy layer reads, the ``ndf`` and ``undf`` formals,
the dofmap and the launch belong to the function space. A kernel taking
both is one region carrying both widths, each formal at its own, and the
arithmetic between two integer field values stays integer -- the writer
types its expressions from their operands, so a quotient of two integer
Views truncates as the Fortran it was generated from does.

A kind the ABI does not name is refused rather than guessed at -- a
16-byte ``r_quad``, an undeclared precision, a module constant whose width
the precision map does not carry. An integer field is not exempt: its
declaration is read like every other formal's, so one at a width the ABI
has no C type for is refused naming the kind rather than written as
``int`` because its intrinsic was admitted.

**A logical scalar is not one of them, because it crosses by conversion
rather than by width.** The dummy is ``logical(c_bool), value`` and the
call site wraps the actual in ``LOGICAL(..., c_bool)``, which is a
conversion the compiler performs; the two kinds therefore need not agree,
and no width assertion is generated for a logical because there is no
width to assert. This matters beyond tidiness: LFRic's ``l_def`` is
``kind(.false.)`` and measures 4 bytes where PSyclone's precision map
records 1, which is issue #1941. Nothing here reads that entry, so the
prototype is correct at either value and a corrected #1941 would not
change a line of what it generates.

A logical **array** stays refused. Conversion is per value, and an array
crosses by reference: a ``View<bool*>`` laid over ``logical(l_def)``
storage would reinterpret 4-byte elements as 1-byte ones rather than
convert them, which is the very failure conversion removes for a scalar.

**A formal declared with no kind at all is accepted if it is an integer
or a logical, and refused if it is a real.** GungHo writes plenty of
both: ``logical, intent(in) :: include_surface``, and the whole of the
integer housekeeping ``nlayers, ndf_w0, undf_w0, map_w0`` that LFRic's
argument ordering supplies. The two are admitted for different reasons.
A logical needs no width at all, by the conversion above, so an unnamed
logical kind is the ``l_def`` case with the name taken off -- ``l_def``
*is* ``kind(.false.)``, so the two declarations say the same thing.

An integer does cross at a width, and the default integer's width is
named by no kind parameter the precision map could be asked about. It is
therefore measured rather than assumed: the generated interface carries
``storage_size(1) == storage_size(1_c_int)`` as a compile-time assertion
like any other kind's, so a build whose default integer is not ``c_int``
fails to compile rather than losing half of every value. The assertion is
labelled ``assert_kind_default_integer``, and that name is this
transformation's own -- ``constants_mod`` declares no such kind, being
unnamed being the whole of what makes it the default -- so it is written
into no ``use`` statement.

A **real** declared with no kind stays refused, and the asymmetry is
deliberate. LFRic names a kind on every real it means: ``r_def``,
``r_solver``, ``r_single`` and ``r_tran`` are all in use and all
different, so the kind is the whole of what a real declaration says about
its width, and one that says nothing is more likely an oversight than a
default. A **width stated in place of a kind name** -- ``integer*8``, or
``real(kind=8)`` -- is refused too, for all three intrinsics: that
declaration did say which width it wanted, and reading it as the default
would be the silent narrowing this whole contract exists to prevent.

**An LMA operator is accepted; a CMA operator is not.** An LMA
operator reaches the kernel as ``ncell_3d`` and a rank-3 array
``dimension(ncell_3d, ndf1, ndf2)`` holding every cell's local stencil
end to end. Every extent is a formal of its own, so the operator becomes
an ordinary read-only View and needs no argument machinery: what it needs
is the cell. A CMA operator stays refused, as before, because a banded
matrix carries its own bandwidth and indexing arguments that this region
has no way to describe.

**An inter-grid kernel spans two meshes and still launches over one.**
Such a kernel takes a field on a coarse mesh and one on a fine mesh, and
LFRic runs it once per *coarse* cell: the cell map ``cell_map(:,:,cell)``
names the ``ncell_f_per_c_x`` by ``ncell_f_per_c_y`` fine cells that cell
refines into, and the body reaches them itself. So the launch is the flat
one it would have been over a single mesh, and the second mesh reaches
the region as data and never as an iteration space -- ``ncell_f`` is the
extent the fine mesh's dofmap is strided by, not a bound anything is
launched over.

Nothing had to be built for that. The cell map is an actual the PSy layer
slices by cell, so the per-cell rule turns it into a rank-3 View like any
other sliced dofmap; the three counts beside it are scalars; the fine
mesh's dofmap arrives whole, ``map_f(:,:)``, which the same rule leaves
whole because it is not sliced; and the coarse mesh is what LFRic's own
loop bound already names. What stood in the way was a refusal, and
removing it is what accepts these kernels.

**A region may iterate into the halo, and the exchange stays in the PSy
layer.** The launch's upper bound is a formal the region reads and never
writes, filled by the PSy layer with the loop's own stop expression, so
what the bound *means* is settled outside the region and only its value
crosses. LFRic writes three such expressions and all three are accepted:
``mesh%get_last_halo_cell(1)`` for a loop taken to the first halo depth,
``field_proxy%vspace%get_last_dof_annexed()`` for a dof loop taken over
the annexed dofs, and ``mesh%get_last_halo_cell(depth)`` for a depth
computed at runtime -- that depth is evaluated where it already was, and
the region never sees it.

Every per-cell View is sliced to the same formal, so a region running into
the halo describes the cells it runs over rather than the owned ones. What
the accepted bounds have in common is that each counts consecutively from
the first cell or dof, which is :py:attr:`_COUNTED_BOUNDS`.

**A loop over the halo alone begins where the owned cells end**, and the
cell it begins at crosses as a second scalar formal beside the count. The
two are filled the same way and from the same place: the PSy layer passes
the loop's own start expression, converted once from LFRic's 1-based
counting to the region's 0-based indexing, so what the bound means is
again settled outside the region and only its value crosses. The two
lower bounds LFRic writes for a loop this transformation accepts are
therefore both accepted -- ``start``, which takes no formal at all, and
``cell_halo_start``, which takes one -- and they are
:py:attr:`_LOWER_BOUNDS`. The bounds redundant computation produces are
refused by name, each being stated relative to a depth index the region
carries nothing for.

A region naming no first cell generates exactly the text it generated
before the formal existed, in all four launch shapes; see
:py:func:`~psyclone.psyir.backend.kokkos_launch.launch_offsets`.

The halo exchange itself does not move. The PSy layer emits it in front of
the loop and it is lowered there, in front of the call to the region; see
:py:meth:`_lower_halo_exchanges`. Nothing is exchanged inside a region.

**A loop over dofs is launched over dofs.** LFRic's ``dof`` and
``owned_dof`` iteration spaces run a kernel once per degree of freedom
rather than once per cell column, handing it a single dof of each field
it takes, and the launch follows: a flat
``Kokkos::RangePolicy`` over the loop's own dof count, with no dofmap
reached and no cell index in the body. The shape is
:py:func:`~psyclone.psyir.backend.kokkos_launch_dof.dof_launch` and the
count is the loop's, ``ndofs`` or ``nannexed`` -- not ``undf``, which
would run the halo dofs the Fortran loop was told to leave.

Almost nothing had to be built for it. A per-dof formal is an actual the
PSy layer subscripts by the loop counter, so the same rule that makes a
per-cell formal a rank-1 View sliced to the count and subscripted by the
launch index makes a per-dof one, and nothing new crosses the ABI. One
iteration writes one dof and no two iterations write the same one, so
this shape needs neither colouring nor an atomic where a launch over cell
columns would need one or the other.

What it does not have is a team. A dof launch is a flat range with
nowhere to place scratch and no members to spread a loop over, so a
kernel over dofs carrying an automatic array, or a loop the dependency
analysis would spread, is refused by name rather than launched over a
shape that would silently drop it. That refusal is asked after the rules
a cell-column launch would apply, so a body a cell launch could not take
either is still refused for the reason it always was.

**An LFRic builtin is refused by name.** ``setval_c`` and its kind are
dof loops PSyclone generates the body of rather than reads from a kernel
file, so there is no source to capture and none of the metadata this
transformation reads from a kernel. Refusing them by name is what keeps
the dof space open for the kernels that do have a file.

**The kernel's cell argument is declared rather than passed.** LFRic gives
a leading ``cell`` formal to exactly the kernels that take an operator,
because a kernel finds its own slice of a local stencil arithmetically --
``ik = (cell - 1) * nlayers + 1`` -- rather than being handed the slice.
The launch already knows which cell it is on, so the region declares
``const int cell = cell_1 + 1;`` from its own index and the formal never
reaches the generated signature. Passing it instead is what makes this
worth stating: it would become a launch parameter fixed for the whole
region, and the actual the PSy layer supplies is its own loop counter,
which lowering removes. Every cell would then compute ``ik`` from an
unassigned variable -- code that compiles, links, runs and is wrong.

**A stencil is accepted by shape**, and the accepted shapes are
:py:attr:`_SUPPORTED_STENCILS` -- ``cross``, ``cross2d`` and ``region``.
A 2-D stencil needs no argument machinery of its own: LFRic hands the
kernel a sliced dofmap and a sliced size array, both of them array
formals, so both become Views with the cell index appended exactly as the
dofmap ``map_w3(:,cell)`` already does.

A 1-D or region stencil hands the size over as a *scalar* formal fed from
``field_stencil_size(cell)``. That is one cell's value and a region runs
every cell at once, so the size crosses the ABI as the whole rank-1 array
and the body subscripts it by the launch's own cell -- a scalar formal
made per-cell, which is what the 2-D shape's size array already is. The
dofmap beside it is declared ``dimension(ndf, stencil_size)``, over that
per-cell value, while LFRic allocates it uniformly. A View's extents fix
its strides, so a per-cell value cannot size one: the region carries the
dofmap's storage extent beside it as a scalar of its own, measured by the
PSy layer as ``SIZE(field_stencil_dofmap, dim=2)``, and sizes the View and
any automatic array declared over the stencil from that instead.

``xory1d`` is refused by name. It carries a direction argument on top of
the 1-D size, chosen per cell in the algorithm layer, and nothing here
describes it.

A stencil also makes the PSy layer emit a halo exchange in front of the
loop, which is lowered before the loop is replaced rather than after; see
:py:meth:`_lower_halo_exchanges`. The matcher refusal above is unaffected:
PSyclone's issue #928 leaves every stencil shape unbuilt, so a stencil
kernel written as a generic interface is still refused there whatever its
shape.

**A basis is accepted by shape** too, and the accepted shapes are
:py:attr:`_SUPPORTED_SHAPES` -- ``gh_quadrature_XYoZ``,
``gh_quadrature_face`` and ``gh_evaluator``. None of them needs argument
machinery of its own. XYoZ quadrature adds two point counts, two weight
arrays and one basis array per function space that asked for one, shaped
``(dim, ndf, np_xy, np_z)``; face quadrature adds a face count, one point
count, a single *rank-2* weight array shaped ``(np_xyz, nfaces)`` and a
basis shaped ``(dim, ndf, np_xyz, nfaces)``; an evaluator adds no rule at
all, tabulating the basis at the nodal points of a target function space
to give ``(dim, ndf, ndf of the target)`` and no weights. Every one of
those is an argument the PSy layer has computed before the loop and every
extent of it is a formal of the same kernel, so the existing scalar and
View descriptions cover them whole. A kernel may name more than one shape,
in which case each space it declares carries a basis array per shape; each
shape is checked on its own so that a refusal names the one that is not
modelled rather than the whole set.

The face rule's weights are the one place the three shapes differ in more
than their extents. XYoZ hands over two rank-1 arrays and a face rule one
rank-2 array, whose leading extent is the point count and therefore the
stride of the generated ``LayoutLeft`` View. A View built with the two
extents the other way round reads a transposed table with every subscript
still in range, so the claim that the shape is covered is checked against
a compiled region and compared by value rather than asserted over
generated text.

Edge quadrature is refused by name. It carries an edge count where a face
rule carries a face count and is otherwise the same shape of argument, but
the released model has no kernel asking for it and nothing here has been
measured against the model for it. Refusing by name says that; accepting
on the strength of the resemblance would not.

**A called subroutine is inlined, not called.** The generated region is
a C++ function and there is no Fortran for it to call into, so a kernel
that calls a helper has that helper's statements made its own before
anything else looks at the body. The rewrite is
:py:class:`~psyclone.psyir.transformations.InlineTrans`, applied one call
at a time and repeated until the body makes none: a helper that calls a
second helper leaves that second call behind in the statements it
contributes, so one pass would not be enough. It runs before every other
rewrite here, because a callee brings its own loops, its own locals and
its own array sections in with it and each of those is then judged like
the kernel's own.

The repetition is bounded by ``LFRicKokkosInlineMixin._INLINE_LIMIT``,
eight calls into one kernel body, and the bound is load-bearing rather
than defensive: ``InlineTrans`` has no recursion check, so a routine that
calls itself is substituted into itself for as long as it is asked.
Reaching the bound is a refusal naming the routine still to be inlined.

**A callee out of scope is refused rather than guessed at.** In scope are
a procedure of the kernel's own module, and a procedure of a module the
kernel names in a ``use`` whose source PSyclone can read -- the latter
is first brought into the kernel's Container by
:py:class:`~psyclone.domain.common.transformations.\
KernelModuleInlineTrans`.
A callee whose module is not on the search path is not: PSyclone has a
name for it and nothing else, and there is no body to inline.

Bringing the callee in moves what it needs with it, so the module it came
from need not be a leaf: a function of one module reading a named constant
from a second arrives with that constant declared alongside it, and the
constant then reaches the region the way every module constant does, as a
by-value formal the PSy layer supplies. This is what puts LFRic's
``face_from_face_selector`` -- a pure function of
``sci_face_selector_support_mod`` reading the face indices ``W``, ``S``,
``E`` and ``N`` from ``reference_element_mod`` -- inside the capture.

It is in scope whether or not the frontend could tell it from an array.
``selector(face)`` standing in an expression is a function reference or an
element of an array, and where the kernel's own file does not settle which
the name is left an unspecialised symbol; ``KernelModuleInlineTrans`` reads
such a symbol at the call site as a datum of the callee's name and declines
to shadow it. A ``Call``'s callee is a routine whether or not the frontend
could say so, so the symbol is made one before the callee is brought in.
Only a bare symbol is: a name PSyclone has already typed as data is left
alone, and asking for its body reaches the datum and is refused below.

Being in scope is not being inlinable, and the rest of the judgement is
PSyclone's rather than this transformation's: a callee reading data
private to its own module, one whose declarations depend on an argument
the call site writes to before calling, one whose actual and formal types
do not agree, one holding a CodeBlock. Each is refused in ``InlineTrans``'
own words with the call named, because those words say what to fix and a
paraphrase would say less. Where the callee could not be brought into the
Container either, that refusal is carried too, after
``bringing it into the container was refused first:``. ``InlineTrans``
alone would say only that the body is in another Container, which is the
symptom; the second text names the reason -- a datum of the callee's own
module that could not travel with it, say -- and so says which of the two
is worth fixing. So is a name that turns out not to be a call
at all: an indexed reference the kernel's own file does not settle the
meaning of is read as one by the frontend, and resolving it can reach a
datum and raise :py:exc:`TypeError` rather than refuse. That too is a
refusal here, so that :py:meth:`validate` declines a kernel it cannot
capture instead of raising out of PSyclone.

**A TARGET formal is inlinable; another unmodelled attribute is not.** A
dummy argument declared ``target`` reaches the PSyIR as an
:py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`, and
``InlineTrans`` refuses a routine having an argument of a type it does not
model, because it cannot tell whether binding the formal to the actual is
enough. ``TARGET`` is the case where it is: the attribute says only that a
pointer somewhere may be aimed at the actual, and an inlined body creates no
pointer and invalidates none. So before the callee is inlined, a formal
whose declaration carries nothing beyond its type, its shape, its ``INTENT``
and ``TARGET`` is given the partial datatype the frontend parsed out of that
declaration, and the refusal has nothing left to fire on. The rewrite is
made on the copy of the callee the capture works on, so a later capture of
another kernel calling the same helper meets the routine as its own module
declares it.

Nothing else is relaxed with it. ``permit_unsupported_type_args`` is not
passed, so a formal declared ``POINTER``, ``ALLOCATABLE``, ``OPTIONAL`` or
``VALUE`` -- or ``TARGET`` together with one of them -- is left as it is and
refused in ``InlineTrans``'s own words. This is what puts LFRic's
``subgrid_vertical_support_mod`` helpers, whose read column is declared
``real(kind=r_tran), target, intent(in) :: field(nlayers)``, past that
refusal. The local ``real(kind=r_tran), pointer :: field_ptr(:)`` those
routines aim at either the column or a logarithm of it is the subject of the
next rule.

**A local POINTER aiming at whole arrays is a View handle.** A helper that
declares ``real(kind=r_def), pointer :: p(:)``, aims it at one whole array
or another, and then reads ``p(k)`` is not using the pointer as storage. It
is giving one array a second name, and every ``p(k)`` after ``p => x`` reads
``x(k)``. A Kokkos ``View`` says the same thing: a ``View`` handle assigned
from another names the same elements, so the whole of the translation is
``p = x;`` and the subscripts are left as the Fortran wrote them.

So a local declared with ``POINTER`` and nothing else PSyclone cannot model
is re-declared, before ``InlineTrans`` sees it, as an array of the type its
declaration was parsed into -- which is what lets the routine be inlined at
all, since a local of a type ``InlineTrans`` cannot place is a refusal of
the whole callee -- and the region records it as an alias of the arrays it
is aimed at. The generated region declares it ``decltype(x) p;`` for the
first of those arrays and writes each pointer assignment as a handle
assignment. This is what carries the
``subgrid_vertical_support_mod`` shape, ``field_ptr => log_field`` in one
branch of a flag and ``field_ptr => field`` in the other.

Everything outside that equivalence is refused by name, because a ``View``
handle does not reproduce it: a target that is a section or an expression,
which aims the pointer at part of an array rather than at the array;
``associated``, which asks a question a ``View`` has no answer to;
``allocate`` or ``deallocate``, which make the pointer storage of its own;
a statement PSyclone could not model naming the pointer, ``nullify`` among
them, which may do either without saying so; the pointer passed as an actual
argument, which hands the question to a routine this cannot read; targets
differing in intrinsic, kind or rank, which are more than one handle can
hold; and targets of which one is a kernel argument and another a
kernel-local array, since the first is a ``View`` of the space the region's
data is in and the second a ``View`` of the launch's scratch, and no one
handle can hold both.

**A name a wildcard** ``use`` **was to supply is resolved before inlining.**
Inlining merges the callee's symbol table into the call site's, and a name
both tables hold has to mean the same thing in each. It may be held
differently: LFRic's FFSL kernels name ``W``, ``S``, ``E`` and ``N`` in a
``use`` of ``reference_element_mod``, and a support routine written beside
the kernel reads the same ``S`` while holding nothing of its own, so one
scope has the name as an import and the other says nothing about where it
came from. That is
:py:meth:`~psyclone.psyir.symbols.SymbolTable.check_for_clashes`'s refusal
about a symbol *present in both tables but unresolved in one*, and it stops
``ffsl_flux_xy_panel_remap_code``, ``hori_dep_dist_ffsl_sphere_code`` and
their neighbours.

The message names a wildcard ``use`` because that is the import which would
settle it, not because one was written. Two things leave a scope unable to
say where a name came from: a wildcard ``use``, and being copied away from
the scope that named it -- ``InlineTrans`` copies the callee, and a routine
detached from its Container no longer has the ``use`` its module wrote.
LFRic's is the second.

So before ``InlineTrans`` is applied, each name the two scopes share is
settled. One that is unresolved in a scope's own table is looked up in the
module the other scope names, through
:py:meth:`~psyclone.psyir.symbols.SymbolTable.resolve_imports` and the
``ModuleManager``'s search path, and becomes an import of that module too.
One a scope holds nothing of and reads from the Container it sits in is
copied down into the routine's own table as a ``use ..., only`` of its own,
so the copy carries the declaration rather than losing it. Only names the
two scopes share are settled -- not every name a wildcard ``use`` might
supply -- and the work is done on the copy of the callee the capture works
on, so the module tree is left as it was for a later kernel.

Where the module cannot be read, or does not publish the name, nothing is
asserted about it: the symbol is left unresolved and the refusal stands in
``InlineTrans``'s own words, which name the module that was not read.

**An imported name a later pass needs the type of is read.** A name brought
in by a ``use ..., only`` that PSyclone never had to type is a bare
:py:class:`~psyclone.psyir.symbols.Symbol`, recording which module it came
from and nothing more. Lowering an array section to a loop needs a type:
``ffsl_flux_z_ppm_code`` bounds a section by ``eps_r_tran``, a
``real(kind=r_tran), parameter`` of LFRic's ``constants_mod``, and the
section lowering refuses *the supplied node should be a Reference to a
DataSymbol*. So each such name a call site or a callee reads is resolved
against the module it already names, on the search path the capture passes
with ``-d``, and the section lowering and the module-constant argument both
then see the module's own declaration, its type and its value. A module the
search path cannot reach leaves the name as it was, and the pass that needed
the type refuses as it did before.

**Actual and formal types agreeing is scored, not identical.**
:py:meth:`~psyclone.psyir.nodes.Call.get_callee` scores each candidate
routine through
:py:func:`~psyclone.psyir.nodes.argument_matching.match_argument`: zero
where every actual argument's type is the formal argument's, and one more
for each argument that agrees only in what Fortran requires of it. Three
such arguments arise in the LFRic kernels. A literal written without a
kind -- ``2.0`` passed where the formal argument is ``real(kind=r_def)``
-- carries no precision of its own to disagree with. An actual whose type
PSyclone cannot resolve, typically a module datum whose declaration holds
an attribute the PSyIR does not model, is unknown rather than wrong. And
an array section passed where the formal argument is declared with an
explicit shape PSyclone reads only partially is compared against that
partial type instead of being rejected for the declaration's remainder.

The lowest-scoring candidate is the callee. Relaxing a comparison this
way is safe only while an ambiguity it creates is caught, so two
candidates tying on a non-zero score are refused rather than chosen
between, naming both and the score. A generic interface whose candidates
differ only in the kind of an argument the call passes a kindless literal
to is the case this arises in, and Fortran would not settle it either.

**An array-valued assignment is lowered to an explicit loop.** Two
shapes reach the lowering. A whole-column array section such as
``a(i:j) = ...``, which the finite-volume kernels use to assign a column
as a unit, is one; a whole-array assignment naming no section at all --
``vector = 0.0_r_def``, where ``vector`` is declared with a shape -- is
the other, and it is expanded into a section by
:py:class:`~psyclone.psyir.transformations.Reference2ArrayRangeTrans`
first, which makes explicit the section it already meant.
:py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
then rewrites both into loops, before the region is described.

The generated region has no way to say ``a(i:j)``, so a shape that
transformation refuses is refused here too, with its reason quoted. The
refusal that arises in GungHo is a loop-carried dependency, which no
order of generated loops could honour.

**An array-valued intrinsic is generated, not refused.** ``MATMUL``,
``DOT_PRODUCT``, ``SUM``, ``MINVAL``, ``MAXVAL`` and ``TRANSPOSE`` are
written by the backend as loop nests over the destination, and no array
temporary is created for any of them: one element of a contraction is a
scalar reduction, one element of a transpose is an index swap, so there
is nothing to hold between the two. ``MATMUL(TRANSPOSE(m), x)`` follows
from that rather than needing a case of its own. What is refused is a
fold this shape cannot express -- a ``dim`` that is not a literal or
names no dimension the operand has, a ``mask``, a reduction nested
directly inside another -- and an operand that is neither a whole array
nor a section of one. ``RESHAPE`` is generated only where the source is
rank 1 and the shape is a literal constructor, which makes it an index
map over storage the two languages already agree about.

An operand refused for its shape is refused by :py:meth:`validate`, not
discovered part-way through generating the region. What decides whether
this tier can write a statement is whether it can take the shape of the
statement's operands, so ``validate`` asks it to take each of them, over
stand-in descriptions of the body's own arrays, and reports what it
refuses in the backend's own words. The refusal that arises in GungHo is
``RESHAPE`` of a constructor of literals --
``sci_w3_to_w2_correction_code``'s ``reshape([5, 3, 4, 2, 5, 3, 4, 2],
[2,4])`` -- which is a value with no place in memory for a reshape to
re-view. The element rules above are not asked there and stay the
backend's: whether one element of a fold can be written is settled over
indices that do not exist until the nest is generated.

A section that is not in an assignment at all is judged by where it is
instead. One that is an actual argument of a call is left to the call:
inlining the callee takes the argument away with it, so the section is
gone before this rule could have refused it, and where the callee cannot
be inlined that is the refusal worth reporting. One in the bounds of an
``ALLOCATE`` is read as the shape it states, below. Anywhere else it is
beyond what lowering can reach and is refused before the backend sees it.

**An array constructor fills an array; it is not a value.** A kernel
writing ``v_dot_n = (/ -1.0, 1.0, 1.0, -1.0 /)``, or filling one
full-extent dimension of an array as ``vert_vec(:,qp1,qp2) = (/ ... /)``,
is generated as one assignment per element, into the array the kernel has
already declared and from the origin that declaration gives. A braced
initialiser is not the alternative it looks like: C accepts one only on a
declaration, and the array is declared before the statement is reached.
Anywhere else -- an actual argument, an operand of an expression, a
constructor nested inside another -- the constructor has to survive as an
array in its own right, which needs a temporary this region does not
create, so the backend refuses it by naming the position and
:py:meth:`apply` reports that refusal as it does any other the backend
raises. An implied-do constructor never reaches the backend at all: the
PSyIR frontend does not model one, so ``[ (i, i=1,n) ]`` arrives as a
CodeBlock and is refused with every other CodeBlock.

**A** ``DO WHILE`` **loop is accepted** and generated as a C ``while``. It
is never spread over the team: the loops that are have a counter and a
step for a ``TeamVectorRange`` to divide, and a while loop states neither.

**An unlabelled** ``EXIT`` **is accepted** and generated as a C
``break``, which leaves the same loop the Fortran leaves. The loop it
leaves is never spread over the team, since a lambda cannot break the
loop it was launched over; see :py:meth:`_parallel_loops`. An ``EXIT``
that names the construct it leaves is not modelled by the PSyIR
frontend, so it arrives as a CodeBlock and is refused with every other
CodeBlock -- which is also what happens to the whole ``DO`` a named
construct wraps.

``LBOUND``, ``UBOUND`` and ``SIZE`` are resolved from the declaration
rather than evaluated. Each is replaced by the bound the kernel's own
symbol table gives, so ``UBOUND(partial, 1)`` on a local declared
``dimension(nlayers)`` becomes ``nlayers`` and the region never asks a
View for a shape the Fortran has already stated. Most of them are not
written by the kernel author at all: the section lowering above puts them
into the loop bounds of every full-extent assignment it rewrites, which is
why the substitution runs after that lowering rather than before it. What
is required is a plain reference to a declared array and a dimension given
as an integer literal within its rank -- ``SIZE(a)`` needs no dimension
only when ``a`` is rank 1 -- and the extents themselves must satisfy the
grammar described below, whose refusal is passed through unchanged.

A kernel-local automatic array -- a temporary such as
``real(kind=r_def), dimension(nlayers) :: x_new``, whose extent is known
only at runtime -- is placed in Kokkos team scratch, so that the cells
sharing a team do not share a temporary. The region is then launched over
a ``TeamPolicy`` rather than a ``RangePolicy``.

That placement is what the refusals protect. The array's element kind must
be one the ABI names, as a formal's must, and every **name** left in its
extents must be a **kernel argument**: the launch computes its scratch
size before it enters the region, so an extent it cannot name there
cannot be sized. A module constant is refused as an extent for that
reason even though the body may read one elsewhere. A scalar local needs
no scratch and is declared in the region body as before.

A name whose value the declaration states is not left in the extent at
all. ``integer(kind=i_def), parameter :: nfaces = 4`` beside
``real(kind=r_tran), dimension(nfaces) :: v_dot_n`` is the commonest
kernel-local shape GungHo has, and the value is in the Fortran: the
extent is resolved to ``4`` before it is read, by the same substitution
that already replaces such a name in the body. Only names the declaration
has values for are replaced, so ``dimension(order+1,nfaces)`` keeps the
formal and loses the constant.

An extent is a declared bound written as C, not a name copied over, so it
need not be a single symbol. ``dimension(max_length,4)`` and
``dimension(nlayers+1)`` are both accepted, and both a formal and a local
are read the same way. What is required is an integer expression over
kernel arguments and literals using ``+``, ``-``, ``*`` and ``/``.
Division is carried rather than refused because Fortran and C++ agree
about it -- both truncate an integer quotient toward zero -- so
``dimension((stencil_size + 1) / 2)`` is sized as the kernel declared it.
What the two languages do not agree about is an allocation of negative
size, which neither defines, so a launch whose scratch extent divides
emits a guard that says which rule it is relying on and aborts rather
than requesting one. A call is still refused: ``max(nlayers, 1)`` and
``pow(nlayers, 2)`` carry a comma, which a ``shmem_size`` argument may
not.

**An allocated local is a declared one written the other way round.**
``real(kind=r_def), allocatable, dimension(:) :: partial`` followed by
``allocate(partial(nlayers))`` states the same shape as
``dimension(nlayers)`` would, one statement later, so the ``ALLOCATE``'s
bounds are read into the declaration and both it and the matching
``DEALLOCATE`` are removed. Every rule about a local array then applies
unchanged, the bounds being subject to the same grammar as a declared
shape. Four allocations are refused instead, each by name: one whose
extent the launch cannot evaluate, since the scratch size is computed on
the host from the region's own scalars and ``allocate(t(minval(map)))``
is not one of them; one inside a loop, which is a different array on each
trip rather than one array with a size; one carrying ``stat``, ``errmsg``,
``source`` or ``mold``, each of which says something the reserved scratch
does not carry; and a second allocation of an array already allocated,
scratch being reserved once with one shape for the whole region.

An assumed-shape local -- one carrying ``dimension(:)`` and no
``ALLOCATE`` -- states no shape anywhere the region can read, and is
refused by name; a *formal* declared the same way is measured at the call
instead, below. A shape the C writer cannot render at all, such as
``dimension(MAX(nlayers-n,1))``, is refused with the writer's own reason
attached. Every one of these is refused by :py:meth:`validate` rather
than discovered by :py:meth:`apply`.

**A declared lower bound need not be 1.** ``dimension(0:nlayers-1)`` and
``dimension(-nlayers:nlayers)`` are accepted alongside
``dimension(nlayers)``: the View is sized by the span the declaration
gives, ``ub - lb + 1``, and every subscript of that array has ``lb``
subtracted from it on the way to the zero-based element, so
``u_e(k)`` over a local declared ``dimension(0:nlayers)`` becomes
``u_e((k - 0))``. The lower bound must satisfy the same grammar as the
upper -- an integer expression over kernel arguments and literals using
``+``, ``-``, ``*`` and ``/`` -- and is refused on the same terms when it
does not. A bound of 1 renders exactly the source it rendered before this
was accepted, the span folding back to the upper bound alone.

The subtraction is written out even where it is zero, because it is
applied in one place -- the back-end's generation of an array accessor --
and a subscript that escaped it would compile and give a wrong answer
rather than fail. What is gained by omitting it is what the C++ optimiser
removes anyway.

**An assumed-shape formal is measured rather than refused.** A
boundary-condition kernel writes ``real(kind=r_def), intent(in) ::
normals(:,:)`` and lets Fortran take the extent from the actual. That
extent is nowhere in the kernel, but it is not unknown: the PSy layer
holds the array it is the shape of. So the region carries one integer
formal per dimension the declaration left out, appended after the
kernel's own and passed as ``SIZE(actual, dim=n)``, and the declaration
is rewritten over it before anything else reads the shape. The View is
sized by that formal, and every ``SIZE``, ``LBOUND`` and ``UBOUND`` the
body asks of the array resolves to it exactly as a declared extent does.

The origin, however, is not taken from the call. Fortran gives an
assumed-shape dummy a lower bound of 1 whatever the actual was declared
from, the dummy being a new descriptor over the actual's elements rather
than the actual itself, so the shift of the paragraph above is the
declaration's own 1 and the caller contributes only the size.

A formal that states one bound and leaves the other -- ``dimension(0:)``
-- is refused by name for that reason: its origin would come from the
kernel and its extent from the caller, and a View shaped out of two
places at once is what reading one declaration once exists to prevent.
The refusal says which of the shapeless declarations it found, so a
reader knows whether to look at the kernel or at its caller.

**A loop inside the kernel body may be spread over the team**, and which
loops those are is PSyclone's own judgement rather than this
transformation's: each is put to
:py:meth:`~psyclone.psyir.tools.DependencyTools.can_loop_be_parallelised`,
and one it accepts becomes a ``Kokkos::TeamVectorRange`` over the team's
members. Only the outermost of an accepted nest is taken, because a
``TeamVectorRange`` may not be nested inside itself, and a loop whose step
is not 1 is left alone, because that range gives every member a unit
stride. In practice these are the level loops: a GungHo kernel's outer
loop runs over a column's levels, and the loops that survive the analysis
are the ones whose iterations touch disjoint elements rather than sweeping
a recurrence.

A kernel with such a loop is launched over one team per cell -- the cell
is the team's league rank -- **whether or not it has an automatic array**.
The two selections are separate, and the region shows it: a kernel with
scratch and no acceptable loop keeps the flat launch, where the team is a
way of owning a per-member temporary and its scratch is per member; a
kernel with both takes the one-team-per-cell shape and its scratch becomes
per team, shared by the members working on that column. A kernel with
neither keeps the ``RangePolicy``.

Everything outside a spread loop runs on **every** member of the team, on
that member's own copy of the scalar locals. That is harmless for a scalar
but not for an array, so an array write outside a spread loop is made by
one member under ``Kokkos::single`` and published to the rest by a team
barrier before the next statement reads it. An array constructor
assignment is such a write even though it names its target without
subscripting it, and all of the element assignments it becomes go inside
one ``Kokkos::single``, which is both cheaper than one region each and
what the Fortran said. A barrier also follows each
spread loop, unconditionally: deciding whether a later reader needs it is
a second analysis this transformation does not do.

``"team_size"`` is the one option, an optional positive ``int``. Absent,
the policy asks for ``Kokkos::AUTO`` and the backend sizes the team --
which on the OpenMP backend is **one member**, so on a host build the
leagues carry all of the parallelism and a spread loop runs serially. A
host build reaches the team-level concurrency only by naming a size here.

**A local is also read for its name, not only its type.** The generated
launch declares identifiers of its own in the scope the kernel body is
generated into, and a kernel-local of the same name would shadow one and
then overwrite it -- a wrong answer rather than a compile error. The bound
formals are three, :py:attr:`_BOUND_NAMES`, and are checked whatever the
launch shape: a loop takes at most two of them, but which two depends on
the loop rather than on the kernel, so all three are reserved for every
kernel and a capture cannot be made to depend on the invoke that reached
it. The two team launches add ``body``, ``league_size``,
``probe``, ``rank``, ``scratch_bytes``, ``team`` and ``team_size``, which
are checked for a kernel that has an automatic array to place **or** a
loop to spread -- either reaches a launch that declares them. The launch
*index* is the exception: it is renamed rather than refused, because a
kernel declaring ``cell`` is a real GungHo shape and the fix is one name
in two places rather than seven threaded through two launch shapes.

**A name Fortran allows and C++ reserves is refused, formal or local.**
Fortran reserves no words, so ``const``, ``new`` and ``operator`` are
ordinary variable names -- and ``const`` is one
``project_eliminated_theta_q32_kernel_mod`` declares. Written out as an
identifier it gives ``const float const,`` in the signature, which no
compiler takes. The names are :py:attr:`_CXX_KEYWORDS` and the comparison
is case-sensitive, only a name already lower case colliding with the
keyword. A rename is not attempted: it would have to reach every place the
backend writes a name, and a refusal that says which name it was costs a
region rather than risking a wrong one.

**A name the body reads that is not one of its arguments reaches the
region one of four ways.** A module-level ``parameter`` declared beside
the kernel -- ``integer(kind=i_def), parameter :: nfaces = 4`` -- is
written into the region as its value. It has to be: a kernel module is
``private`` by default and publishes only its ``_code`` routine, so
importing the name into the PSy layer would not compile. The value need
not be a literal: one written as an arithmetic over other parameters is
folded to what those state, and every name in it goes with it.

An array ``parameter`` has no single value to substitute -- its subscripts
are computed where a scalar's use is not -- so its elements are declared
in the generated launch body as a ``const`` array of its own, beside the
body's other locals. Not at file scope, which is what a ``parameter``
beside a kernel most resembles: a namespace-scope array is host data and
a device compiler will not read one. One built by a call, ``reshape``
among them, states no elements to declare and is refused, as is one of
rank above one, a C array being written in the one storage order that
agrees with Fortran's subscripts only in one dimension.

A constant *imported* from another module is passed by value, which needs
its kind, and so needs the source of its container on PSyclone's module
search path; without that it is refused with a message naming the module
to add rather than a guess at its width.

A **variable** of the kernel's own module -- ``real(kind=r_def), public ::
profile_heights(100)``, which is how the profile reaches
``profile_interp_kernel_mod`` -- is state rather than a value, so the
region takes it as a formal of its own and the PSy layer imports it from
the kernel module to pass it: a scalar by value, an array as a read-only
View of the extents the declaration states. What the module holds when the
launch is made is what the region sees, which is why a body that assigns
to one is refused rather than quietly losing the assignment. So is one the
module keeps ``private``, since there is no name to import; one whose
declared extents name a size the region has no argument for; and one with
no declared extents at all, ``allocatable`` among them. A variable
declared inside the *routine* with an initialiser is refused for a
different reason: Fortran gives that one the ``SAVE`` attribute, so it
keeps a value from one call to the next that the region has nowhere to
put, and nothing outside the routine declares it for the PSy layer to
pass.

Last, a name appearing only as an intrinsic's ``kind`` argument -- the
``r_def`` of ``real(x, r_def)`` -- is none of these: it names a type, the
cast consumes it, and the region carries the width rather than the name.

**An intrinsic the backend cannot spell is refused by** :py:meth:`validate`
**rather than discovered by** :py:meth:`apply`. The only thing that knows
what the backend can write is the backend, so ``validate`` asks it: every
intrinsic the body holds is put to the writer with its arguments replaced
by references of their own types, and the ones that raise are reported as
``NAME/arity``. A second list kept beside the writer would be a list to
keep in step.

The writer is asked twice, because an intrinsic can be refused for two
different reasons and only one of them is a missing spelling. That probe
steps over an array-valued intrinsic where the array tier writes it -- a
right-hand side that is not itself a constructor -- because no handler
writes it there and a handler's answer would be about the wrong thing.
The second asks that tier what it cannot take the shape of, as the
contract above describes. Between them, nothing this transformation
accepts is left for the backend to refuse.

It captures all information needed by the Kokkos backend before lowering
the LFRic loop. The LFRic loop is then lowered so that its bound setup and
halo-dirty calls are retained, and only the resulting generic loop is
replaced.

**A write two cells share has two answers, and both are generated.**
Two cells of one launch meet at a dof whenever the field they write is on
a function space LFRic does not name discontinuous, and they meet there in
one of two ways. Under ``gh_inc`` and ``gh_readinc`` each cell
*contributes* to the dof, which is a read-modify-write of it, so a launch
running both cells at once would lose one of the two contributions. Under
``gh_write`` to a continuous space each cell *replaces* it, which loses no
contribution because there is none, and is still two threads writing one
element.

**A dof-iterating kernel is captured with plain stores, whatever its
spaces say.** Sharing between cells is a property of *cell* iteration:
two cells' dofmaps meet at a dof on a continuous space, and both cell
iterations write it. A loop over LFRic's ``dof`` or ``owned_dof``
iteration space visits each dof once and writes it once, so there is no
second writer to make atomic and no colour to separate, and a
``gh_write`` on ``any_space_1`` from such a kernel is generated as an
ordinary assignment where the same metadata on a cell kernel would take
an ``atomic_store``. The metadata predicates alone would not say so --
they read the access and the function space, and neither mentions what
the loop iterates over -- so the question is answered from the kernel's
``operates_on`` before they are asked.

Answering it there also settles what would otherwise be a refusal about
the wrong thing. Which *formal* carries a shared field is read by walking
a cell kernel's argument order -- ``nlayers``, the data of each field,
then ``ndf``, ``undf`` and the dofmap of each space -- where a dof
kernel's formals are its fields and scalars alone. The two counts
disagree, and the disagreement was reported as a kernel this
transformation could not describe. It is instead read as the question
being the wrong one for that kernel: ``swift_inner_update`` and
``swift_outer_update`` in ``ffsl_advective_updates_alg_mod`` are captured
rather than refused for a formal count they cannot meet.

The default answer to both is an atomic, and the two kinds of sharing take
different ones. A contribution is written as ``Kokkos::atomic_add`` -- or
``_sub``, ``_mul``, ``_div``, by the operator the update carries -- on the
View element; a replacement is written as ``Kokkos::atomic_store`` of the
value the statement computed. It is applied per argument and not per
region, so a ``gh_write`` to a *discontinuous* space in the same kernel
stays a plain assignment, and it is the update and not the read that is
atomic, so a ``gh_readinc`` reads its element plainly and updates it
atomically. Where a ``gh_write`` field is written by a read-modify-write
all the same, the update is what is generated: an ``atomic_store`` of
``acc + src`` would drop one cell's contribution where ``atomic_add``
keeps both.

**What an atomic store buys is narrow, and worth being exact about.** It
makes the write indivisible, so the element is written whole and no reader
sees a value neither cell stored. It decides nothing about which of the
two cells stores last, and it does not make the two values agree. LFRic
permits ``gh_write`` on a continuous space precisely because the kernel's
author guarantees that every iteration writes the same value to a given
shared dof -- see :ref:`lfric-kernel-valid-access`, which says in the same
breath that PSyclone cannot check it -- and where that guarantee holds the
order does not matter. Neither the metadata nor the body records the
guarantee, so the transformation does not rely on it; the atomic is what
can be done without it.

One shape is refused rather than stored: a statement whose value reads the
element it replaces. The read happens before the store and outside it, so
making the store indivisible leaves the race exactly where it was, and the
refusal names colouring, which removes it.

**A caller who knows the stores are disjoint may say so.** The default is
conservative -- a ``gh_write`` to a continuous space is read as shared
because the space says the dofs *can* be shared -- and many such kernels
write only dofs their own cell owns. Nothing here can tell the two apart,
so it is the caller's to state, through the ``disjoint_writes`` option of
:py:meth:`apply`, whose name says what is being asserted about the kernel
rather than what the transformation should emit, because it is the
assertion that has to be true. Under it the stores are generated as plain
assignments and the fields are not read as shared anywhere else either, so
``atomics=False`` on such an uncoloured loop stops being a refusal. Three
ways of making the assertion are refused rather than resolved: on a
coloured loop, which says the opposite about the same loop; beside an
explicit ``atomics`` request, which asks for an answer to the sharing it
denies; and on a kernel with a ``gh_inc`` or ``gh_readinc`` argument,
whose metadata states a sharing no assertion about the loop makes untrue.

The other answer is colouring, which is what LFRic's own OpenMP path
takes. A loop
:py:class:`~psyclone.domain.lfric.transformations.LFRicColourTrans` has
already rewritten is accepted: the inner ``cells_in_colour`` loop is the
one captured, the outer loop over colours stays in the PSy layer and
enters the region once per colour, and no atomic is generated, because
the cells of one colour meet at no dof. Such a region carries three
arguments an uncoloured one does not -- LFRic's colour map, the colour
being launched, and the number of colours -- and declares its cell from
them, ``const int cell = cmap(colour - 1, cell_in_colour) - 1;``. The
number of colours is not redundant beside the map: the map crosses the
ABI as bare storage and is rebuilt as a rank-2 View inside the region,
where that number is the ``LayoutLeft`` stride and so the extent that
has to be exact.

Which answer is used follows the loop unless :py:attr:`_ATOMICS_OPTION`
says otherwise, and the two are alternatives rather than a ranking. Both
are correct; which is faster is a measurement on a GPU, and neither this
class nor the branch that added the second answer has taken it.

**They do differ in one property that is not a measurement.**
Floating-point addition is not associative, so the order in which the
contributions to a shared dof arrive is part of the answer. An atomic
launch adds them in the order its threads reach the dof, which is cell
order on one thread and an order that varies between runs on more; a
coloured launch adds them in colour order, which is fixed whatever the
concurrency. So an atomic region reproduces a serial Fortran run bit for
bit on one thread and not on several, and a coloured region reproduces
itself on any thread count and a serial Fortran run on none. Measured
over a ten-timestep LFRic model run, the differences are one part in
1e10 or smaller, and neither is a defect.

One thing a coloured region does is worth stating, because a debug build
will say so. Its per-cell Views are strided by the launch's cell count,
which for a coloured launch is the cells of *this* colour, while the
index they are read at is the mesh cell the colour map returns. Under
``LayoutLeft`` the last extent takes no part in the address, so the
addresses are the ones the Fortran computes; a build with
``KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK`` would nonetheless report the index
as out of range.

**A loop this transformation leaves behind may still be transformed
afterwards**, colouring included, even though capturing forces the
Invoke's PSy-layer symbols to be set up early. The region's actual
arguments are built by
:py:class:`~psyclone.domain.lfric.KernCallArgList`, which reads symbols
that
:py:meth:`~psyclone.domain.lfric.LFRicInvoke.setup_psy_layer_symbols`
specialises, so that pass has to run here rather than at code generation:
by then the kernel that would supply them has been removed from the tree.
It is not idempotent, so the call code generation would otherwise make is
suppressed, and whatever a later transformation has since made necessary
is added instead by
:py:meth:`~psyclone.domain.lfric.LFRicInvoke.complete_psy_layer_symbols`.
Colouring is the case that needs it:
:py:class:`~psyclone.domain.lfric.transformations.LFRicColourTrans`
creates the colourmap symbols and replaces one loop with two, and without
that completion the colourmaps would be declared and never assigned and
the new loops would keep the placeholder bounds
:py:class:`~psyclone.domain.lfric.LFRicLoop` gave them.

The one thing completion cannot repair is a colourmap look-up in an
Invoke that has no mesh object -- one built without distributed memory
whose loops were all uncoloured when the capture ran -- because the mesh
is obtained from a kernel argument that capture has removed. That raises
:py:class:`~psyclone.errors.GenerationError` at code generation, naming
the Invoke, rather than emitting a look-up on an unassigned pointer.
Colouring such an Invoke *before* capturing it needs no repair and is
accepted: the symbols are set up from here, after the colouring, so the
mesh is read from an argument the capture has not yet removed. That is
the order the coloured arm above asks for, and the completion pass then
finds the look-ups already emitted and does not emit them again.

.. autoclass:: psyclone.domain.lfric.transformations.LFRicLoopFuseTrans
    :members:
    :noindex:

.. autoclass:: psyclone.domain.lfric.transformations.RaisePSyIR2LFRicKernTrans
    :members:
    :noindex:

.. autoclass:: psyclone.transformations.LFRicOMPParallelLoopTrans
    :members:
    :noindex:

.. autoclass:: psyclone.transformations.LFRicAsyncHaloExchangeTrans
    :members:
    :noindex:

.. autoclass:: psyclone.transformations.LFRicColourTrans
    :members:
    :noindex:

.. autoclass:: psyclone.transformations.LFRicKernelConstTrans
    :members:
    :noindex:

.. autoclass:: psyclone.transformations.LFRicOMPLoopTrans
    :members:
    :noindex:

.. autoclass:: psyclone.domain.lfric.transformations.LFRicRedundantComputationTrans
    :members:
    :noindex:

.. footbibliography::
