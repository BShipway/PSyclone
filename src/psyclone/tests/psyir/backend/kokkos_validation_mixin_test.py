# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the checks the Kokkos writer makes before generating a region.

The checks the writer's own tests reach are left with them; these are the
ones nothing else reaches, each a description no transformation builds but a
hand-written region could.
"""

from dataclasses import replace

import pytest

from psyclone.psyir.backend.kokkos import KokkosScalar, KokkosWriter
from psyclone.psyir.nodes import Routine
from psyclone.tests.psyir.backend.kokkos_test import _region


def _with_argument(index, argument):
    """Return the test region with one argument replaced.

    :param int index: the position of the argument to replace.
    :param argument: what to put in its place.
    :type argument: :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar` |
        :py:class:`psyclone.psyir.backend.kokkos.KokkosView` | object

    :returns: the region with that one argument different.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    region = _region()
    arguments = list(region.arguments)
    arguments[index] = argument
    return replace(region, arguments=tuple(arguments))


def test_the_writer_takes_only_a_region():
    """Anything but a region is refused by what it is."""
    with pytest.raises(TypeError, match="expects a KokkosRegion but found "
                       "'str'"):
        KokkosWriter()("moist_dyn_gas_kokkos")


def test_the_schedule_must_be_a_kernel_schedule():
    """A plain routine is not the body of a kernel."""
    region = replace(_region(), schedule=Routine.create("plain"))

    with pytest.raises(TypeError, match="must be a KernelSchedule"):
        KokkosWriter()(region)


def test_an_argument_must_be_a_scalar_or_a_view():
    """An argument is described as one of the two things the ABI passes."""
    with pytest.raises(TypeError, match="must be KokkosScalar or KokkosView"):
        KokkosWriter()(_with_argument(0, "nlayers"))


def test_an_argument_name_must_be_an_identifier():
    """A name that is not a C identifier cannot be declared."""
    with pytest.raises(ValueError,
                       match="Kokkos argument name '1layers' is invalid"):
        KokkosWriter()(_with_argument(0, KokkosScalar("1layers", "int")))


def test_an_abi_name_is_passed_once():
    """Two arguments may not share the name the C ABI knows them by."""
    region = _region()
    region = replace(region, arguments=region.arguments + (
        KokkosScalar("nlayers", "int"),))

    with pytest.raises(ValueError,
                       match="Duplicate C ABI argument 'nlayers'"):
        KokkosWriter()(region)


def test_the_cell_count_must_be_a_scalar_argument():
    """The launch is bounded by a formal the region passes."""
    region = replace(_region(), cell_count="elsewhere")

    with pytest.raises(ValueError,
                       match="Cell count 'elsewhere' is not a scalar"):
        KokkosWriter()(region)


def test_every_kernel_argument_is_described():
    """A formal of the kernel the region leaves out cannot be passed."""
    region = _region()
    region = replace(region, arguments=region.arguments[1:])

    with pytest.raises(ValueError,
                       match="does not describe kernel arguments: nlayers"):
        KokkosWriter()(region)


@pytest.mark.parametrize("change, error, message", [
    ({"data_name": "1data"}, ValueError,
     "Kokkos View data name '1data' is invalid"),
    ({"extents": ()}, ValueError,
     "'moist_dyn_gas' must have extents that are integer expressions"),
    ({"extents": ("undf_wtheta", "ncells")}, ValueError,
     "'moist_dyn_gas' dimensions do not match"),
    ({"index_offsets": (1.5,)}, TypeError,
     "'moist_dyn_gas' index offsets must be integers"),
    ({"extents": ("undf_wtheta", "ncells"), "extra_indices": ("1cell",)},
     ValueError, "'moist_dyn_gas' has an invalid region index"),
    ({"random_access": True}, ValueError,
     "'moist_dyn_gas' uses RandomAccess but is"),
])
def test_a_view_is_refused_by_its_description(change, error, message):
    """Each part of a View's description is checked before it is declared.

    The View is the one the kernel writes, so it is neither read-only nor
    indexed by the cell until a case makes it so.
    """
    view = replace(_region().arguments[1], **change)

    with pytest.raises(error, match=message):
        KokkosWriter()(_with_argument(1, view))
