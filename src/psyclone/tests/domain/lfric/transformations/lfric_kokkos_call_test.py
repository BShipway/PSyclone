# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosCallMixin: scratch, counter and imports."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _ALGORITHM, _KERNEL, _SECOND_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans


# The same kernel twice in one invoke, so that the captured loop has a sibling
# still counting with the PSy layer's cell counter.
_PAIRED_ALGORITHM = """
program kokkos_paired_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  implicit none
  type(field_type) :: out_field, other_field, in_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling),   &
              scaled_copy_kernel_type(other_field, in_field, scaling))
end program kokkos_paired_test
"""


# The target kernel reading its one imported constant twice. Each reference is
# a separate node, and describing the second would re-resolve a symbol already
# on the ABI and offer the PSy layer the same argument twice.
_REPEATED_IMPORT_KERNEL = _KERNEL.replace(
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "            recip_epsilon + recip_epsilon * mr_v_at_dof")


# The target kernel renaming its imported constant, as LFRic's moisture
# kernels rename the latent heats they read. The module declares one name and
# the body reads another, so the generated PSy layer has to repeat the rename
# rather than import either name alone.
_RENAMED_IMPORT_KERNEL = _KERNEL.replace(
    "  use planet_config_mod, only : recip_epsilon",
    "  use planet_config_mod, only : recip => recip_epsilon").replace(
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "            1.0_r_def + recip * mr_v_at_dof")


@pytest.fixture(name="paired_target")
# pylint: disable-next=unused-argument
def paired_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose captured loop has a cell loop beside it."""
    return _invoke(
        tmp_path, "scaled_copy", _PAIRED_ALGORITHM, _SECOND_KERNEL)


@pytest.fixture(name="repeated_import_target")
# pylint: disable-next=unused-argument
def repeated_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads one imported constant twice."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _REPEATED_IMPORT_KERNEL)


@pytest.fixture(name="renamed_import_target")
# pylint: disable-next=unused-argument
def renamed_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel renames the constant it imports."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _RENAMED_IMPORT_KERNEL)


def test_lfric_kokkos_trans_drops_an_orphaned_counter(second_target):
    """Capturing an invoke's only cell loop undeclares its counter.

    LFRic compiles the PSy layer with -Werror=unused-variable, so a counter
    left declared by the loop that has just been replaced is a build failure
    rather than untidiness.

    """
    psy, loop, _ = second_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "call scaled_copy_kokkos(" in fortran
    assert ":: cell" not in fortran


def test_lfric_kokkos_trans_keeps_a_shared_counter(paired_target):
    """A sibling cell loop still counting keeps the declaration."""
    psy, loop, _ = paired_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # A Loop holds its counter as an attribute rather than as a Reference, so
    # the surviving loop is only visible to a check that asks the loops.
    assert "call scaled_copy_kokkos(" in fortran
    assert "call scaled_copy_code(" in fortran
    assert "integer(kind=i_def) :: cell" in fortran


def test_lfric_kokkos_trans_describes_each_local_array(local_target):
    """``_local_arrays`` names, types and sizes every automatic array."""
    _, _, kernel = local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    scratch = LFRicKokkosTrans._local_arrays(schedule)

    assert [item.name for item in scratch] == ["partial", "swept"]
    assert all(item.c_type == "double" for item in scratch)
    assert all(item.extents == ("nlayers",) for item in scratch)
    # Fortran declares from 1 and C indexes from 0, as for a formal. The
    # offset is the declared origin rendered as C, not an assumed 1, so it is
    # the string the backend writes into the subscript.
    assert all(item.index_offsets == ("1",) for item in scratch)


def test_lfric_kokkos_trans_passes_a_repeated_import_once(
        repeated_import_target):
    """An imported constant read twice reaches the ABI once.

    Each reference is a separate node, so the walk meets ``recip_epsilon``
    twice; describing it twice would re-resolve a symbol already on the ABI
    and hand the PSy layer the same argument in two places.
    """
    psy, loop, kernel = repeated_import_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert [description[:4]
            for description in LFRicKokkosTrans._constants(schedule)] == [
        ("recip_epsilon", "planet_config_mod", None, "double")]

    LFRicKokkosTrans().apply(loop)

    generated = str(psy.gen)
    assert generated.count("map_wtheta, loop0_stop, recip_epsilon)") == 1


def test_lfric_kokkos_trans_carries_a_renamed_import(renamed_import_target):
    """A constant renamed on import is typed, and the rename is repeated.

    ``use planet_config_mod, only : recip => recip_epsilon`` leaves the body
    reading ``recip``, a name ``planet_config_mod`` does not declare. Typing
    it means looking it up in the module under the name it has there, and
    importing it into the PSy layer means writing the rename out again: a
    plain ``use planet_config_mod, only : recip`` would not compile.
    """
    psy, loop, kernel = renamed_import_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert [description[:4]
            for description in LFRicKokkosTrans._constants(schedule)] == [
        ("recip", "planet_config_mod", "recip_epsilon", "double")]

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const double recip" in cpp
    assert "use planet_config_mod, only : recip=>recip_epsilon" in fortran
    assert "map_wtheta, loop0_stop, recip)" in fortran
