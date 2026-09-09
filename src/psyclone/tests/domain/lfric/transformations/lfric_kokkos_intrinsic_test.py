# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosIntrinsicMixin: intrinsics and allocations."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import IntrinsicCall
from psyclone.psyir.transformations import TransformationError


def test_kokkos_allocate_becomes_scratch(allocate_local_target):
    """An ALLOCATE whose extents are region-entry values becomes scratch.

    A kernel-local ``allocatable`` sized from the kernel's own arguments and
    freed before the routine returns is the automatic array the local-array
    branch already places in team scratch, written the other way round. The
    conversion rewrites the declaration from the ALLOCATE and removes both
    statements, after which every rule about a local array applies unchanged.
    """
    _, loop, _ = allocate_local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "team.team_scratch" in cpp
    assert "partial" in cpp and "swept" in cpp
    for name in ("allocate", "ALLOCATE", "deallocate", "DEALLOCATE"):
        assert name not in cpp


def test_kokkos_allocate_rewrites_the_declaration(allocate_local_target):
    """The converted local is described exactly as a declared one is."""
    _, loop, kernel = allocate_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._lower_allocations(schedule)

    assert [(scratch.name, scratch.extents, scratch.index_offsets)
            for scratch in LFRicKokkosTrans._local_arrays(schedule)] \
        == [("partial", ("nlayers",), ("1",)),
            ("swept", ("nlayers",), ("1",))]
    assert not [call for call in schedule.walk(IntrinsicCall)
                if call.intrinsic in (IntrinsicCall.Intrinsic.ALLOCATE,
                                      IntrinsicCall.Intrinsic.DEALLOCATE)]
    assert loop is not None


def test_kokkos_allocate_refused_when_the_extent_is_not_known(
        unknown_allocate_target):
    """An extent the launch cannot evaluate is refused, with the array named.

    The scratch size is computed on the host before the launch, where the
    only values in scope are the region's own scalars. A reduction over a
    kernel argument is not one of them, so this is a refusal rather than a
    conversion.
    """
    _, loop, _ = unknown_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)


def test_kokkos_allocate_refused_inside_a_loop(looped_allocate_target):
    """An allocation inside a loop is a different array each trip."""
    _, loop, _ = looped_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "outside every loop" in str(error.value)


def test_lfric_kokkos_trans_generates_minval(minval_target):
    """MINVAL over a kernel-local generates, closing wave A's skip.

    ``conservative_neg_fix_code`` was accepted by ``validate`` and refused by
    the writer, and was skipped in the optimisation script for that reason.
    The reduction is now written, so the skip has nothing left to protect.
    """
    _, loop, _ = minval_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::reduction_identity<double>::min()" in cpp
    assert "Kokkos::min(" in cpp
    assert "MINVAL" not in cpp


def test_lfric_kokkos_trans_refuses_an_intrinsic_the_writer_lacks(
        tiny_target):
    """An intrinsic no writer can spell is refused by validate, by name.

    ``validate`` used to accept any body whose *shape* was capturable and
    leave the writer to discover that it could not spell one of its
    intrinsics, which surfaced as a back-end error out of ``apply``. The
    check asks the writer's own tables, so the two cannot part company.
    """
    _, loop, _ = tiny_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "TINY/1" in str(error.value)


def test_kokkos_allocate_refused_when_it_carries_an_option(
        option_allocate_target):
    """An option beside the arrays says something scratch cannot say.

    ``stat=``, ``errmsg=``, ``source=`` and ``mold=`` each state part of what
    the allocation is to do, and the reserved scratch carries none of it.
    Dropping the statement would drop the option with it, so the refusal
    names it, and names the arguments as they were written.
    """
    _, loop, _ = option_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'mold'" in str(error.value)
    assert "'0.0_r_def'" in str(error.value)


def test_kokkos_allocate_refused_when_allocated_twice(twice_allocate_target):
    """Two allocations of one local are two shapes over one reservation."""
    _, loop, _ = twice_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "allocated more than once" in str(error.value)


def test_kokkos_allocate_refused_when_it_states_no_shape(
        shapeless_allocate_target):
    """An allocation stating no bounds leaves nothing to reserve."""
    _, loop, _ = shapeless_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "states no explicit shape" in str(error.value)


def test_kokkos_allocate_refused_for_a_module_array(module_allocate_target):
    """Allocating module storage is not the kernel-local case.

    A module-scope allocatable outlives the region and is shared with
    whatever else the module lets at it, so giving it the shape the statement
    states and reserving scratch for it would move storage the model reads
    after the invoke.
    """
    _, loop, _ = module_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'profile_heights'" in str(error.value)
    assert "not a kernel-local allocatable array" in str(error.value)


# set_exner_code's shape: a whole array assigned the value of a MATMUL,
# `exner_e(:) = MATMUL(inv_mass_matrix_w3, rhs_e)`. The section lowering
# refuses this assignment -- ArrayAssignment2LoopsTrans takes only a
# scalar-valued or elemental right-hand side -- so it has to be left to the
# backend, which writes the contraction as a nest over the destination.
_MATMUL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def), dimension(3,3) :: mass\n"
    "    real(kind=r_def), dimension(3) :: rhs_e\n"
    "    real(kind=r_def), dimension(3) :: column_e\n").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    mass(:,:) = 1.0_r_def\n"
    "    rhs_e(:) = partial(1)\n"
    "    column_e(:) = matmul(mass, rhs_e)\n"
    "    swept(nlayers) = partial(nlayers) + column_e(3)")


@pytest.fixture(name="matmul_target")
# pylint: disable-next=unused-argument
def matmul_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns a whole array from MATMUL."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MATMUL_KERNEL)


def test_lfric_kokkos_trans_generates_a_matmul_assignment(matmul_target):
    """A whole array assigned from MATMUL is left to the backend.

    The section lowering is what rewrites `a(:) = ...` into a loop, and it
    declines a right-hand side that is neither scalar-valued nor elemental.
    Refusing on that would refuse the very shape this tier exists for, so an
    assignment holding one of its intrinsics is kept from the lowering and
    written as a nest instead.
    """
    _, loop, _ = matmul_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "MATMUL" not in cpp
    assert "+= mass(" in cpp
    assert "column_e((_kae_i0 - 1)) = _kae_r0;" in cpp


# sci_w3_to_w2_correction_code's shape, and the one the whole-model capture
# was left unable to model: a whole array assigned the value of a RESHAPE
# whose source is a constructor of literals, `perp_cells(:,:) = reshape([5, 3,
# 4, 2, 5, 3, 4, 2], [2,4])`. The tier writes a reshape by re-viewing its
# source, so the source has to be an array with a place in memory; a
# constructor is a value with none.
_CONSTRUCTOR_OPERAND_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def), dimension(2,4) :: perp_cells\n").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    perp_cells(:,:) = reshape([5, 3, 4, 2, 5, 3, 4, 2], [2,4])\n"
    "    swept(nlayers) = partial(nlayers) + perp_cells(1,1)")


@pytest.fixture(name="constructor_operand_target")
# pylint: disable-next=unused-argument
def constructor_operand_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel reshapes a constructor of literals."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _CONSTRUCTOR_OPERAND_KERNEL)


def test_lfric_kokkos_trans_validate_refuses_a_constructor_operand(
        constructor_operand_target, matmul_target):
    """An operand the array tier cannot shape is refused by `validate`.

    Nothing `validate` accepts may be refused by `apply`, and the array tier
    was the one part of the writer it never asked: the intrinsic probe steps
    over a tier intrinsic on a right-hand side, because the tier rather than
    a handler writes it there, so a RESHAPE of a constructor passed
    validation and was refused half-way through generation. It is refused
    here instead, in the writer's own words -- and the whole-array operand
    the same tier does write is not refused with it, which is the other half
    of the invariant: `apply` after a `validate` that passed generates.
    """
    _, loop, _ = constructor_operand_target
    _, matmul_loop, _ = matmul_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "cannot take the shape of" in str(error.value)
    assert ("an array-valued intrinsic's operand must be a whole array"
            in str(error.value))

    # `apply` refuses it in the same words, from `validate` rather than from
    # the backend part-way through the region.
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().apply(loop)

    assert "cannot take the shape of" in str(error.value)
    assert "cannot express" not in str(error.value)

    # The shape the tier does write is not refused with it.
    LFRicKokkosTrans().validate(matmul_loop)
    assert "MATMUL" not in LFRicKokkosTrans().apply(matmul_loop)
