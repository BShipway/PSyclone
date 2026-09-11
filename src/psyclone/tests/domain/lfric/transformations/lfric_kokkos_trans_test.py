# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately narrow LFRic-to-Kokkos transformation."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _DEPENDENT_SECTION_KERNEL, _SECTION_ALGORITHM, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import CodeBlock, IntrinsicCall, Range
from psyclone.psyir.transformations import TransformationError


def test_lfric_kokkos_trans_validate_leaves_sections_alone(section_target):
    """Validation predicts the lowering; it must not perform it.

    validate() is public and callable on its own, so a caller deciding
    whether a loop is capturable would otherwise silently rewrite the
    kernel it was only asking about.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]

    LFRicKokkosTrans().validate(loop)

    assert len(schedule.walk(Range)) == 3


def test_lfric_kokkos_trans_splices_one_loop(target):
    """The selected loop becomes one typed call and retains halo updates."""
    psy, loop, _ = target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void moist_dyn_gas_kokkos(' in cpp
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp
    assert "map_wtheta((df - 1), cell)" in cpp
    assert "const int *map_wtheta_data" in cpp
    assert ("map_wtheta_data, lfric_kokkos::Role::readonly, "
            "ndf_wtheta, ncells)") in cpp
    assert "const double recip_epsilon" in cpp

    assert "subroutine moist_dyn_gas_kokkos(" in fortran
    assert "bind(C)" in fortran
    # The interface imports the kinds its own arguments declare and no
    # others, so widening the ABI leaves an int/double region untouched.
    assert "use iso_c_binding, only : c_int, c_double" in fortran
    assert "c_float" not in fortran
    # The widths the C++ body assumed, asserted where the two languages meet.
    # The double-precision region needs this as much as the single-precision
    # one does: it is only correct while r_def really is 8 bytes.
    assert "use constants_mod, only : i_def, r_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert ("storage_size(1.0_r_def) == &\n        "
            "storage_size(1.0_c_double))), parameter :: assert_kind_r_def = 0"
            in fortran)
    assert "dimension(*), intent(inout) :: moist_dyn_gas" in fortran
    assert "call moist_dyn_gas_kokkos(" in fortran
    assert "map_wtheta, loop0_stop, recip_epsilon)" in fortran
    assert "call moist_dyn_proxy%set_dirty()" in fortran
    assert fortran.index("call moist_dyn_gas_kokkos(") < fortran.index(
        "call moist_dyn_proxy%set_dirty()")
    assert "call moist_dyn_gas_code(" not in fortran
    assert str(psy.gen) == fortran


def test_lfric_kokkos_trans_derives_a_second_kernel_contract(second_target):
    """Nothing about the region is specific to the first kernel captured."""
    psy, loop, _ = second_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # The region is named after the kernel, and its r_tran scalar is on the
    # ABI as a double even though the kernel the transformation was written
    # against has no scalar at all and never mentions r_tran.
    assert 'extern "C" void scaled_copy_kokkos(' in cpp
    assert "const double scaling" in cpp

    # Two function spaces, so two dofmaps, each given the cell extent the
    # PSy layer would otherwise have sliced away.
    assert ("map_w3_data, lfric_kokkos::Role::readonly, "
            "ndf_w3, ncells)") in cpp
    assert ("map_wtheta_data, lfric_kokkos::Role::readonly, "
            "ndf_wtheta, ncells)") in cpp
    assert "field_out(((map_w3((df - 1), cell) + k) - 1)) = " in cpp
    assert "(scaling * field_in(((map_wtheta((df - 1), cell) + k) - 1)))" \
        in cpp

    assert "subroutine scaled_copy_kokkos(" in fortran
    assert "real(c_double), value :: scaling" in fortran
    assert "dimension(*), intent(inout) :: field_out" in fortran
    assert "call scaled_copy_kokkos(" in fortran
    # No module constant, so the cell count is the last actual argument.
    assert "map_wtheta, loop0_stop)" in fortran
    assert "call scaled_copy_code(" not in fortran


def test_untransformed_invoke_can_be_generated_repeatedly(target):
    """Preparing a temporary generation copy must not mark the source tree."""
    psy, _, _ = target
    first = str(psy.gen)
    assert str(psy.gen) == first


def test_lfric_kokkos_trans_rejects_wrong_node():
    """Only an LFRic loop is a valid capture boundary."""
    with pytest.raises(TransformationError, match="LFRicLoop"):
        LFRicKokkosTrans().validate(CodeBlock(
            [], structure=CodeBlock.Structure.STATEMENT))


@pytest.mark.parametrize("fixture_name", [
    "literal_local_target", "arithmetic_local_target",
    "explicit_one_local_target", "lower_bound_local_target",
    "zero_based_local_target", "negative_origin_local_target",
    "named_constant_local_target", "divided_local_target",
    "allocate_local_target", "minval_target"])
def test_lfric_kokkos_trans_validate_accepts_what_apply_generates(
        fixture_name, request):
    """Every widened shape passes ``validate`` as well as ``apply``.

    The agreement between the two is the property Task 3a.4 established, and
    it holds in the accepting direction as well as the refusing one: a
    transformation whose ``validate`` were stricter than its ``apply`` would
    report a capturable loop as blocked and understate its own coverage.
    """
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    trans.validate(loop)
    assert trans.apply(loop)


@pytest.mark.parametrize("fixture_name", [
    "unsized_local_target", "unmapped_local_target",
    "unrenderable_origin_target", "unsized_expression_target",
    "unwritable_shape_target", "unknown_allocate_target",
    "looped_allocate_target", "option_allocate_target",
    "twice_allocate_target", "shapeless_allocate_target",
    "module_allocate_target", "tiny_target"])
def test_lfric_kokkos_trans_validate_and_apply_agree_on_locals(
        fixture_name, request):
    """Both refusals are made by ``validate``, not discovered by ``apply``.

    This is the property that was false before local arrays were modelled:
    ``validate`` accepted a kernel like ``tri_solve`` and ``apply`` then
    raised from the backend. A caller asking whether a loop is capturable got
    "yes" and a caller capturing it got an error, about the same loop.
    """
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    with pytest.raises(TransformationError) as predicted:
        trans.validate(loop)
    with pytest.raises(TransformationError) as attempted:
        trans.apply(loop)

    assert str(predicted.value) == str(attempted.value)


@pytest.mark.parametrize("fixture_name", [
    "scalar_bound_target", "element_bound_target", "variable_bound_target",
    "out_of_range_bound_target", "whole_size_target"])
def test_lfric_kokkos_trans_validate_and_apply_agree_on_bounds(
        fixture_name, request):
    """Every bound refusal is made by ``validate``, not found by ``apply``."""
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    with pytest.raises(TransformationError) as predicted:
        trans.validate(loop)
    with pytest.raises(TransformationError) as attempted:
        trans.apply(loop)

    assert str(predicted.value) == str(attempted.value)


def test_lfric_kokkos_trans_validate_leaves_bounds_alone(bound_target):
    """``validate`` predicts the substitution over a copy.

    The schedule it is handed is cached by ``LFRicKern.get_callees``, so a
    substitution made during validation would persist and a second call would
    see a schedule with no enquiries left in it.
    """
    _, loop, kernel = bound_target

    LFRicKokkosTrans().validate(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert any(call.intrinsic in LFRicKokkosTrans._BOUND_INTRINSICS
               for call in schedule.walk(IntrinsicCall))


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_bounds_defer_to_the_section_refusal(
        tmp_path, clear_module_manager_instance):
    """An unlowerable section is not re-reported as a bound problem.

    ``_validate_bounds`` has to lower a copy before it can see the bounds the
    lowering writes, so a schedule the lowering refuses leaves it with nothing
    to check. ``_validate_sections`` owns that refusal, and the coverage
    survey calls each predicate independently, so reporting it twice would
    make one fact look like two blocked patterns.
    """
    # pylint: disable-next=unused-variable
    _, loop, kernel = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _DEPENDENT_SECTION_KERNEL)
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._validate_bounds(schedule)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_sections(schedule)
    assert "cannot lower an array section to a loop" in str(error.value)


def test_team_size_option_reaches_the_launch(level_target):
    """``team_size`` replaces ``Kokkos::AUTO`` in the policy.

    On the OpenMP backend ``Kokkos::AUTO`` is one member, so a host build
    reaches the team-level concurrency only by asking for a size. The option
    is the only way to ask.
    """
    _, loop, _ = level_target

    cpp = LFRicKokkosTrans().apply(loop, options={"team_size": 4})

    assert "TeamPolicy(ncells, 4)," in cpp
    assert "Kokkos::AUTO" not in cpp


@pytest.mark.parametrize("value", ["4", 4.0, True, 0, -1])
def test_team_size_option_is_validated(level_target, value):
    """A team size that is not a positive integer is refused.

    ``True`` is in the list because ``isinstance(True, int)`` is true in
    Python, so a bare integer check would let it through and write
    ``TeamPolicy(ncells, True)``.
    """
    _, loop, _ = level_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop, options={"team_size": value})

    assert ("'team_size' option must be a positive integer, but found "
            f"'{value}'" in str(error.value))


def test_lfric_kokkos_trans_validate_probe_is_not_the_schedule(section_target):
    """``validate()`` leaves the kernel schedule exactly as it found it.

    That is the whole reason the probe is a copy, and it is worth asserting
    directly rather than through the generated text: the probe is lowered,
    has its bounds substituted and its allocations resolved, and every one of
    those would be a side effect of *validating* if it reached the schedule.
    The section kernel witnesses it, because the lowering it would undergo is
    visible in the tree -- an array section becomes a loop nest -- rather than
    only in a symbol's type.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]
    before = schedule.debug_string()
    assert schedule.walk(Range)

    LFRicKokkosTrans().validate(loop)

    assert kernel.get_callees()[0] is schedule
    assert schedule.debug_string() == before
    assert schedule.walk(Range)


def test_lfric_kokkos_trans_describes_itself_without_naming_a_cell():
    """``__str__`` names the loops the transformation now takes.

    PSyclone lists a transformation by this string, and it said
    "cell-column loop" before a loop over dofs was one of them.
    """
    assert str(LFRicKokkosTrans()) == (
        "Capture a supported LFRic loop as a Kokkos launch")
