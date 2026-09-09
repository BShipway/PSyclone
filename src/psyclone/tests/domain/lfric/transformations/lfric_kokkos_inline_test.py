# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosInlineMixin: inlining a called routine."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _ALGORITHM, _HDIV_SECTION_KERNEL, _KERNEL, _LOCAL_ALGORITHM, _LOCAL_KERNEL,
    _SECTION_ALGORITHM, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall
from psyclone.psyir.transformations import TransformationError


# The same kernel passing a column of the local to a routine, which is how
# convert_hdiv_native_code reaches native_jacobian. The section is outside an
# assignment and so beyond the lowering, but the call is beyond the capture
# altogether, and that is the reason worth reporting.
_HDIV_CALL_KERNEL = _HDIV_SECTION_KERNEL.replace(
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)",
    "    call native_jacobian(vector(:,1))\n"
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)")


# A kernel calling a function from a module PSyclone has not read. Without the
# module the frontend cannot tell `helper(k)` from an array reference, so the
# symbol is a plain Symbol until something resolves it -- which is why the
# refusal has to be the one about calls rather than one about module data.
_CALLED_ROUTINE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_mod, only : helper").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - helper(k)")


# A second module holding the sweep, and holding state of its own that the
# sweep reads. This is the shape coordinate_jacobian has -- a GungHo helper in
# a module the kernel `use`s, reading a rotation matrix that module keeps --
# and it is the shape inlining cannot reach: the body could be moved but the
# datum it reads could not, since the kernel's module neither declares it nor
# has any name for it.
_EXTERNAL_STATE_MODULE = """
module helper_state_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  real(kind=r_def) :: relaxation = 0.5_r_def
  public :: sweep_column
contains
  subroutine sweep_column(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - relaxation * source(j)
    end do
  end subroutine sweep_column
end module helper_state_mod
"""


# The kernel that calls it. Written the same way as _MODULE_PROCEDURE_KERNEL
# so that the only difference between the two is where the callee lives.
_EXTERNAL_CALLEE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_state_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# A configuration module holding an array, and the kernel that reads it. The
# kernel's own file says nothing about what ``blend_weights`` is, so the
# frontend leaves ``blend_weights(1)`` as a Call: an indexed name in an
# expression is a function reference or an array element, and only the other
# module's source settles which. This is the shape the limited-area kernels
# have, and it is the shape that reaches a TypeError rather than a refusal
# when PSyclone is asked for the callee.
_ARRAY_LIKE_MODULE = """
module weights_config_mod
  use constants_mod, only : r_def
  implicit none
  private
  real(kind=r_def), public :: blend_weights(3) = &
      (/ 1.0_r_def, 2.0_r_def, 3.0_r_def /)
end module weights_config_mod
"""


_ARRAY_LIKE_CALL_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use weights_config_mod, only : blend_weights").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) * blend_weights(1)")


# The sweep of _LOCAL_KERNEL taken out into a module procedure of the kernel's
# own module. This is the shape inlining can reach: the callee is a procedure
# of the very Container the call site is in, so no import has to be followed
# and no second module's source has to be read. Its formals are an extent, a
# read column and a written one, which is what a GungHo helper looks like.
_MODULE_PROCEDURE_KERNEL = _LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same module procedure reached through a chain: the kernel calls one
# helper and that helper calls a second. Inlining one call exposes the next,
# so the rewrite has to be repeated until none is left rather than run once.
_CHAINED_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    call seed_column(n, source, result)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n").replace(
    "  end subroutine sweep_column\n",
    "  end subroutine sweep_column\n"
    "  subroutine seed_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine seed_column\n")


# A callee that calls itself. Inlining it once leaves a call to it behind, so
# a rewrite run to a fixed point would never reach one; the depth limit is
# what turns that into a refusal naming the routine.
_RECURSIVE_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    result(n) = source(n)\n"
    "    j = n - 1\n"
    "    call sweep_column(j, source, result)\n")


# A kernel handing a column of a rank-2 local to a module procedure, which is
# the shape convert_hdiv_native_code reaches native_jacobian with. The actual
# is a contiguous whole-dimension section, so before inlining it is a section
# outside an assignment; afterwards there is no argument at all, because the
# formal's every use has become a subscript of the local itself.
_SECTION_ACTUAL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def), dimension(nlayers, 2) :: column\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    column(:,1) = partial(:)\n"
    "    call sweep_column(nlayers, column(:,1), swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The target kernel subscripting one field by a constant its *module* imports,
# on both sides of an assignment. That is the shape LFRic's inter-grid
# prolongation has -- 'fine_field(map_fine(SWB, ...))', with SWB an
# 'integer, parameter' from reference_element_mod -- and nothing about it is
# inter-grid: what matters is that the name is resolved through the kernel
# module's symbol table rather than the subroutine's, and that the loop reads
# and writes one array so that the dependence analysis has two subscripts to
# compare symbolically. 'n_moist' is planet_config_mod's integer parameter.
_MODULE_INDEX_KERNEL = _KERNEL.replace(
    "  use planet_config_mod, only : recip_epsilon",
    "  use planet_config_mod, only : n_moist, recip_epsilon").replace(
    "        moist_dyn_gas(map_wtheta(df) + k) = &\n"
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "        moist_dyn_gas(map_wtheta(n_moist) + k) = &\n"
    "            moist_dyn_gas(map_wtheta(n_moist) + k) + &\n"
    "            recip_epsilon * mr_v_at_dof")


@pytest.fixture(name="called_routine_target")
# pylint: disable-next=unused-argument
def called_routine_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel calls an unresolvable function."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CALLED_ROUTINE_KERNEL)


@pytest.fixture(name="module_procedure_target")
# pylint: disable-next=unused-argument
def module_procedure_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel calls a procedure of its own module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MODULE_PROCEDURE_KERNEL)


@pytest.fixture(name="chained_procedure_target")
# pylint: disable-next=unused-argument
def chained_procedure_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel's callee itself calls."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CHAINED_PROCEDURE_KERNEL)


@pytest.fixture(name="recursive_procedure_target")
def recursive_procedure_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel calls a self-recursive procedure."""
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _RECURSIVE_PROCEDURE_KERNEL)


@pytest.fixture(name="section_actual_target")
# pylint: disable-next=unused-argument
def section_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a contiguous section to a module procedure."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SECTION_ACTUAL_KERNEL)


@pytest.fixture(name="external_callee_target")
# pylint: disable-next=unused-argument
def external_callee_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose callee is readable but in another module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _EXTERNAL_CALLEE_KERNEL,
        extra={"helper_state_mod": _EXTERNAL_STATE_MODULE})


@pytest.fixture(name="array_like_call_target")
# pylint: disable-next=unused-argument
def array_like_call_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke reading an array the frontend takes for a call."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARRAY_LIKE_CALL_KERNEL,
        extra={"weights_config_mod": _ARRAY_LIKE_MODULE})


@pytest.fixture(name="module_index_target")
# pylint: disable-next=unused-argument
def module_index_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel indexes a field by a module constant."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _MODULE_INDEX_KERNEL)


def test_lfric_kokkos_trans_defers_a_section_actual_to_the_call(
        tmp_path, clear_module_manager_instance):
    """A section given to a routine is refused for the routine, not the shape.

    ``call native_jacobian(vector(:,1))`` carries two reasons a capture
    cannot proceed: the section stands outside an assignment, and the callee
    cannot be inlined. Only the second is the loop's real blocker -- inline
    the callee and the argument, section and all, goes with it -- so the
    section rule steps aside and lets the inlining rule answer. The coverage
    survey asks each rule on its own and keeps every message, so a rule that
    answered here would report this loop as an array-section blocker that no
    array-section work could ever clear.
    """
    # pylint: disable=unused-argument
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _HDIV_CALL_KERNEL)

    # Asked on its own, as the survey asks it, the section rule has nothing
    # to say about this kernel: every assignment in it lowers, and the one
    # section that does not is the call's argument.
    LFRicKokkosTrans._validate_sections(
        LFRicKokkosTrans._schedule(loop.kernels()[0]))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'native_jacobian'" in str(error.value)
    assert "array section outside an assignment" not in str(error.value)


def test_lfric_kokkos_trans_names_a_called_routine_as_a_call(
        called_routine_target):
    """A function in an unread module is refused as a call, once.

    Without the module the frontend cannot tell ``helper(k)`` from an array
    reference, so the symbol is a plain ``Symbol`` and the constant machinery
    used to claim it as module data it could not pass by value. The survey
    calls each predicate independently, so that made one fact look like two
    blocked patterns.
    """
    _, loop, kernel = called_routine_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._constants(schedule)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'helper'" in str(error.value)


def test_lfric_kokkos_trans_inlines_a_module_procedure(
        module_procedure_target):
    """A procedure of the kernel's own module is inlined into the region.

    The generated region has no Fortran to call into, so the callee's
    statements have to become the kernel's own before the backend sees them.
    ``InlineTrans`` binds the formals to the actuals as it goes, so the
    inlined body reads ``partial`` and ``swept`` where the callee wrote
    ``source`` and ``result``.
    """
    _, loop, kernel = module_procedure_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    # The callee's own statements, with its formals bound to the actuals.
    assert "swept((nlayers - 1)) = partial((nlayers - 1));" in cpp


def test_lfric_kokkos_trans_inlines_a_call_with_a_section_actual(
        section_actual_target):
    """A contiguous section given as an actual reaches the inlined body.

    ``column(:,1)`` is a section outside an assignment, which is beyond what
    the section lowering can reach and is left to this rule by
    :py:meth:`_validate_sections`. Inlining removes the argument altogether:
    every use of the formal becomes a subscript of the local the section was
    taken from, so the region never has to pass an array at all.
    """
    _, loop, kernel = section_actual_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "column" in cpp


def test_lfric_kokkos_trans_inlines_a_chain(chained_procedure_target):
    """A callee that itself calls is inlined to a fixed point.

    Inlining one call exposes the next, so the rewrite is repeated rather than
    run once. The repetition is bounded by
    :py:attr:`LFRicKokkosTrans._INLINE_LIMIT`, which is what makes a chain
    terminate on its own terms rather than on the recursion limit.
    """
    _, loop, kernel = chained_procedure_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "seed_column" not in cpp
    assert LFRicKokkosTrans._INLINE_LIMIT > 1


def test_lfric_kokkos_trans_refuses_an_unresolvable_callee(
        called_routine_target):
    """A callee whose module was never read is refused, naming both.

    The message keeps PSyclone's own text, which names the container the
    symbol was imported from, so a reader is told which module to put on the
    search path rather than that something unnamed could not be found.
    """
    _, loop, _ = called_routine_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'helper'" in message
    assert "helper_mod" in message


def test_lfric_kokkos_trans_refuses_a_recursive_callee(
        recursive_procedure_target):
    """A callee that calls itself is refused rather than inlined forever.

    ``InlineTrans`` has no recursion check of its own -- inlining the body
    simply leaves another call to the same routine behind -- so the depth
    limit is the whole of what makes this terminate.
    """
    _, loop, _ = recursive_procedure_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "sweep_column" in message
    assert str(LFRicKokkosTrans._INLINE_LIMIT) in message


def test_lfric_kokkos_trans_refuses_an_external_callee(
        external_callee_target):
    """A callee outside the kernel's module is refused rather than guessed at.

    ``helper_state_mod`` is on the search path here, so the callee resolves
    and the refusal is not about finding it. The callee reads a datum its own
    module keeps private, so its body cannot be moved into the kernel's
    Container and cannot be inlined from where it is; PSyclone's own message
    says which Container the call site is in.
    """
    _, loop, _ = external_callee_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "column_solve_kernel_mod" in message


def test_lfric_kokkos_trans_refuses_an_array_like_call(
        array_like_call_target):
    """A name PSyclone reads as a call and cannot type is refused, not raised.

    ``blend_weights(1)`` is an element of an array ``weights_config_mod``
    keeps, but the kernel's own file does not say so and the frontend leaves
    it as a Call. Asking PSyclone for that callee reaches the datum and
    raises :py:exc:`TypeError` rather than a
    :py:class:`~psyclone.psyir.transformations.TransformationError`, since a
    DataSymbol cannot be specialised into a RoutineSymbol. The mixin turns
    that into a refusal, so the caller is told the kernel is not captured
    instead of seeing PSyclone's traceback.
    """
    _, loop, _ = array_like_call_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'blend_weights'" in message
    assert "specialise" in message


def test_lfric_kokkos_trans_validate_keeps_the_module_scope_chain(
        module_index_target):
    """A name the kernel *module* imports is still resolvable in the probe.

    ``validate()`` predicts the rewrite on a copy, and the copy has to keep
    the FileContainer the kernel was read from: a Routine copied on its own
    leaves every symbol the module ``use``d at module level out of the scope
    chain. The dependence analysis is where that is felt, because comparing
    two subscripts symbolically means looking each name in them up --
    ``moist_dyn_gas(map_wtheta(n_moist) + k)`` on both sides of one
    assignment, with ``n_moist`` imported by the module.

    The failure this covers is not a refusal: it is
    ``KeyError: "Could not find 'n_moist' in the Symbol Table."`` coming
    straight out of ``SymPyWriter``, which breaks ``validate()``'s contract
    that a rejection is a ``TransformationError``. Found on LFRic's inter-grid
    prolongation, where the constant is ``SWB``; reproduced here on a
    single-mesh kernel, because nothing about it is inter-grid.
    """
    _, loop, _ = module_index_target

    try:
        LFRicKokkosTrans().validate(loop)
    except TransformationError as err:
        pytest.fail(f"the capture contract refused the kernel: {err}")
