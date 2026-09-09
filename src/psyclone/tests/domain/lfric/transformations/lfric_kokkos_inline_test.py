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
    _ALGORITHM, _COLUMN_SELECT_MODULE, _EDGE_INDEX_MODULE,
    _HDIV_SECTION_KERNEL, _KERNEL, _LOCAL_ALGORITHM, _LOCAL_KERNEL,
    _SECTION_ALGORITHM, _SIBLING_CALLEE_KERNEL, _SIBLING_CALLEE_MODULE,
    _SIBLING_STATE_MODULE, _STATIC_SIBLING_MODULE, _USED_FUNCTION_KERNEL,
    _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall
from psyclone.psyir.symbols import RoutineSymbol
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


@pytest.fixture(name="used_function_target")
# pylint: disable-next=unused-argument
def used_function_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel calls a function of a used module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _USED_FUNCTION_KERNEL,
        extra={"column_select_mod": _COLUMN_SELECT_MODULE,
               "edge_index_mod": _EDGE_INDEX_MODULE})


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


def test_module_inline_refusal_is_reported(external_callee_target):
    """A callee that cannot be brought in is refused with both reasons.

    ``sweep_column`` lives in a module that is on the search path and reads a
    datum that module keeps private, so bringing it into the kernel's
    Container is refused first and inlining it where it stands is refused
    after. Reporting only the second would name the Container the call site is
    in and leave the reader to guess why the callee was not moved into it, so
    the refusal carries both texts: what ``InlineTrans`` said, and what
    ``KernelModuleInlineTrans`` said before it.
    """
    _, loop, _ = external_callee_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    # InlineTrans's own words: the callee's body is in another Container.
    assert "column_solve_kernel_mod" in message
    # And KernelModuleInlineTrans's, which say why it is still there.
    assert "bringing it into the container was refused first" in message
    assert "relaxation" in message


def test_captures_a_kernel_calling_a_used_function(used_function_target):
    """A pure function of a used module is brought in and then inlined.

    This is `face_from_face_selector`'s shape: the callee is a function of a
    second module the kernel names in a ``use``, and that module reads a named
    constant from a third. The symbol at the call site is an unspecialised
    ``Symbol``, because an indexed name in an expression could as easily be an
    array element, and a Call's callee is a routine whether or not the
    frontend could say so.

    Bringing the function into the kernel's Container carries the constant
    with it, so the capture has no import left to follow: ``top_edge`` reaches
    the region the way every module constant does, as a by-value formal the
    PSy layer supplies from the module that declares it, rather than as a name
    the generated C++ has no declaration for.
    """
    psy, loop, kernel = used_function_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    # The callee is gone: its statements are the kernel's own.
    assert "selected_level" not in cpp
    assert "inlined_level = (k + top_edge)" in cpp
    # And the constant it read, two containers away, came with it.
    assert "const int top_edge" in cpp
    generated = str(psy.gen).lower()
    assert "use edge_index_mod, only : top_edge" in generated
    assert "top_edge" in generated.split("column_solve_kokkos(")[1]


def test_callee_is_local_of_a_detached_call():
    """A call outside any Container has no Container holding its callee.

    Every call this mixin is asked about is read from a file and so sits
    inside at least a FileContainer, but the question is asked of the tree
    rather than assumed of it: a detached Call answers no, so that a refusal
    is reported rather than an AttributeError raised from a helper.
    """
    assert not LFRicKokkosTrans._callee_is_local(
        Call.create(RoutineSymbol("sweep_column")))


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


# --------------------------------------------------------------------------
# Task E2: what get_callee should accept. Each kernel below reaches
# InlineTrans through a call whose actual arguments the PSyIR cannot pair
# with the callee's formals by type alone.
# --------------------------------------------------------------------------


# A helper taking a repeat count, called with the literal 2. Fortran gives a
# literal written without a kind the default kind of its type, which is the
# formal's kind wherever the code compiles; the PSyIR records it as UNDEFINED
# and so cannot pair 'Scalar<INTEGER, UNDEFINED>' with 'integer(kind=i_def)'
# for itself. This is the commonest shape in the GungHo corpus.
_LITERAL_ACTUAL_KERNEL = _LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, 2, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, repeat, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n, repeat\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n) * repeat\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# A column of a rank-2 local given to a formal the PSyIR does not fully model
# -- 'target' is enough to make a declaration unsupported, and LFRic's
# helpers carry such attributes -- whose partial datatype is the explicit
# shape 'source(n)'. Comparing the actual's type with that partial type
# compares two shape expressions written in different scopes, which are never
# equal even when they describe the same extent.
_PARTIAL_SECTION_KERNEL = _LOCAL_KERNEL.replace(
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
    "    real(kind=r_def), dimension(n), intent(in), target :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same section given to a formal the PSyIR does model in full. Fortran
# passes an array actual by shape rather than by bounds, so the extents the
# two are written with never have to agree.
_SUBSECTION_ACTUAL_KERNEL = _PARTIAL_SECTION_KERNEL.replace(
    "    call sweep_column(nlayers, column(:,1), swept)\n",
    "    call sweep_column(nlayers, partial(1:nlayers), swept)\n").replace(
    "intent(in), target :: source", "intent(in) :: source")


# The same helper behind a generic interface of two specifics differing in
# the kind of the column they take. The actual states its kind, so one
# specific matches exactly and the other not at all, and the call resolves.
_GENERIC_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_tran").replace(
    "end type column_solve_kernel_type\ncontains\n",
    "end type column_solve_kernel_type\n"
    "  interface sweep_column\n"
    "    module procedure sweep_column_tran, sweep_column_def\n"
    "  end interface sweep_column\n"
    "contains\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column_def(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column_def\n"
    "  subroutine sweep_column_tran(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_tran), dimension(n), intent(in) :: source\n"
    "    real(kind=r_tran), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "  end subroutine sweep_column_tran\n"
    "end module column_solve_kernel_mod")


# The same interface, called with a kindless real literal in the one position
# the two specifics differ in. Fortran resolves this by the default kind of
# the literal; the PSyIR does not know it, so both specifics match equally
# well and neither may be chosen.
_AMBIGUOUS_GENERIC_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_tran").replace(
    "end type column_solve_kernel_type\ncontains\n",
    "end type column_solve_kernel_type\n"
    "  interface scale_column\n"
    "    module procedure scale_column_def, scale_column_tran\n"
    "  end interface scale_column\n"
    "contains\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call scale_column(nlayers, 2.0, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine scale_column_def(n, factor, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), intent(in) :: factor\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n) * factor\n"
    "  end subroutine scale_column_def\n"
    "  subroutine scale_column_tran(n, factor, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_tran), intent(in) :: factor\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n) * factor\n"
    "  end subroutine scale_column_tran\n"
    "end module column_solve_kernel_mod")


@pytest.fixture(name="literal_actual_target")
# pylint: disable-next=unused-argument
def literal_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a kindless literal to a module procedure."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LITERAL_ACTUAL_KERNEL)


@pytest.fixture(name="subsection_actual_target")
# pylint: disable-next=unused-argument
def subsection_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a section to an explicit-shape formal."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SUBSECTION_ACTUAL_KERNEL)


@pytest.fixture(name="partial_section_target")
# pylint: disable-next=unused-argument
def partial_section_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a section to a partially-typed formal."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _PARTIAL_SECTION_KERNEL)


@pytest.fixture(name="generic_kind_target")
# pylint: disable-next=unused-argument
def generic_kind_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling through a two-specific generic interface."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _GENERIC_KIND_KERNEL)


@pytest.fixture(name="ambiguous_generic_target")
# pylint: disable-next=unused-argument
def ambiguous_generic_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose generic call nothing in the PSyIR resolves."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _AMBIGUOUS_GENERIC_KERNEL)


def test_lfric_kokkos_trans_inlines_a_literal_actual(literal_actual_target):
    """A helper called with a literal that states no kind is inlined.

    The kind of a Fortran literal is the formal's kind, which is why the
    code compiles at all; before this the pairing was refused for
    ``Scalar<INTEGER, UNDEFINED>`` not being ``integer(kind=i_def)`` and the
    kernel was not captured.
    """
    _, loop, kernel = literal_actual_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp


def test_lfric_kokkos_trans_inlines_a_section_actual_against_a_shape(
        subsection_actual_target):
    """A section given to an explicit-shape formal is inlined.

    ``partial(1:nlayers)`` is a section of a local column handed to
    ``source(n)``. Fortran passes it by shape rather than by bounds, and the
    pairing is made on rank and intrinsic type for exactly that reason.
    """
    _, loop, kernel = subsection_actual_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp


def test_lfric_kokkos_trans_pairs_a_section_with_a_partial_type(
        partial_section_target):
    """A section reaches a partially-typed formal, and is refused later.

    The formal's declaration carries an attribute the PSyIR does not model,
    so all it has of the formal is a partial datatype -- the explicit shape
    ``source(n)``. Comparing the actual's own shape with that expression
    compares two expressions written in different scopes and can never
    succeed, so the rank-and-intrinsic rule is applied to it instead and the
    callee is resolved.

    Resolving it is as far as this kernel gets: ``InlineTrans`` will not
    inline a routine having an argument whose declaration it does not model,
    whatever the call site passes. So the refusal moves from the argument
    pairing to that rule, which is the one a reader can act on.
    """
    _, loop, _ = partial_section_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "Argument partial type mismatch" not in message
    assert ("Symbol 'source' which is an Argument of UnsupportedType"
            in message)


def test_lfric_kokkos_trans_inlines_through_a_generic_interface(
        generic_kind_target):
    """A generic call whose actual states its kind resolves and is inlined.

    One specific matches exactly and the other not at all, so the scoring
    has one candidate to choose and the ambiguity guard has nothing to say.
    """
    _, loop, kernel = generic_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp


def test_lfric_kokkos_trans_refuses_an_ambiguous_generic_call(
        ambiguous_generic_target):
    """A generic call the relaxed rules cannot decide is refused.

    Both specifics differ from the actual only in the kind of a literal, so
    both match weakly and nothing in the PSyIR says which Fortran would
    choose. Returning the first would make the capture depend on the order
    the interface happens to name its specifics in, so it is refused
    instead.
    """
    _, loop, _ = ambiguous_generic_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'scale_column'" in message
    assert "Ambiguous call to 'scale_column'" in message
    assert "both match with score 1" in message


# --------------------------------------------------------------------------
# Task E7's checks belong here, and are added by the branch that writes them.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Task E8: a callee that calls, or reads, its own module.
# --------------------------------------------------------------------------


def _sibling_invoke(tmp_path, module_source):
    """Build an invoke whose kernel calls a procedure of ``module_source``.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str module_source: the helper module the kernel ``use``s.

    :returns: as :py:func:`lfric_kokkos_sources._invoke` does.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _SIBLING_CALLEE_KERNEL,
                   extra={"sweep_support_mod": module_source,
                          "edge_index_mod": _EDGE_INDEX_MODULE})


def test_callee_calling_its_own_module_is_inlined(
        tmp_path, clear_module_manager_instance):
    """A callee that calls its own module's procedure is still brought in.

    `crosses_panel_edge`'s shape. The callee calls another procedure of its
    own module, which ``KernelModuleInlineTrans`` reads as an access to
    something declared beside it and refuses. It is not data, and the
    Container the two share is where the call is already inlinable, so the
    sibling is absorbed there before the body travels.

    Both bodies end in the region and neither name survives into it. The
    constant the sibling reads is the module's rather than the sibling's, and
    its arriving on the ABI is what says the sibling was carried with its
    scope rather than out of it.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _sibling_invoke(tmp_path, _SIBLING_CALLEE_MODULE)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "damping" not in cpp
    # Twice: an unresolved symbol would have refused the second inlining.
    assert cpp.count("1.0 / ") == 2
    assert "const int top_edge" in cpp
    assert "use edge_index_mod, only : top_edge" in str(psy.gen).lower()


def test_callee_reading_a_module_variable_is_refused(
        tmp_path, clear_module_manager_instance):
    """A callee reading a variable of its own module is refused, and named.

    `chi2xyz`'s shape. Absorbing the sibling settles the call and leaves the
    datum, so the refusal names the datum: a variable of the callee's module
    is neither something the body can carry nor something the PSy layer can
    pass, the module that declares it keeping it private.

    That the message moved from the sibling to the datum is the check: one
    still naming `damping` would send the reader to fix the wrong thing.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _sibling_invoke(tmp_path, _SIBLING_STATE_MODULE)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "bringing it into the container was refused first" in message
    assert ("contains accesses to 'relaxation' which is declared in the "
            "callee module scope") in message
    assert "damping" not in message


def test_a_sibling_that_cannot_be_absorbed_leaves_the_call(
        tmp_path, clear_module_manager_instance):
    """A sibling ``InlineTrans`` refuses is left where the file put it.

    The sibling declares a local with an initialiser, which Fortran gives the
    SAVE attribute and ``InlineTrans`` will not inline. Absorbing it is
    attempted and refused, and that refusal is not what the reader is told:
    what they are told is that the callee still calls the name, which is the
    fact the move was refused for.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _sibling_invoke(tmp_path, _STATIC_SIBLING_MODULE)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert ("contains accesses to 'damping' which is declared in the callee "
            "module scope") in message
