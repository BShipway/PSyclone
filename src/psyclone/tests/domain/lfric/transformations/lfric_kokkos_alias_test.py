# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosAliasMixin: a pointer local that aliases an array."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Literal, Routine)
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType
from psyclone.psyir.transformations import TransformationError


# The shape LFRic's FFSL vertical-support helpers have, reduced to the two
# things that matter: a helper declaring a whole-array pointer local, aiming
# it at one of two whole arrays by a flag, and then reading it. The caller's
# two arrays are its own locals, so both targets become scratch and the alias
# has to take that space's type rather than the argument Views'.
#
# Written out here rather than in `lfric_kokkos_sources`, which is within
# fifteen lines of the thousand pylint's C0302 fires at.
_ALIAS_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "    real(kind=r_def), dimension(nlayers) :: swept\n",
    "    real(kind=r_def), dimension(nlayers) :: swept\n"
    "    real(kind=r_def), dimension(nlayers) :: spare\n"
    "    logical(kind=l_def) :: pick_spare\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    do k = 1, nlayers\n"
    "      spare(k) = partial(k) * 2.0_r_def\n"
    "    end do\n"
    "    pick_spare = nlayers > 3\n"
    "    call sweep_column(nlayers, partial, spare, pick_spare, swept)\n"
    ).replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, other, pick_other, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), target, intent(in) :: source(n)\n"
    "    real(kind=r_def), target, intent(in) :: other(n)\n"
    "    logical(kind=l_def), intent(in) :: pick_other\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    real(kind=r_def), pointer :: chosen(:)\n"
    "    integer(kind=i_def) :: j\n"
    "    if (pick_other) then\n"
    "      chosen => other\n"
    "    else\n"
    "      chosen => source\n"
    "    end if\n"
    "    result(n) = chosen(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - abs(chosen(j))\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same helper aiming the pointer at a section of one target. A section is
# part of an array rather than the array, and a View handle copy does not say
# that, so it is refused rather than generated as one.
_SECTION_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "      chosen => other\n", "      chosen => other(1:n)\n")


# The same helper aiming the pointer at an expression. Nothing owns the
# result of `source + other`, so there is nothing for a handle to be a second
# name for.
_EXPRESSION_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "      chosen => other\n", "      chosen => other + source\n")


# The same helper aiming the pointer nowhere. The fparser2 frontend builds no
# IntrinsicCall for `nullify`, so this arrives as a CodeBlock and is refused
# as one: a statement PSyclone did not model may aim the pointer anywhere.
_NULLIFY_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    result(n) = chosen(n)\n",
    "    nullify(chosen)\n"
    "    result(n) = source(n)\n")


# The same helper asking the pointer whether it is aimed anywhere. A Kokkos
# View has no answer to that question, so the helper is refused by name.
_ASSOCIATED_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    result(n) = chosen(n)\n",
    "    if (associated(chosen)) then\n"
    "      result(n) = chosen(n)\n"
    "    end if\n")


# The same helper handing the pointer to a second routine. What that routine
# does with it -- aim it elsewhere, ask whether it is associated -- is beyond
# what this can read, so the pointer is refused rather than assumed harmless.
_ACTUAL_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    result(n) = chosen(n)\n",
    "    call seed_column(n, chosen, result)\n").replace(
    "  end subroutine sweep_column\n",
    "  end subroutine sweep_column\n"
    "  subroutine seed_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine seed_column\n")


# The same helper choosing between a kernel argument and a kernel-local
# array. This is the shape LFRic's vertical-support kernels have -- a field
# passed in or a column worked out on the way -- and it is the one shape of
# the pattern this refuses: the two are Views of different Kokkos spaces.
_MIXED_SPACE_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    call sweep_column(nlayers, partial, spare, pick_spare, swept)\n",
    "    call sweep_column(nlayers, field_in, spare, pick_spare, swept)\n")


# The same helper aiming one pointer at arrays of two different ranks. One
# View handle holds one rank, so this is refused rather than generated as a
# copy the C++ compiler would reject a long way from its cause.
_RANK_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    real(kind=r_def), target, intent(in) :: other(n)\n",
    "    real(kind=r_def), target, intent(in) :: other(n, 2)\n").replace(
    "    call sweep_column(nlayers, partial, spare, pick_spare, swept)\n",
    "    call sweep_column(nlayers, partial, pair, pick_spare, swept)\n"
    ).replace(
    "    real(kind=r_def), dimension(nlayers) :: spare\n",
    "    real(kind=r_def), dimension(nlayers) :: spare\n"
    "    real(kind=r_def), dimension(nlayers, 2) :: pair\n")


# The same helper aiming the pointer at a local of its own declared TARGET.
# `InlineTrans` refuses a callee holding a local of a type it cannot model,
# and a TARGET local is one, so this shape only inlines because the alias
# relaxes the target as well as the pointer.
_TARGET_LOCAL_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    real(kind=r_def), pointer :: chosen(:)\n",
    "    real(kind=r_def), pointer :: chosen(:)\n"
    "    real(kind=r_def), target :: buffer(n)\n").replace(
    "      chosen => other\n",
    "      buffer(:) = other(:)\n"
    "      chosen => buffer\n")


# The same helper declaring a second pointer and aiming it nowhere. A pointer
# no statement aims is not an alias of anything, and a handle declared from a
# target that does not exist is not a thing this could write.
_IDLE_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    real(kind=r_def), pointer :: chosen(:)\n",
    "    real(kind=r_def), pointer :: chosen(:)\n"
    "    real(kind=r_def), pointer :: idle(:)\n")


# The same helper aiming a scalar pointer. There is no array type to read
# from the declaration, so the refusal is the one about the declaration
# rather than one about what the pointer is aimed at.
_SCALAR_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    real(kind=r_def), pointer :: chosen(:)\n",
    "    real(kind=r_def), pointer :: chosen(:)\n"
    "    real(kind=r_def), pointer :: knob\n"
    "    real(kind=r_def), target :: base\n").replace(
    "    result(n) = chosen(n)\n",
    "    base = 1.0_r_def\n"
    "    knob => base\n"
    "    result(n) = chosen(n) * knob\n")


# The same helper aiming its pointer at a name it imports. Nothing here can
# read what that name is declared as, so nothing here can say the handle and
# the target agree.
_IMPORTED_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    integer(kind=i_def) :: j\n",
    "    integer(kind=i_def) :: j\n").replace(
    "      chosen => other\n",
    "      chosen => reference_column\n").replace(
    "  subroutine sweep_column(n, source, other, pick_other, result)\n",
    "  subroutine sweep_column(n, source, other, pick_other, result)\n"
    "    use reference_column_mod, only : reference_column\n")


# The same helper remapping the pointer's bounds as it aims it. fparser2
# builds no Assignment for a remapping, so what arrives is a CodeBlock and
# the refusal is the one about a statement PSyclone did not model.
_SUBSCRIPT_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "      chosen => other\n", "      chosen(1:n) => other\n")


# The same helper aiming its pointer at an ALLOCATABLE local. The alias
# relaxes a pointer and a TARGET target and no other unmodelled attribute, so
# the allocatable stays what it was and is refused where a deferred-shape
# local is refused. That the pointer aimed at it changes nothing is the point.
_ALLOCATABLE_ALIAS_KERNEL = _ALIAS_KERNEL.replace(
    "    real(kind=r_def), pointer :: chosen(:)\n",
    "    real(kind=r_def), pointer :: chosen(:)\n"
    "    real(kind=r_def), allocatable :: stored(:)\n").replace(
    "      chosen => other\n", "      chosen => stored\n")


@pytest.fixture(name="alias_target")
# pylint: disable-next=unused-argument
def alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling a helper that aliases one of two arrays."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ALIAS_KERNEL)


@pytest.fixture(name="section_alias_target")
# pylint: disable-next=unused-argument
def section_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aims its pointer at a section."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SECTION_ALIAS_KERNEL)


@pytest.fixture(name="expression_alias_target")
# pylint: disable-next=unused-argument
def expression_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aims its pointer at an expression."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _EXPRESSION_ALIAS_KERNEL)


@pytest.fixture(name="nullify_alias_target")
# pylint: disable-next=unused-argument
def nullify_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper nullifies its pointer."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _NULLIFY_ALIAS_KERNEL)


@pytest.fixture(name="associated_alias_target")
# pylint: disable-next=unused-argument
def associated_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper asks whether its pointer is aimed."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ASSOCIATED_ALIAS_KERNEL)


@pytest.fixture(name="actual_alias_target")
# pylint: disable-next=unused-argument
def actual_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper passes its pointer to a routine."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ACTUAL_ALIAS_KERNEL)


@pytest.fixture(name="mixed_space_alias_target")
# pylint: disable-next=unused-argument
def mixed_space_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aliases an argument and a local."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MIXED_SPACE_ALIAS_KERNEL)


@pytest.fixture(name="rank_alias_target")
# pylint: disable-next=unused-argument
def rank_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aliases two ranks with one pointer."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RANK_ALIAS_KERNEL)


@pytest.fixture(name="target_local_alias_target")
# pylint: disable-next=unused-argument
def target_local_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aliases a TARGET local of its own."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _TARGET_LOCAL_ALIAS_KERNEL)


@pytest.fixture(name="idle_alias_target")
# pylint: disable-next=unused-argument
def idle_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper declares a pointer it never aims."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _IDLE_ALIAS_KERNEL)


@pytest.fixture(name="scalar_alias_target")
# pylint: disable-next=unused-argument
def scalar_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aims a scalar pointer."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SCALAR_ALIAS_KERNEL)


@pytest.fixture(name="imported_alias_target")
# pylint: disable-next=unused-argument
def imported_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aims its pointer at an import."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _IMPORTED_ALIAS_KERNEL)


@pytest.fixture(name="allocatable_alias_target")
# pylint: disable-next=unused-argument
def allocatable_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper aims its pointer at an allocatable."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _ALLOCATABLE_ALIAS_KERNEL)


@pytest.fixture(name="subscript_alias_target")
# pylint: disable-next=unused-argument
def subscript_alias_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper remaps its pointer's bounds."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SUBSCRIPT_ALIAS_KERNEL)


# ---------------------------------------------------------------------------
# Task E10: a local pointer that aliases an array.
# ---------------------------------------------------------------------------


def test_alias_pointer_is_captured_as_a_view_handle(alias_target):
    """A helper choosing between two whole arrays by a flag is captured.

    The three things the generated region has to say are asserted here and
    not one of them is implied by the others: the handle is declared with its
    target's type, each branch of the choice assigns a handle rather than
    copying elements, and the reads through the pointer are subscripts of the
    alias exactly as the Fortran wrote them.
    """
    _, loop, _ = alias_target

    cpp = LFRicKokkosTrans().apply(loop)

    # Declared from the first target the body aims it at, which is the one
    # the leading branch of the choice names.
    assert "decltype(spare) chosen;" in cpp
    assert "chosen = spare;" in cpp
    assert "chosen = partial;" in cpp
    # Read through the alias, with the target's Fortran origin applied.
    assert "swept((nlayers - 1)) = chosen((nlayers - 1));" in cpp
    # A handle copy, not an element-by-element copy of the target.
    assert "for" not in cpp.split("chosen = spare;")[0].split(
        "if (pick_spare)")[-1]
    assert "sweep_column" not in cpp


def test_alias_pointer_takes_no_scratch_of_its_own(alias_target):
    """The alias is a handle, so the launch reserves nothing for it.

    A pointer re-declared as an array of deferred shape looks like an
    automatic array to everything that walks the symbol table, and reserving
    scratch for one would both fail to size -- it has no extent -- and
    reserve a column no read ever reaches.
    """
    _, loop, _ = alias_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "chosen_scratch_t" not in cpp
    assert "double * restrict chosen" not in cpp


def test_alias_pointer_to_a_section_is_refused(section_alias_target):
    """A pointer aimed at part of an array is refused by name."""
    _, loop, _ = section_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "a section or an expression rather than at a whole array" in message


def test_alias_pointer_to_an_expression_is_refused(expression_alias_target):
    """A pointer aimed at a value nothing owns is refused by name."""
    _, loop, _ = expression_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("a section or an expression rather than at a whole array"
            in str(error.value))


def test_nullified_alias_pointer_is_refused(nullify_alias_target):
    """A pointer named by a statement PSyclone cannot read is refused.

    `nullify` is the statement in reach here, and the refusal is the general
    one rather than one naming it: the frontend builds no IntrinsicCall for
    it, so what arrives is a CodeBlock like any other.
    """
    _, loop, _ = nullify_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "a statement PSyclone does not model names it" in message


def test_associated_alias_pointer_is_refused(associated_alias_target):
    """A pointer asked whether it is aimed anywhere is refused by name."""
    _, loop, _ = associated_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "'associated' is called on it" in message


def test_alias_pointer_passed_as_an_actual_is_refused(actual_alias_target):
    """A pointer handed to another routine is refused by name."""
    _, loop, _ = actual_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "it is passed as an actual argument" in message


def test_alias_pointer_across_two_spaces_is_refused(mixed_space_alias_target):
    """A pointer aimed at an argument and at a local is refused by name.

    Refused at the transformation rather than left to the writer: both are
    arrays of one element type and rank, so nothing about the description
    says they cannot share a handle, and the C++ that results fails to
    compile a long way from the pointer that caused it.
    """
    _, loop, _ = mixed_space_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen'" in message
    assert ("aimed both at a kernel argument and at a kernel-local array"
            in message)


def test_alias_pointer_of_two_ranks_is_refused(rank_alias_target):
    """One pointer aimed at a vector and at a matrix is refused by name."""
    _, loop, _ = rank_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "differ in intrinsic, kind or rank" in message


def test_alias_pointer_relaxes_a_target_local(target_local_alias_target):
    """A TARGET local a pointer is aimed at is relaxed with the pointer.

    Two rewrites, and the capture needs both: without the pointer's, the
    callee holds a symbol `InlineTrans` will not place; without the target's,
    it holds a second one, and the helper is refused for the local rather
    than for the pointer that named it.
    """
    _, loop, _ = target_local_alias_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "decltype(buffer) chosen;" in cpp
    assert "chosen = buffer;" in cpp
    assert "chosen = partial;" in cpp
    assert "sweep_column" not in cpp


def test_unaimed_alias_pointer_is_refused(idle_alias_target):
    """A pointer no statement aims at an array is refused by name."""
    _, loop, _ = idle_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'idle' of 'sweep_column'" in message
    assert "nothing in the routine aims it at an array" in message


def test_scalar_alias_pointer_is_refused(scalar_alias_target):
    """A pointer whose declaration holds no array type is refused by name."""
    _, loop, _ = scalar_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'knob' of 'sweep_column'" in message
    assert "not one an array type could be read from" in message


def test_alias_pointer_to_an_import_is_refused(imported_alias_target):
    """A pointer aimed at a name this cannot read is refused by name."""
    _, loop, _ = imported_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "aimed at something that is not a declared array" in message


def test_remapped_alias_pointer_is_refused(subscript_alias_target):
    """A pointer whose bounds are remapped as it is aimed is refused.

    Not by the clause about a subscript, which is what the shape looks like:
    fparser2 builds no Assignment for a bounds remapping at all, so what the
    walk finds is a CodeBlock and the refusal is the general one.
    """
    _, loop, _ = subscript_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer 'chosen' of 'sweep_column'" in message
    assert "a statement PSyclone does not model names it" in message


def test_subscripted_alias_assignment_is_refused():
    """A pointer assignment through a subscript is refused by name.

    Reached directly, because no Fortran this frontend reads produces one: a
    bounds remapping arrives as a CodeBlock. The clause is kept because the
    check is a type test on a node the rest of the walk trusts, and a tree
    built by hand -- or by a later frontend -- can hold what this refuses.
    """
    routine = FortranReader().psyir_from_source(
        "subroutine sweep(x)\n"
        "  real, target, intent(inout) :: x(3)\n"
        "  real, pointer :: p(:)\n"
        "  p => x\n"
        "end subroutine sweep\n").walk(Routine)[0]
    symbol = routine.symbol_table.lookup("p")
    assignment = routine.walk(Assignment)[0]
    integer = ScalarType(ScalarType.Intrinsic.INTEGER,
                         ScalarType.Precision.UNDEFINED)
    real = ScalarType(ScalarType.Intrinsic.REAL,
                      ScalarType.Precision.UNDEFINED)
    assignment.lhs.replace_with(ArrayReference.create(
        DataSymbol("p", ArrayType(real, [3])), [Literal("1", integer)]))

    refusal = LFRicKokkosTrans._alias_pointer_refusal(
        routine, symbol, [assignment])

    assert refusal == "it is aimed at through a subscript"


def test_alias_pointer_to_an_allocatable_is_refused(allocatable_alias_target):
    """An ALLOCATABLE the pointer names is refused as a deferred shape.

    The alias relaxes a pointer and a TARGET target and nothing else. An
    allocatable is storage whose size is decided at run time, which is not
    something a View handle or a scratch array names, so it is refused by
    the contract's own words about a deferred shape -- for the local, not
    for the pointer aimed at it, and before the alias rewrite is reached.
    """
    _, loop, _ = allocatable_alias_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot capture the pointer" not in message
    assert "stored" in message
    assert "declared with a deferred shape" in message
