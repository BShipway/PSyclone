# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for a name one scope imports and the other reaches by a wildcard."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall, Routine
from psyclone.psyir.symbols import ContainerSymbol, RoutineSymbol
from psyclone.psyir.transformations import TransformationError


# `reference_element_mod`'s shape: a module of nothing but the four face
# indices, named by LFRic's FFSL kernels and read by the vertical-support
# helpers they call. `s` is the name both scopes hold, and the one the
# refusal this module exists for is about.
_FACE_INDEX_MODULE = """
module face_index_mod
  use constants_mod, only : i_def
  implicit none
  private
  integer(kind=i_def), public, parameter :: w = 1_i_def
  integer(kind=i_def), public, parameter :: s = 2_i_def
  integer(kind=i_def), public, parameter :: e = 3_i_def
  integer(kind=i_def), public, parameter :: n = 4_i_def
end module face_index_mod
"""


# The helper the kernel calls, naming the index in an `only` list.
_NAMED_FACE_HELPER = """
module face_sweep_mod
  use constants_mod, only : i_def, r_def
  use face_index_mod, only : s
  implicit none
  private
  public :: sweep_column
contains
  subroutine sweep_column(levels, source, result)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: source
    real(kind=r_def), dimension(levels), intent(inout) :: result
    integer(kind=i_def) :: j
    result(levels) = source(levels) * real(s, r_def)
    do j = levels - 1, 1, -1
      result(j) = result(j + 1) - source(j) * real(s, r_def)
    end do
  end subroutine sweep_column
end module face_sweep_mod
"""


# The kernel that calls the helper and reads the index itself, through a
# wildcard `use` of the module that declares it. Reading it at the call site
# is what makes the name a clash: a name only one of the two scopes holds is
# merged without a question being asked about it.
_WILDCARD_FACE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use face_index_mod\n"
    "  use face_sweep_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n"
    "    swept(1) = swept(1) + real(s, r_def)\n")


# LFRic's own orientation, which is a kernel and a helper of one module and
# not two: `ffsl_flux_xy_special_edge_code` calls
# `ffsl_flux_xy_special_edge_1d`, written beside it in
# `ffsl_flux_xy_special_edge_kernel_mod`. The module reaches `face_index_mod`
# with a wildcard, so the helper's own table holds `s` unresolved and carries
# no ``use`` to say where it came from; the kernel's own table holds it as an
# import, which is what an earlier inlining leaves there in LFRic and what
# the ``use`` inside the kernel says here. The two tables are what the merge
# compares, and they disagree.
_SIBLING_FACE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use face_index_mod").replace(
    "    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3",
    "    use face_index_mod, only : s\n"
    "    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n"
    "    swept(1) = swept(1) + real(s, r_def)\n").replace(
    "  end subroutine column_solve_code\n",
    "  end subroutine column_solve_code\n"
    "  subroutine sweep_column(levels, source, result)\n"
    "    integer(kind=i_def), intent(in) :: levels\n"
    "    real(kind=r_def), dimension(levels), intent(in) :: source\n"
    "    real(kind=r_def), dimension(levels), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(levels) = source(levels) * real(s, r_def)\n"
    "    do j = levels - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j) * real(s, r_def)\n"
    "    end do\n"
    "  end subroutine sweep_column\n")


# LFRic's shape exactly, and the one the FFSL rows have: the module names
# the index in an `only` list of its own and neither the kernel nor the
# helper beside it holds anything -- both read the one symbol of the module's
# table. `InlineTrans` copies the helper away from that module, and every
# name the copy read from it arrives at the call site unresolved however
# plainly the module declared it, so the disagreement is with a name the
# module resolved perfectly well.
_INHERITED_FACE_KERNEL = _SIBLING_FACE_KERNEL.replace(
    "  use face_index_mod\n", "  use face_index_mod, only : s\n")


# A module whose parameter is a bound rather than a value: `eps_r_tran` of
# LFRic's `constants_mod`, which an FFSL vertical helper trims a section by.
_TRIM_MODULE = """
module trim_index_mod
  use constants_mod, only : i_def
  implicit none
  private
  integer(kind=i_def), public, parameter :: trim_top = 1_i_def
end module trim_index_mod
"""


# The helper that trims a section by that parameter. The module imports the
# name and nothing asks what it is, so the frontend records where it came
# from and leaves it a bare `Symbol`; lowering the section to a loop needs
# its type, and a bare symbol has none.
_TRIMMED_FACE_KERNEL = _INHERITED_FACE_KERNEL.replace(
    "  use kernel_mod, only : kernel_type\n"
    "  use face_index_mod, only : s\n",
    "  use kernel_mod, only : kernel_type\n"
    "  use face_index_mod, only : s\n"
    "  use trim_index_mod, only : trim_top\n").replace(
    "    result(levels) = source(levels) * real(s, r_def)\n",
    "    result(:) = 0.0_r_def\n"
    "    result(1:levels - trim_top) = source(1:levels - trim_top)\n"
    "    result(levels) = source(levels) * real(s, r_def)\n")


# The same kernel and helper reading their index from a module no file
# provides. Nothing can be read from it, so nothing settles the disagreement.
_MISSING_FACE_KERNEL = _SIBLING_FACE_KERNEL.replace(
    "face_index_mod", "missing_face_mod")


def _face_invoke(tmp_path, kernel_source, helper_source=None):
    """Build an invoke whose kernel and helper both read a face index.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str kernel_source: the kernel module the algorithm calls.
    :param helper_source: the module the kernel calls into, where the helper
        is not written beside the kernel.
    :type helper_source: Optional[str]

    :returns: as :py:func:`lfric_kokkos_sources._invoke` does.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    extra = {"face_index_mod": _FACE_INDEX_MODULE,
             "trim_index_mod": _TRIM_MODULE}
    if helper_source is not None:
        extra["face_sweep_mod"] = helper_source
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM, kernel_source,
                   extra=extra)


def test_a_wildcard_at_the_call_site_agrees_with_the_callee(
        tmp_path, clear_module_manager_instance):
    """A name the kernel reaches by a wildcard is the callee's import.

    The kernel `use`s the whole of `face_index_mod` and reads `s` from it, so
    the frontend leaves that name unresolved; the helper names it in an
    `only` list, so the helper's scope has it as an import. Merging the two
    tables is what inlining is, and the merge refuses a name the two scopes
    say different things about.

    They say the same thing about it -- one of them says it less precisely --
    and reading the module settles which. The check is the capture: the
    index reaches the region as the by-value formal every module constant
    reaches it as, with the value the module gives it.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _face_invoke(
        tmp_path, _WILDCARD_FACE_KERNEL, _NAMED_FACE_HELPER)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "const int s" in cpp
    generated = str(psy.gen).lower()
    assert "use face_index_mod, only : s" in generated
    assert "s" in generated.split("column_solve_kokkos(")[1]


def test_a_wildcard_in_the_callee_agrees_with_the_call_site(
        tmp_path, clear_module_manager_instance):
    """The same disagreement the other way round is settled the same way.

    LFRic's own orientation, and the one this task was opened for: the kernel
    holds `s` as an import and the helper written beside it reaches the same
    name through its module's wildcard `use`, so the helper's own table holds
    it unresolved and says nothing about where it came from. That is the pair
    of tables the merge compares, and it refuses them. Which scope is the
    unresolved one is not something the rewrite should turn on, so both
    directions are asked.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _face_invoke(tmp_path, _SIBLING_FACE_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "const int s" in cpp
    assert "use face_index_mod, only : s" in str(psy.gen).lower()


def test_an_index_the_module_imports_reaches_both_of_its_routines(
        tmp_path, clear_module_manager_instance):
    """A name neither scope holds itself is settled where the module holds it.

    This is the shape the FFSL flux kernels have. The module names the index
    in an `only` list, the kernel names it again in one of its own, and the
    sibling helper holds nothing of it and reads the module's symbol. Nothing
    here is a wildcard and nothing is unresolved. What breaks is the copy --
    the helper detached from its module reads a name the module was going to
    supply -- so the helper is given the module's own `use` of it first, and
    the copy then carries a declaration instead of losing one.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _face_invoke(tmp_path, _INHERITED_FACE_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "const int s" in cpp
    assert "use face_index_mod, only : s" in str(psy.gen).lower()


def test_an_imported_bound_is_read_before_the_section_is_lowered(
        tmp_path, clear_module_manager_instance):
    """A section trimmed by an imported parameter is lowered by its value.

    `trim_top` is used for nothing but a bound, so the frontend never had to
    know what it is and left it a bare `Symbol` that says only which module
    it came from. Lowering the section to a loop needs a type, and refuses a
    symbol that has none. The declaration is on the search path, so it is
    read: the parameter reaches the region as the constant it is, and the
    bound is written in terms of it.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _face_invoke(tmp_path, _TRIMMED_FACE_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "const int trim_top" in cpp
    assert "trim_top" in cpp.split("column_solve_kokkos(")[1]


def test_a_clash_no_module_settles_is_still_refused(
        tmp_path, clear_module_manager_instance):
    """A name whose module cannot be read leaves the refusal where it was.

    Both scopes name `missing_face_mod`, which no file provides, so nothing
    can be read from it and nothing is asserted about it: the symbol is left
    unresolved and the reader is given `InlineTrans`'s own words, which name
    the module that was not readable and say what would make it readable. A
    rewrite that guessed here would put a name in the region that no module
    declares.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _face_invoke(tmp_path, _MISSING_FACE_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert ("routine 'sweep_column' contains accesses to 's' which is "
            "unresolved. It is probably brought into scope from one of "
            "['missing_face_mod']") in message


def test_a_call_outside_a_routine_has_no_scopes_to_agree():
    """A call with no routine around it is left alone rather than read.

    Nothing in the transformation reaches `_agree_on_imports` or
    `_read_declarations` with such a call, and the guards are there so that a
    caller which did would get back a tree it recognises rather than an
    `AttributeError` from inside.
    """
    call = Call.create(RoutineSymbol("sweep_column"))

    assert LFRicKokkosTrans._agree_on_imports(call) is None
    assert LFRicKokkosTrans._read_declarations(call) is None
    assert call.parent is None


def test_a_shared_name_the_other_scope_declares_is_left(fortran_reader):
    """Only an import says where a name came from, so only an import is read.

    The other scope may hold the shared name as something of its own -- a
    local, an argument, a routine -- and none of those is evidence about
    where the unresolved one came from. Resolving towards them would be
    guessing, so the name is left as the frontend found it and whatever the
    merge makes of it is the merge's to say.
    """
    container = fortran_reader.psyir_from_source("""
    module face_sweep_mod
      use face_index_mod
      implicit none
    contains
      subroutine reads_the_index(total)
        integer, intent(inout) :: total
        total = total + s
      end subroutine reads_the_index
      subroutine declares_the_index(total)
        integer, intent(inout) :: total
        integer :: s
        s = 2
        total = total + s
      end subroutine declares_the_index
    end module face_sweep_mod
    """).children[0]
    unresolved, declared = container.walk(Routine)

    LFRicKokkosTrans._resolve_shared_names(unresolved,
                                           declared.symbol_table)

    assert unresolved.symbol_table.lookup("s").is_unresolved


def test_a_module_name_the_table_uses_for_something_else_is_left(
        fortran_reader):
    """A module cannot be read through a name that is not a Container.

    A table whose own name for the module is a datum of its own has nowhere
    to put the import and nothing to read the module through, and taking the
    name for the Container would rewrite a declaration the file made. So the
    symbol is left unresolved, which is a refusal the reader is given rather
    than a wrong answer.
    """
    routine = fortran_reader.psyir_from_source("""
    module face_sweep_mod
      use face_index_mod
      implicit none
    contains
      subroutine reads_the_index(total)
        integer, intent(inout) :: total
        integer :: reference_element_mod
        reference_element_mod = 1
        total = total + s + reference_element_mod
      end subroutine reads_the_index
    end module face_sweep_mod
    """).walk(Routine)[0]
    table = routine.symbol_table

    LFRicKokkosTrans._import_by_name(table, table.lookup("s"),
                                     "reference_element_mod")

    assert table.lookup("s").is_unresolved
    assert not isinstance(table.lookup("reference_element_mod"),
                          ContainerSymbol)
