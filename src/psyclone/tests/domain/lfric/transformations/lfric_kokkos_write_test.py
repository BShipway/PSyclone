# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosWriteMixin: what a captured loop may write."""

# pylint: disable=protected-access

import re

import pytest

from lfric_kokkos_sources import _invoke

from psyclone.core import AccessType
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.transformations import TransformationError


_VECTOR_INC_KERNEL = """
module vector_inc_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_inc, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: vector_inc_kernel_type
    type(arg_type) :: meta_args(2) = (/                        &
         arg_type(gh_field*3, gh_real, gh_inc,  w2),           &
         arg_type(gh_field,   gh_real, gh_read, w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: vector_inc_code
  end type vector_inc_kernel_type
contains
  subroutine vector_inc_code(nlayers, acc_1, acc_2, acc_3, src, &
                             ndf_w2, undf_w2, map_w2, &
                             ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    integer(kind=i_def), intent(in) :: ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: acc_1
    real(kind=r_def), dimension(undf_w2), intent(inout) :: acc_2
    real(kind=r_def), dimension(undf_w2), intent(inout) :: acc_3
    real(kind=r_def), dimension(undf_w3), intent(in) :: src
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w2
        acc_1(map_w2(df) + k) = acc_1(map_w2(df) + k) + src(map_w3(1) + k)
        acc_2(map_w2(df) + k) = acc_2(map_w2(df) + k) + src(map_w3(1) + k)
        acc_3(map_w2(df) + k) = acc_3(map_w2(df) + k) + src(map_w3(1) + k)
      end do
    end do
  end subroutine vector_inc_code
end module vector_inc_kernel_mod
"""


_VECTOR_INC_ALGORITHM = """
program kokkos_vector_inc_test
  use field_mod, only : field_type
  use vector_inc_kernel_mod, only : vector_inc_kernel_type
  implicit none
  type(field_type) :: acc(3), src
  call invoke(vector_inc_kernel_type(acc, src))
end program kokkos_vector_inc_test
"""


# The shape of shared write an atomic cannot answer: the shared field takes
# the value of an array-valued intrinsic, which the C writer accumulates in
# a nest of its own. A section alone does not do it -- those are lowered to
# PSyIR loops before the question is asked, and each statement the lowering
# leaves is an ordinary read-modify-write an atomic does carry. Real GungHo
# kernels write this way -- 'nodal_coordinates' and 'strong_curl' among
# them -- and the coloured arm is what carries them.
_WHOLE_ARRAY_INC_KERNEL = """
module whole_inc_probe_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_inc, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: whole_inc_probe_kernel_type
    type(arg_type) :: meta_args(2) = (/                        &
         arg_type(gh_field, gh_real, gh_inc,  w2),             &
         arg_type(gh_field, gh_real, gh_read, w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: whole_inc_probe_code
  end type whole_inc_probe_kernel_type
contains
  subroutine whole_inc_probe_code(nlayers, acc, src, &
                                  ndf_w2, undf_w2, map_w2, &
                                  ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    integer(kind=i_def), intent(in) :: ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: acc
    real(kind=r_def), dimension(undf_w3), intent(in) :: src
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(ndf_w2, ndf_w3) :: weights
    weights(:, :) = 1.0_r_def
    acc(1:ndf_w2) = matmul(weights, src(1:ndf_w3))
  end subroutine whole_inc_probe_code
end module whole_inc_probe_kernel_mod
"""


_WHOLE_ARRAY_INC_ALGORITHM = """
program kokkos_whole_array_inc_test
  use field_mod, only : field_type
  use whole_inc_probe_kernel_mod, only : whole_inc_probe_kernel_type
  implicit none
  type(field_type) :: acc, src
  call invoke(whole_inc_probe_kernel_type(acc, src))
end program kokkos_whole_array_inc_test
"""


@pytest.fixture(name="vector_inc_target")
# pylint: disable-next=unused-argument
def vector_inc_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel accumulates into a shared field vector."""
    return _invoke(
        tmp_path, "vector_inc", _VECTOR_INC_ALGORITHM, _VECTOR_INC_KERNEL)


def test_lfric_kokkos_trans_refuses_a_continuous_write_reading_its_target(
        continuous_read_back_target):
    """A store to a shared dof computed from that dof is still refused.

    The one continuous write the atomic arm has no answer for, and the rule
    is the one the operand of an accumulation has always been held to: the
    read happens before the store and outside it, so making the store
    indivisible leaves the race exactly where it was. Refused rather than
    generated with a note, because the generated source would be wrong and
    would look right.
    """
    _, loop, _ = continuous_read_back_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("replaces an element of the shared field 'flux' with a value "
            "that reads it, which is a race no atomic store answers. Colour "
            "the loop instead." in str(error.value))


def test_lfric_kokkos_trans_permits_a_continuous_read(target):
    """Only the written spaces decide whether cells may run in parallel."""
    _, loop, kernel = target
    kernel.arguments.args[1].function_space._orig_name = "w0"
    LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_an_unmodelled_shared_update(target):
    """A shared field written by something no atomic carries out is refused.

    ``gh_inc`` says two cells contribute to one element; it does not say how,
    and this kernel's body simply assigns to the element. Assignment is not a
    contribution, so an atomic cannot express it and a coloured launch would
    still leave the answer depending on which cell ran last. The refusal is
    raised by validate rather than by the backend, so that a whole-model
    capture leaves the loop as Fortran instead of failing part-way through
    it.

    Monkeypatching the access rather than writing a kernel for it is
    deliberate: what is under test is the pairing of the metadata with the
    body, and no LFRic kernel in the tree carries that pairing -- a real
    ``gh_inc`` kernel accumulates.
    """
    _, loop, kernel = target
    kernel.arguments.args[0]._access = AccessType.INC
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "read-modify-write" in str(err.value)
    assert "Kokkos::atomic_add" in str(err.value)


def test_lfric_kokkos_trans_atomics_reach_a_continuous_gh_inc(
        shared_write_target):
    """The default arm captures a continuous ``gh_inc`` with an atomic.

    The end-to-end claim the two backend tests cannot make: that an LFRic
    loop whose kernel accumulates into a field on a continuous space is
    accepted, that the accumulation is generated as an atomic, and that
    nothing else about the capture changes. The field's own actual is passed
    exactly as any other written field's is -- an atomic is a property of the
    update, not of the interface -- and the write beside it, to a
    discontinuous space no neighbour reaches, stays a plain assignment.
    """
    psy, loop, _ = shared_write_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::atomic_add(&acc(" in cpp
    assert cpp.count("Kokkos::atomic") == 1
    # The View is declared as any other written field's is. An 'Atomic'
    # memory trait would have been the other way to spell this, and is not
    # used: it would make every access to the field atomic, including the
    # plain reads a 'gh_readinc' kernel makes, and it would change the type
    # the region declares rather than the statement that needed changing.
    assert ("Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged> acc(acc_data, undf_w2);" in cpp)
    assert "Kokkos::Atomic" not in cpp

    fortran = str(psy.gen)
    assert "real(c_double), dimension(*), intent(inout) :: acc" in fortran
    assert "call inc_probe_kokkos(" in fortran


def test_lfric_kokkos_trans_accepts_a_continuous_gh_inc(shared_write_target):
    """An accumulation into a shared dof keeps the answer D3 gave it.

    The continuous-write work routes a *store* to a shared dof through the
    same two arms, and the two kinds of sharing are not the same question:
    a contribution has to be combined with the other cell's, and a store has
    only to be indivisible. Reading them as one would generate an
    ``atomic_store`` for an accumulation, which compiles, runs and loses
    every contribution but the last.
    """
    _, loop, _ = shared_write_target

    LFRicKokkosTrans().validate(loop)
    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::atomic_add(&acc(" in cpp
    assert "atomic_store" not in cpp
    assert cpp.count("Kokkos::atomic") == 1


def test_lfric_kokkos_trans_treats_a_continuous_gh_write_as_shared(
        continuous_write_target):
    """A store to a continuous space is a shared write, and takes an atomic.

    LFRic permits ``gh_write`` on a continuous space because the kernel
    author guarantees that every cell reaching a shared dof stores the same
    value to it. That guarantee is not in the metadata, is not in the body,
    and is not checkable here, so the transformation does not rely on it: it
    reads the sharing the function space states and gives the store the
    default arm, which is an atomic. What that buys is narrow and worth being
    exact about -- the element is written whole, so no reader sees a value
    neither cell stored -- and the order of the two stores is still not
    decided by anything. Agreeing values make the order not matter, and it is
    the kernel that promises they agree.

    The write beside it is on W3, which no neighbouring cell reaches, and
    stays the plain assignment stages 5 to 10 generated.
    """
    _, loop, _ = continuous_write_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::atomic_store(&flux(" in cpp
    assert cpp.count("Kokkos::atomic") == 1
    # Nothing in the generated source assumes the two cells agree: the shared
    # field is never the target of a plain assignment, and no comment claims
    # a guarantee the transformation cannot check.
    assert not re.search(r"^\s*flux\(.*\) = ", cpp, re.MULTILINE)
    assert re.search(r"^\s*out\(.*\) = ", cpp, re.MULTILINE)
    # The View is declared as any other written field's is, for the reason
    # the 'gh_inc' one is: the atomic is a property of the statement, not of
    # the interface.
    assert ("Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged> flux(flux_data, undf_w2);" in cpp)
    assert "Kokkos::Atomic" not in cpp


def test_lfric_kokkos_trans_discontinuous_write_is_unchanged(target):
    """A write to a discontinuous space stays the plain assignment it was.

    The path stages 5 to 10 built, asserted here because the continuous-write
    work is the first to make a written space decide how the statement is
    generated. A rule that read 'written field' where it meant 'shared field'
    would put an atomic on every capture in the model, which is a cost paid
    by every loop to answer a question none of them asks: two cells never
    meet at a dof of W3 or of Wtheta.
    """
    _, loop, _ = target

    cpp = LFRicKokkosTrans().apply(loop)

    assert re.search(r"^\s*moist_dyn_gas\(.*\) = ", cpp, re.MULTILINE)
    assert "atomic" not in cpp


def test_lfric_kokkos_trans_refuses_a_non_bool_atomics_option(
        shared_write_target):
    """The option chooses between two arms, so it takes no third value."""
    _, loop, _ = shared_write_target

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop, options={"atomics": "yes"})

    assert ("LFRicKokkosTrans' 'atomics' option must be absent or a bool, "
            "but found 'yes'." in str(err.value))


def test_lfric_kokkos_trans_refuses_neither_answer_to_a_shared_write(
        shared_write_target):
    """Turning the atomics off without colouring first is refused.

    It is the one combination that is not merely redundant: the launch would
    run every cell at once and two of them would read-modify-write the same
    dof, losing a contribution silently and only sometimes.
    """
    _, loop, _ = shared_write_target

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop, options={"atomics": False})

    assert ("'atomics' option is False on a loop that is not coloured, whose "
            "kernel writes a field two cells share" in str(err.value))


def test_lfric_kokkos_trans_atomics_may_be_asked_for_explicitly(
        shared_write_target):
    """Asking for the default arm by name gets the default arm."""
    _, loop, _ = shared_write_target

    cpp = LFRicKokkosTrans().apply(loop, options={"atomics": True})

    assert cpp.count("Kokkos::atomic_add") == 1


def test_lfric_kokkos_trans_accepts_an_asserted_disjoint_write(
        continuous_write_target):
    """A caller may state that the loop's stores reach no shared element.

    The default is conservative and costs an atomic on every store to a
    continuous space, including the many where the dofs a kernel writes are
    its own cell's after all. Nothing here can read that from the metadata,
    so it is the caller's to state -- and the option is named for what is
    being asserted about the kernel rather than for what the transformation
    should do with it, because it is the assertion that has to be true.
    """
    _, loop, _ = continuous_write_target

    cpp = LFRicKokkosTrans().apply(loop, options={"disjoint_writes": True})

    assert re.search(r"^\s*flux\(.*\) = ", cpp, re.MULTILINE)
    assert "atomic" not in cpp
    # The name states the caller's claim about the kernel, not an instruction
    # about the generated source.
    assert LFRicKokkosTrans._DISJOINT_OPTION == "disjoint_writes"


def test_lfric_kokkos_trans_refuses_a_non_bool_disjoint_option(
        continuous_write_target):
    """The assertion is made or not made, so it takes no third value."""
    _, loop, _ = continuous_write_target

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop, options={"disjoint_writes": "yes"})

    assert ("LFRicKokkosTrans' 'disjoint_writes' option must be absent or a "
            "bool, but found 'yes'." in str(err.value))


def test_lfric_kokkos_trans_refuses_a_disjoint_assertion_beside_atomics(
        continuous_write_target):
    """Asking for an answer to the sharing there is said to be none of.

    The two options are read in this order so that the caller is told about
    the assertion they made rather than about what it did to the other
    option, which is the one of the two they can act on.
    """
    _, loop, _ = continuous_write_target

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"disjoint_writes": True, "atomics": True})

    assert ("'disjoint_writes' option is True beside a 'atomics' request" in
            str(err.value))


def test_lfric_kokkos_trans_refuses_a_disjoint_assertion_over_an_accumulation(
        shared_write_target):
    """No assertion about a loop makes 'gh_inc' mean something else.

    The option answers the one kind of sharing the metadata leaves open --
    a store to a continuous space, which may or may not reach a dof another
    cell reaches. An accumulation is not open: LFRic states that two cells
    contribute to one element, and a caller asserting otherwise has
    misunderstood the option rather than described their kernel.
    """
    _, loop, _ = shared_write_target

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop, options={"disjoint_writes": True})

    assert ("'disjoint_writes' option is True on a kernel that accumulates "
            "into 'acc'" in str(err.value))


def test_lfric_kokkos_trans_disjoint_writes_permit_atomics_off(
        continuous_write_target):
    """With the sharing asserted away, asking for no atomic is not a race.

    The refusal that stands between the two options is the one about an
    uncoloured loop whose kernel writes a shared field, and the assertion is
    read where that rule asks which fields those are. Otherwise a caller
    would have to state the same thing twice or not at all.
    """
    _, loop, _ = continuous_write_target

    cpp = LFRicKokkosTrans().apply(
        loop, options={"disjoint_writes": True, "atomics": False})

    assert "atomic" not in cpp


def test_lfric_kokkos_trans_atomics_off_passes_over_a_scalar(second_target):
    """The question of what cells share is asked of every argument.

    A scalar is not a field and no cell of the launch writes it, so the walk
    that answers which arguments are shared passes over it rather than asking
    its function space, which it does not have. The option is then accepted
    on a kernel that shares nothing, which is what says the walk got to the
    end of the argument list.
    """
    _, loop, _ = second_target

    cpp = LFRicKokkosTrans().apply(loop, options={"atomics": False})

    assert "atomic" not in cpp


def test_lfric_kokkos_trans_colour_names_avoid_each_other(target):
    """A generated name steps aside for one generated beside it.

    ``next_available_name`` answers for the symbol table, which holds none of
    the four names the coloured launch declares, so two of them could be
    handed the same answer. They share one C++ scope.
    """
    psy, _, _ = target
    table = psy.invokes.invoke_list[0].schedule.symbol_table
    taken = {"cell"}

    first = LFRicKokkosTrans._clear_name(table, "colour", taken)
    second = LFRicKokkosTrans._clear_name(table, "colour", taken)

    assert first == "colour"
    assert second != first
    assert LFRicKokkosTrans._clear_name(table, "cell", taken) != "cell"


def test_lfric_kokkos_trans_atomics_reach_every_component_of_a_vector(
        vector_inc_target):
    """A ``gh_inc`` field vector shares every one of its components.

    LFRic passes a vector field as one formal per component, and the access
    is declared once for all of them. Reading the access off the first
    component alone would leave the rest as plain assignments -- correct
    Kokkos that loses contributions from every cell but one.
    """
    _, loop, _ = vector_inc_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert cpp.count("Kokkos::atomic_add") == 3
    for component in ("acc_1", "acc_2", "acc_3"):
        assert f"Kokkos::atomic_add(&{component}(" in cpp
    # The field it reads is not shared, and stays a plain load.
    assert "Kokkos::atomic_add(&src(" not in cpp


def test_lfric_kokkos_trans_refuses_a_whole_array_shared_update(
        tmp_path, clear_module_manager_instance):
    # pylint: disable=unused-argument
    """A shared field written whole is refused, and told what does carry it.

    An atomic covers one statement. A shared field given the value of
    ``MATMUL`` is not written by a statement the region holds: the writer
    accumulates it in a nest, and what would have to be made atomic is that
    nest. The refusal names the other arm rather than only saying no,
    because colouring is what captures these kernels.
    """
    _, loop, _ = _invoke(
        tmp_path, "whole_inc_probe", _WHOLE_ARRAY_INC_ALGORITHM,
        _WHOLE_ARRAY_INC_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("updates the shared field 'acc' with a whole-array expression, "
            "which no single atomic carries out. Colour the loop instead."
            in str(error.value))


def test_lfric_kokkos_trans_refuses_a_formal_count_the_metadata_denies(
        shared_write_target, monkeypatch):
    """Which formals are shared is read by position, so counts must agree.

    The metadata says which arguments are shared and the walk that reads it
    gives their positions; the names are the kernel's own formals in the
    same order. If the two lengths differ then no position can be trusted
    and a wrong formal would be declared shared, so the mismatch is refused
    rather than indexed into. LFRic checks the same agreement earlier, which
    is why the disagreement is arranged here instead of written in Fortran.
    """
    _, loop, _ = shared_write_target
    monkeypatch.setattr(
        LFRicKokkosTrans, "_implicit_extents",
        classmethod(lambda cls, table: {"acc": ("acc", 1)}))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert ("cannot say which formals of 'inc_probe_code' are shared between "
            "cells" in message)
    counts = re.search(
        r"declares (\d+) of them and its metadata describes (\d+)\.", message)
    assert counts and int(counts.group(2)) - int(counts.group(1)) == 1


# A kernel that operates on a dof and writes a space this library cannot call
# discontinuous. 'any_space_1' is answered as continuous everywhere else in
# this module, because LFRic states nowhere which continuity such a space will
# have -- so a *cell* kernel writing it has a shared write and takes an
# atomic. The point of this pair is that the same metadata on a *dof* kernel
# does not: the sharing an atomic answers is a property of cell iteration.
# The formals are the fields and the scalar alone, with no nlayers, no ndf,
# no undf and no dofmap, which is the disagreement the formal-count check
# used to report about a kernel it could not describe.
_DOF_SHARED_KERNEL = """
module dof_scale_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, &
                           gh_write, gh_read, any_space_1, dof
  use constants_mod, only : r_def
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: dof_scale_kernel_type
    type(arg_type) :: meta_args(3) = (/                        &
         arg_type(gh_field,  gh_real, gh_write, any_space_1),  &
         arg_type(gh_field,  gh_real, gh_read,  any_space_1),  &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = dof
  contains
    procedure, nopass :: dof_scale_code
  end type dof_scale_kernel_type
contains
  subroutine dof_scale_code(out_dof, in_dof, scale)
    real(kind=r_def), intent(inout) :: out_dof
    real(kind=r_def), intent(in) :: in_dof, scale
    out_dof = scale * in_dof
  end subroutine dof_scale_code
end module dof_scale_kernel_mod
"""


_DOF_SHARED_ALGORITHM = """
program kokkos_dof_shared_test
  use constants_mod, only : r_def
  use field_mod, only : field_type
  use dof_scale_kernel_mod, only : dof_scale_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  real(kind=r_def) :: scale
  call invoke(dof_scale_kernel_type(out_field, in_field, scale))
end program kokkos_dof_shared_test
"""


@pytest.fixture(name="dof_shared_target")
# pylint: disable-next=unused-argument
def dof_shared_target_fixture(tmp_path, clear_module_manager_instance):
    """Create a dof-iterating invoke writing a space that may be continuous.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param clear_module_manager_instance: fixture resetting the module
        manager so that this invoke's kernel is the one resolved.

    :returns: the PSy layer, its first loop and that loop's first kernel.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    return _invoke(
        tmp_path, "dof_scale", _DOF_SHARED_ALGORITHM, _DOF_SHARED_KERNEL)


def test_shared_formals_are_none_for_a_dof_kernel(dof_shared_target):
    """A dof-iterating kernel shares nothing, whatever its spaces say.

    Two cells meet at a dof of a continuous space and both write it; that is
    what makes a cell kernel's store shared. A dof loop visits each dof once
    and writes it once, so there is no second writer to make atomic and no
    colour to separate. The metadata predicates would answer otherwise --
    'any_space_1' is not among the discontinuous names -- and the walk that
    turns their answer into formal positions is a *cell* kernel's argument
    order, so it would then refuse the kernel for a formal count that
    disagrees. The question is the wrong one for this kernel, and the answer
    is that it has no shared formals at all.
    """
    _, loop, kernel = dof_shared_target
    assert loop.iteration_space in ("dof", "owned_dof")
    assert kernel.iterates_over == "dof"

    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._shared_formals(kernel, schedule) == {}
    assert LFRicKokkosTrans._shared_formals(kernel, schedule, True) == {}
    # The rule is not reached at all, rather than reached and satisfied: the
    # loop is accepted where it was refused for a count it cannot meet.
    LFRicKokkosTrans().validate(loop)


def test_dof_kernel_capture_generates_no_atomic(dof_shared_target):
    """The captured dof loop stores plainly and runs the dof launch.

    The two halves of the claim are asserted together because either alone
    would be satisfied by something wrong: a region with no atomic could be a
    cell launch that lost its guard, and a dof launch is only safe without
    one because its index is the dof.
    """
    _, loop, _ = dof_shared_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::atomic_" not in cpp
    assert "Kokkos::RangePolicy<>(0, ndofs)" in cpp
    assert "KOKKOS_LAMBDA(const int df) {" in cpp
    assert re.search(r"^\s*out_dof\(df\) = ", cpp, re.MULTILINE)
    # No colouring either: the other answer to a shared write is as absent as
    # the atomic, and for the same reason.
    assert "cmap" not in cpp


def test_cell_kernel_shared_formals_are_unchanged(shared_write_target):
    """A cell kernel still reports its shared formal.

    The dof answer is an early return, so the test that matters beside it is
    that it is not reached where the question is the right one: this kernel
    accumulates into 'acc' on W2 and the walk still says so, mapped to
    ``False`` because the cells contribute to the element rather than each
    replacing it.
    """
    _, _, kernel = shared_write_target
    assert kernel.iterates_over == "cell_column"

    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._shared_formals(kernel, schedule) == {"acc": False}
