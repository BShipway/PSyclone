# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosIterationMixin: where a captured loop runs."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _HALO_CELL_ALGORITHM, _HALO_CELL_KERNEL, _KERNEL,
    _OWNED_AND_HALO_ALGORITHM, _OWNED_AND_HALO_KERNEL, _invoke)

from psyclone.configuration import Config
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Literal, Reference
from psyclone.psyir.symbols import DataSymbol, ScalarType
from psyclone.psyir.transformations import TransformationError


# An invoke of a builtin, which LFRic writes as a loop over dofs rather than
# over cell columns. With annexed dofs computed it runs to the last annexed
# one, which is the bound twenty-three of the loops the coverage survey
# records under 'halo-depth' carry.
_BUILTIN_ALGORITHM = """
program kokkos_builtin_test
  use field_mod, only : field_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(setval_X(out_field, in_field))
end program kokkos_builtin_test
"""


def test_lfric_kokkos_trans_accepts_a_literal_halo_depth(halo_operator_target):
    """A loop assembling an operator runs to the first halo depth.

    Nothing asks for that bound: LFRic gives it to every loop writing an
    operator, because an operator's columns are needed one cell beyond the
    ones this rank owns. The launch is bounded by the count the loop carried
    rather than by the owned cells, and the count reaches it as the region's
    existing formal filled from the loop's own stop expression -- so the halo
    arithmetic stays where the PSy layer already does it and no part of it
    crosses into the generated C++.
    """
    psy, loop, _ = halo_operator_target
    assert loop.upper_bound_name == "cell_halo"
    assert loop.upper_bound_halo_depth.value == "1"

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # The bound is a formal of the region, and the per-cell dofmap View is
    # sized by that same formal rather than by any owned-cell count.
    assert "const int ncells" in cpp
    assert "map1_data, lfric_kokkos::Role::readonly, ndf1, ncells)" in cpp
    assert "get_last_halo_cell" not in cpp
    assert "loop0_stop = mesh%get_last_halo_cell(1)" in fortran
    assert "loop0_stop)" in fortran.split(
        "call operator_setval_x_kokkos(")[1].split("\n")[0]


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_accepts_an_annexed_bound(
        tmp_path, monkeypatch, clear_module_manager_instance):
    """The last annexed dof is a bound the launch can be given.

    Twenty-three of the loops the coverage survey records under this blocker
    are dof loops running from the first dof to the last annexed one. That is
    a count from the first dof exactly as a halo bound is a count from the
    first cell, so the bound rule accepts it, and the PSy layer fills the
    launch's formal from a different member than a halo bound uses -- the
    field's function space rather than the mesh.

    The loop this is asked of holds a builtin, which is refused by name and
    for a reason of its own: it has no kernel file to capture. That is a rule
    of its own and this test is about the bound, so the separation is
    asserted rather than assumed.

    Whether annexed dofs are computed is a configuration option, and the test
    configuration has it off; the model this prototype targets has it on, so
    the bound is asked for here rather than waited for.
    """
    monkeypatch.setattr(
        Config.get().api_conf("lfric"), "_compute_annexed_dofs", True)
    psy, loop, _ = _invoke(
        tmp_path, "moist_dyn_gas", _BUILTIN_ALGORITHM, _KERNEL)
    assert loop.upper_bound_name == "nannexed"

    LFRicKokkosTrans._validate_halo_depth(loop)

    fortran = str(psy.gen)
    assert ("loop0_stop = out_field_proxy%vspace%get_last_dof_annexed()"
            in fortran)
    assert "get_last_halo_cell" not in fortran

    with pytest.raises(TransformationError, match="LFRic builtin"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_accepts_a_runtime_halo_depth(
        halo_operator_target):
    """A depth computed at run time is evaluated where it already is.

    Six of the loops carrying this blocker take their depth from a variable
    rather than from a literal. The depth is an expression of the PSy layer's
    own symbols, and it stays there: the PSy layer evaluates it into the loop
    bound it already computes, and only that value crosses the ABI. Nothing
    in the generated C++ names the depth, so a region generated for a depth
    of one and a region generated for a depth read at run time differ in
    nothing but what the caller passes.
    """
    psy, loop, _ = halo_operator_target
    depth = psy.invokes.invoke_list[0].schedule.symbol_table.new_symbol(
        "halo_depth", symbol_type=DataSymbol,
        datatype=ScalarType(ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.UNDEFINED))
    loop.set_upper_bound("cell_halo", halo_depth=Reference(depth))

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "loop0_stop = mesh%get_last_halo_cell(halo_depth)" in fortran
    assert "halo_depth" not in cpp
    assert "const int ncells" in cpp


def test_lfric_kokkos_trans_rejects_a_shifted_lower_bound(target):
    """A lower bound stated relative to a depth index is refused by name.

    The launch begins either at zero or at a first cell the region takes as a
    formal, and the PSy layer can fill that formal from the loop's own lower
    bound. What it cannot do is fill it from one of the bounds redundant
    computation produces: those are stated relative to a depth index nothing
    on the ABI carries. A launch beginning at zero over such a loop would run
    the cells it was told to skip -- a wrong answer rather than a compile
    error, which is why the upper bound being one this understands is not on
    its own enough.
    """
    _, loop, _ = target
    loop._lower_bound_name = "inner"
    with pytest.raises(TransformationError, match="'inner' loop lower bound"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_iteration_space_predicate_refuses_a_colouring(
        target):
    """The iteration-space rule is askable on its own.

    The coverage survey reports every blocker a loop carries rather than the
    first, so each rule has to be a predicate of its own. A coloured loop is
    refused for its loop *type* rather than for its space: it runs the cells
    of one colour through a colour map, so a launch from zero to a count
    would run the wrong cells whatever space the map indexes into.
    """
    _, loop, _ = target
    loop._loop_type = "colours"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_iteration_space(loop)

    assert ("LFRicKokkosTrans supports only a loop over cell columns, "
            "coloured or not, or a loop over dofs." in str(error.value))


def test_lfric_kokkos_trans_iteration_space_predicate_accepts_a_dof_loop(
        target):
    """Both dof spaces pass the rule, as all four cell-column spaces do.

    Asked of the predicate directly and space by space, because the survey
    asks it that way: a loop carrying a second blocker still has to report
    this one truthfully.
    """
    _, loop, _ = target

    for space in ("cell_column", "owned_cell_column", "halo_cell_column",
                  "owned_and_halo_cell_column", "dof", "owned_dof"):
        loop._iteration_space = space
        LFRicKokkosTrans._validate_iteration_space(loop)


def test_lfric_kokkos_trans_halo_depth_predicate_accepts_counted_bounds(
        target):
    """Every bound counting from the first cell or dof passes the rule.

    A depth and a depthless halo bound are one question rather than two:
    'cell_halo' with no depth carries no halo depth to find, and a survey
    reporting them apart would count one fact twice. Both are now accepted,
    and so are the three dof bounds, because a launch covers whatever count
    its loop carried.
    """
    _, loop, _ = target
    loop._upper_bound_halo_depth = Literal(
        "1", ScalarType(ScalarType.Intrinsic.INTEGER,
                        ScalarType.Precision.UNDEFINED))

    for bound in ("ncells", "cell_halo", "ndofs", "nannexed", "dof_halo",
                  "ncolour", "colour_halo"):
        loop._upper_bound_name = bound
        LFRicKokkosTrans._validate_halo_depth(loop)


def test_lfric_kokkos_trans_halo_depth_predicate_refuses_an_unknown_bound(
        target):
    """A bound that is not a count from the first cell is refused by name.

    The tiled bounds are the ones this excludes. The plain coloured pair is
    not: ``ncolour`` and ``colour_halo`` count the cells of one colour from
    the first of them, which is the same shape of bound the uncoloured ones
    are, and the region reads which mesh cell each of those is from the
    colour map. A tiled colouring counts something else again and is refused
    for its iteration space too, but the survey asks each rule on its own and
    this one has its own answer.
    """
    _, loop, _ = target
    loop._upper_bound_name = "ntilecolours"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_halo_depth(loop)

    assert ("LFRicKokkosTrans does not support the 'ntilecolours' loop bound."
            in str(error.value))


def test_lfric_kokkos_trans_halo_depth_predicate_refuses_a_shifted_start(
        target):
    """The lower-bound half of the same rule is askable on its own.

    The three bounds refused here are the ones redundant computation
    produces. Each is stated relative to a depth index -- the previous halo
    depth, or the inner region's -- and the region has no formal carrying
    one, so there is no value for the PSy layer to fill the launch's first
    cell from.
    """
    _, loop, _ = target

    for bound in ("inner", "ncells", "cell_halo"):
        loop._lower_bound_name = bound

        with pytest.raises(TransformationError) as error:
            LFRicKokkosTrans._validate_halo_depth(loop)

        assert (f"LFRicKokkosTrans does not support the '{bound}' loop lower "
                "bound." in str(error.value))


def test_lfric_kokkos_trans_halo_depth_predicate_accepts_a_halo_start(
        target):
    """The first halo cell is a lower bound the launch can be given.

    It is the one bound other than the first cell that the PSy layer holds a
    value for, and the launch takes it as a formal of its own beside the
    count.
    """
    _, loop, _ = target
    loop._lower_bound_name = "cell_halo_start"

    LFRicKokkosTrans._validate_halo_depth(loop)


# A kernel that operates on a dof rather than on a cell column. LFRic hands
# such a kernel one dof of each of its fields -- 'call scale_field_code(
# out_field_data(df), in_field_data(df), scale)' -- so there is no dofmap in
# the call, no ndf and no nlayers. wtheta because a dof loop writing a
# continuous space would be refused for the write instead.
_DOF_ALGORITHM = """
program kokkos_dof_test
  use constants_mod, only : r_def
  use field_mod, only : field_type
  use scale_field_kernel_mod, only : scale_field_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  real(kind=r_def) :: scale
  call invoke(scale_field_kernel_type(out_field, in_field, scale))
end program kokkos_dof_test
"""


_DOF_KERNEL = """
module scale_field_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, &
                           gh_write, gh_read, dof
  use constants_mod, only : r_def
  use fs_continuity_mod, only : wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: scale_field_kernel_type
    type(arg_type) :: meta_args(3) = (/                  &
         arg_type(gh_field,  gh_real, gh_write, wtheta), &
         arg_type(gh_field,  gh_real, gh_read,  wtheta), &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = dof
  contains
    procedure, nopass :: scale_field_code
  end type scale_field_kernel_type
contains
  subroutine scale_field_code(out_dof, in_dof, scale)
    real(kind=r_def), intent(inout) :: out_dof
    real(kind=r_def), intent(in) :: in_dof, scale
    out_dof = scale * in_dof
  end subroutine scale_field_code
end module scale_field_kernel_mod
"""


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_captures_a_dof_loop(
        tmp_path, clear_module_manager_instance):
    """A loop over dofs is captured as a flat range over the dof count.

    Twenty-three of the twenty-nine loops the coverage survey records under
    'iteration-space' are dof loops. Each one hands its kernel a single dof
    of every field it takes, so each of those formals crosses the ABI as a
    rank-1 View sliced to the dof count and subscripted by the launch index
    -- which is exactly what the region already does with a per-cell scalar.
    Nothing new crosses the interface: the count the launch is bounded by is
    the same trailing formal a cell launch takes.
    """
    psy, loop, _ = _invoke(
        tmp_path, "scale_field", _DOF_ALGORITHM, _DOF_KERNEL)
    assert loop.iteration_space == "dof"

    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::RangePolicy<>(0, ndofs)" in code
    assert "KOKKOS_LAMBDA(const int df) {" in code
    assert "// One iteration per dof." in code
    assert "out_dof(df) = (scale * in_dof(df));" in code
    # No dofmap and no cell index reaches the region, because the loop had
    # neither. 'cell' itself does appear, in the generated comment saying
    # which of the two this launch is not.
    assert "map_" not in code
    assert "const int cell" not in code
    assert "ncells" not in code

    # The PSy layer fills the count from the field's own function space, and
    # passes each field whole rather than one dof of it.
    fortran = str(psy.gen)
    assert ("loop0_stop = out_field_proxy%vspace%get_last_dof_owned()"
            in fortran)
    assert ("call scale_field_kokkos(out_field_data, in_field_data, scale, "
            "loop0_stop)" in fortran)
    # The count is the only bound that crosses: this loop starts at the first
    # dof, so it takes no first-cell formal and loop0_start goes nowhere.
    assert "loop0_start" not in fortran.split("loop0_start = 1")[1]


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_accepts_an_owned_and_halo_cell_column(
        tmp_path, clear_module_manager_instance):
    """A loop over the owned cells and the halo together needs only the count.

    Four of the twenty-nine loops carrying this blocker run over both. The
    loop still begins at the first cell, so the launch begins at zero, and
    the bound it runs to is the trailing count formal the launch already
    takes -- filled from the mesh's last halo cell rather than its last owned
    one. This is the case the iteration-space rule was refusing for no reason
    of its own.
    """
    psy, loop, _ = _invoke(
        tmp_path, "wide_write", _OWNED_AND_HALO_ALGORITHM,
        _OWNED_AND_HALO_KERNEL)
    assert loop.iteration_space == "owned_and_halo_cell_column"

    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::RangePolicy<>(0, ncells)" in code
    assert "KOKKOS_LAMBDA(const int cell) {" in code
    assert "first_cell" not in code

    fortran = str(psy.gen)
    assert "loop0_stop = mesh%get_last_halo_cell(hdepth)" in fortran


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_accepts_a_halo_cell_column(
        tmp_path, clear_module_manager_instance):
    """A loop over the halo alone hands the launch its first cell.

    Two of the twenty-nine loops carrying this blocker skip the owned cells.
    That is the one shape the launch cannot express with a count alone, so
    the region takes a second scalar formal beside the count. It carries a
    value and not an expression, as the count does: the PSy layer already
    computes the loop's Fortran lower bound, and the conversion to the
    launch's zero-based index is the subtraction of one, done there.
    """
    psy, loop, _ = _invoke(
        tmp_path, "halo_write", _HALO_CELL_ALGORITHM, _HALO_CELL_KERNEL)
    assert loop.iteration_space == "halo_cell_column"

    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::RangePolicy<>(first_cell, ncells)" in code
    assert "const int first_cell" in code

    fortran = str(psy.gen)
    assert "loop0_start = mesh%get_last_edge_cell() + 1" in fortran
    assert "loop0_stop = mesh%get_last_halo_cell(hdepth)" in fortran
    assert "loop0_stop, loop0_start - 1)" in fortran


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_refuses_a_builtin(
        tmp_path, clear_module_manager_instance):
    """A builtin is refused by name, not by its iteration space.

    LFRic writes a builtin as a loop over dofs, so widening the
    iteration-space rule to admit dof loops reaches them. It must not: a
    builtin has no kernel file and no kernel schedule to capture -- PSyclone
    lowers it into the PSy layer itself -- so every rule below this one is
    asked of something that is not there. Refusing it by name is what keeps
    that from surfacing as an AttributeError, which the coverage survey would
    record as an error row and a whole-model capture would crash on.

    The survey excludes builtins from the catalogue by design, so no
    catalogue row turns on this refusal.
    """
    _, loop, _ = _invoke(
        tmp_path, "moist_dyn_gas", _BUILTIN_ALGORITHM, _KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the LFRic builtin "
            "'setval_x'." in str(error.value))


def test_lfric_kokkos_trans_refuses_an_unmodelled_iteration_space(target):
    """An iteration space the launch has no shape for is refused by name.

    The rule names the space rather than listing the four it admits, because
    the survey reports one blocker per loop and the name is what tells two
    unadmitted spaces apart. 'domain' is the real one this excludes: a
    kernel operating on the whole domain is called once, with no loop for a
    launch to become.
    """
    _, loop, _ = target
    loop._iteration_space = "domain"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_iteration_space(loop)

    assert ("LFRicKokkosTrans does not support the 'domain' iteration space."
            in str(error.value))


def test_lfric_kokkos_trans_refuses_an_array_local_of_a_dof_kernel(tmp_path):
    """A dof launch has nowhere to put a kernel-local array.

    A cell-column launch places one in team scratch and takes the team
    launch to do it. A dof launch is a flat range with no team at all, so
    the array is refused by name rather than dropped by a shape that has no
    scratch to carry it -- which would compile and give a wrong answer.
    """
    kernel = _DOF_KERNEL.replace(
        "    real(kind=r_def), intent(in) :: in_dof, scale\n",
        "    real(kind=r_def), intent(in) :: in_dof, scale\n"
        "    real(kind=r_def), dimension(3) :: partial\n")
    kernel = kernel.replace(
        "    out_dof = scale * in_dof\n",
        "    partial(1) = scale * in_dof\n"
        "    out_dof = partial(1)\n")
    _, loop, _ = _invoke(tmp_path, "scale_field", _DOF_ALGORITHM, kernel)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans cannot place the kernel-local array 'partial' "
            "of a loop over dofs: the dof launch is a flat range and has no "
            "team to hold scratch." in str(error.value))


def test_lfric_kokkos_trans_refuses_a_spreadable_loop_of_a_dof_kernel(
        tmp_path):
    """A dof launch has no members to spread a loop over.

    Asked of the predicate directly with the loops it would have been given,
    because a kernel over dofs holding a loop the dependence analysis
    accepts would have to hold an array for it to write, and that is refused
    by the rule above before this one is reached.
    """
    _, loop, kernel = _invoke(
        tmp_path, "scale_field", _DOF_ALGORITHM, _DOF_KERNEL)
    schedule = kernel.get_callees()[0]

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_dof_body(loop, schedule, (loop,))

    assert ("LFRicKokkosTrans cannot spread a loop of a kernel over dofs "
            "across a team: the dof launch is a flat range and has no team."
            in str(error.value))


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_dof_capture_takes_no_atomic_even_when_asked(
        tmp_path, clear_module_manager_instance):
    """The dof arm generates no atomic, with the option on or off.

    A dof launch's freedom from write conflicts is a property of its
    iteration space: one iteration writes one dof, and no two iterations
    write the same one. There is therefore nothing for the atomic arm to
    make safe, and asking for it changes nothing -- which is asserted rather
    than left to be inferred from a kernel that happens to have no shared
    write, because the option is the caller's way of asking for the arm and
    an arm that fired here would be generating a lock on unshared data.
    """
    _, loop, _ = _invoke(
        tmp_path, "scale_field", _DOF_ALGORITHM, _DOF_KERNEL)

    plain = LFRicKokkosTrans().apply(loop, options={"atomics": True})

    assert "Kokkos::RangePolicy<>(0, ndofs)" in plain
    assert "KOKKOS_LAMBDA(const int df) {" in plain
    assert "Kokkos::atomic" not in plain
    assert "cmap" not in plain
