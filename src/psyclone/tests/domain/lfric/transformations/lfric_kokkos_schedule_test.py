# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosScheduleMixin: selecting the schedule."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.transformations import TransformationError


# A kind-polymorphic kernel: one metadata name over several implementations
# that differ only in the precision of their real arguments. Built from a
# template rather than written out three times because the three fixtures below
# differ only in which two kinds the interface carries and which kind the
# algorithm passes, and that difference is the whole point of each test.
#
# The layout is sci_tri_solve_kernel_mod's: a public generic interface beside
# the metadata type, with the specific procedures named for their kind.
_POLYMORPHIC_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, field_out, field_in, scaling, &
                                ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind={kind}), dimension(undf_w3), intent(inout) :: field_out
    real(kind={kind}), dimension(undf_w3), intent(in) :: field_in
    real(kind={kind}), intent(in) :: scaling
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = scaling * field_in(map_w3(df) + k)
      end do
    end do
  end subroutine {name}_code_{kind}
"""


def _polymorphic_kernel(name, first, second, stencil=False):
    """Build a kernel module whose code is an interface over two kinds.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.
    :param bool stencil: whether the read field carries a cross stencil. It
        changes nothing about the precisions; it puts the metadata outside
        what ``KernelInterface`` builds, which is a separate refusal.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_MEMBER.format(name=name, kind=kind)
        for kind in (first, second))
    imports = "gh_read, cell_column"
    read_arg = "arg_type(gh_field,  gh_real, gh_read,  w3)"
    if stencil:
        imports = "gh_read, cell_column, stencil, cross"
        read_arg = "arg_type(gh_field,  gh_real, gh_read,  w3, stencil(cross))"
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, gh_write, &
                           {imports}
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver, r_quad
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real, gh_write, w3),                 &
         {read_arg},                 &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


# A kind-polymorphic kernel that also asks for an evaluator. It is the shape
# GungHo's sample_field_kernel_mod has: the basis is tabulated at the nodal
# points of the written field's space and stays r_def whatever the fields do,
# because the PSy layer computes it once at the model's own working precision.
_POLYMORPHIC_EVALUATOR_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, field_out, field_in,           &
                                ndf_{target}, undf_{target}, map_{target},  &
                                ndf_{space}, undf_{space}, map_{space},     &
                                basis_{space}_on_{target})
    integer(kind=i_def), intent(in) :: nlayers
    integer(kind=i_def), intent(in) :: ndf_{target}, undf_{target}
    integer(kind=i_def), intent(in) :: ndf_{space}, undf_{space}
    integer(kind=i_def), dimension(ndf_{target}), intent(in) :: map_{target}
    integer(kind=i_def), dimension(ndf_{space}), intent(in) :: map_{space}
    real(kind={kind}), dimension(undf_{target}), intent(inout) :: field_out
    real(kind={kind}), dimension(undf_{space}), intent(in) :: field_in
    real(kind=r_def), dimension({dim},ndf_{space},ndf_{target}),           &
                                     intent(in) :: basis_{space}_on_{target}
    integer(kind=i_def) :: k, df, dg
    real(kind={kind}) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_{target}
        total = 0.0_{kind}
        do dg = 1, ndf_{space}
          total = total + basis_{space}_on_{target}({dim},dg,df)          &
                * field_in(map_{space}(dg) + k)
        end do
        field_out(map_{target}(df) + k) = total
      end do
    end do
  end subroutine {name}_code_{kind}
"""


def _polymorphic_evaluator_kernel(name, first, second, space="w1",
                                  target="w3", dim=3):
    """Build a kernel module whose evaluator code is an interface over kinds.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.
    :param str space: the function space the basis is asked for on.
    :param str target: the function space of the written field, which is what
        an evaluator tabulates at.
    :param int dim: the basis's first extent, which metadata fixes for a
        named space and leaves unknown for an ``any_space``.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_EVALUATOR_MEMBER.format(
            name=name, kind=kind, space=space, target=target, dim=dim)
        for kind in (first, second))
    # An any_space is named by argument_mod; a named function space by
    # fs_continuity_mod.
    generic = [fs for fs in (space, target) if fs.startswith("any_")]
    named = [fs for fs in (space, target) if not fs.startswith("any_")]
    generic_import = ", " + ", ".join(generic) if generic else ""
    named_import = (f"  use fs_continuity_mod, only : {', '.join(named)}\n"
                    if named else "")
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real,        &
                           gh_write, gh_read, gh_basis, cell_column,      &
                           gh_evaluator{generic_import}
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver
{named_import}  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(2) = (/                                  &
         arg_type(gh_field, gh_real, gh_write, {target}),                 &
         arg_type(gh_field, gh_real, gh_read,  {space}) /)
    type(func_type) :: meta_funcs(1) = (/                                &
         func_type({space}, gh_basis) /)
    integer :: gh_shape = gh_evaluator
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


def _polymorphic_evaluator_algorithm(name, field_module, field_type):
    """Build an algorithm invoking one polymorphic evaluator kernel.

    An evaluator carries no quadrature object, so the invoke passes nothing
    but the two fields; their precision is the whole of what selects a member.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.

    :returns: Fortran source for the program.
    :rtype: str
    """
    return f"""
program kokkos_{name}_test
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: out_field, in_field
  call invoke({name}_kernel_type(out_field, in_field))
end program kokkos_{name}_test
"""


# A kind-polymorphic inter-grid kernel, the shape the model's
# sci_restrict_scalar_unweighted_kernel_mod has: one restriction written twice
# over, once per real kind, behind a generic interface.
_POLYMORPHIC_INTERGRID_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, cell_map, ncell_f_per_c_x,      &
                                ncell_f_per_c_y, ncell_f, coarse_field,  &
                                fine_field, undf_c, map_c, ndf, undf_f,  &
                                map_f)
    integer(kind=i_def), intent(in) :: nlayers, ncell_f_per_c_x
    integer(kind=i_def), intent(in) :: ncell_f_per_c_y, ncell_f
    integer(kind=i_def), intent(in) :: ndf, undf_f, undf_c
    integer(kind=i_def), dimension(ncell_f_per_c_x, ncell_f_per_c_y), &
        intent(in) :: cell_map
    real(kind={kind}), dimension(undf_c), intent(inout) :: coarse_field
    real(kind={kind}), dimension(undf_f), intent(in) :: fine_field
    integer(kind=i_def), dimension(ndf), intent(in) :: map_c
    integer(kind=i_def), dimension(ndf, ncell_f), intent(in) :: map_f
    integer(kind=i_def) :: k, df, x_idx, y_idx, fine_cell
    do y_idx = 1, ncell_f_per_c_y
      do x_idx = 1, ncell_f_per_c_x
        fine_cell = cell_map(x_idx, y_idx)
        do k = 0, nlayers - 1
          do df = 1, ndf
            coarse_field(map_c(df) + k) = coarse_field(map_c(df) + k) &
                + fine_field(map_f(df, fine_cell) + k)
          end do
        end do
      end do
    end do
  end subroutine {name}_code_{kind}
"""


def _polymorphic_intergrid_kernel(name, first, second):
    """Build an inter-grid kernel module whose code is an interface.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_INTERGRID_MEMBER.format(name=name, kind=kind)
        for kind in (first, second))
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_readwrite,      &
                           gh_read, gh_coarse, gh_fine, cell_column,       &
                           any_discontinuous_space_1,                      &
                           any_discontinuous_space_2
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_readwrite,                         &
                  any_discontinuous_space_1, mesh_arg=gh_coarse),          &
         arg_type(gh_field, gh_real, gh_read,                              &
                  any_discontinuous_space_2, mesh_arg=gh_fine) /)
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


def _polymorphic_intergrid_algorithm(name, field_module, field_type):
    """Build an algorithm invoking one polymorphic inter-grid kernel.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.

    :returns: Fortran source for the program.
    :rtype: str
    """
    return f"""
program kokkos_{name}_test
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: coarse_field, fine_field
  call invoke({name}_kernel_type(coarse_field, fine_field))
end program kokkos_{name}_test
"""


def _polymorphic_algorithm(name, field_module, field_type, kind,
                           stencil=False):
    """Build an algorithm invoking one polymorphic kernel.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.
    :param str kind: the kind of the scalar argument.
    :param bool stencil: whether the kernel's read field carries a stencil, in
        which case the invoke passes its extent as well.

    :returns: Fortran source for the program.
    :rtype: str
    """
    extent_declaration = ""
    extent_actual = ""
    if stencil:
        extent_declaration = "\n  integer :: extent = 1"
        extent_actual = ", extent"
    return f"""
program kokkos_{name}_test
  use constants_mod, only : {kind}
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: out_field, in_field
  real(kind={kind}) :: scaling{extent_declaration}
  call invoke({name}_kernel_type(out_field, in_field{extent_actual}, scaling))
end program kokkos_{name}_test
"""


@pytest.fixture(name="polymorphic_intergrid_target")
# pylint: disable-next=unused-argument
def polymorphic_intergrid_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke of a kind-polymorphic inter-grid kernel.

    As for ``polymorphic_target``, the r_double member is declared first and
    the algorithm passes r_solver fields, so the match is the second member.
    """
    return _invoke(
        tmp_path, "restrict_poly",
        _polymorphic_intergrid_algorithm(
            "restrict_poly", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_intergrid_kernel("restrict_poly", "r_double", "r_single"))


@pytest.fixture(name="polymorphic_target")
# pylint: disable-next=unused-argument
def polymorphic_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface exactly one member of which matches.

    The r_double member is declared first, so a selection that returned
    ``schedules[0]`` would return the wrong one and every assertion about the
    generated types would fail. The algorithm passes r_solver fields, which
    ``precision_map`` gives 4 bytes, so the r_single member is the match.
    """
    return _invoke(
        tmp_path, "tri_scale",
        _polymorphic_algorithm(
            "tri_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver"),
        _polymorphic_kernel("tri_scale", "r_double", "r_single"))


@pytest.fixture(name="unmatched_target")
# pylint: disable-next=unused-argument
def unmatched_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface no member of which matches.

    r_single is 4 bytes and r_quad is 16; the algorithm's ``field_type`` is
    r_def at 8. No monkeypatching: the widths are ``psyclone.cfg``'s.
    """
    return _invoke(
        tmp_path, "quad_scale",
        _polymorphic_algorithm(
            "quad_scale", "field_mod", "field_type", "r_def"),
        _polymorphic_kernel("quad_scale", "r_single", "r_quad"))


@pytest.fixture(name="ambiguous_target")
# pylint: disable-next=unused-argument
def ambiguous_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface every member of which matches.

    ``precision_map`` gives r_single and r_solver the same 4 bytes, so a
    matcher working in widths cannot separate them. Fortran can, because it
    resolves by name.
    """
    return _invoke(
        tmp_path, "dual_scale",
        _polymorphic_algorithm(
            "dual_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver"),
        _polymorphic_kernel("dual_scale", "r_single", "r_solver"))


@pytest.fixture(name="polymorphic_evaluator_target")
# pylint: disable-next=unused-argument
def polymorphic_evaluator_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke of a polymorphic interface asking for an evaluator.

    The precisions are ``polymorphic_target``'s, so the selection question is
    the one that already has an answer; what is new is the ``gh_evaluator``
    metadata standing between the matcher and it. The r_double member is
    declared first, so a selection returning ``schedules[0]`` returns the
    wrong one.
    """
    return _invoke(
        tmp_path, "eval_scale",
        _polymorphic_evaluator_algorithm(
            "eval_scale", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("eval_scale", "r_double", "r_single"))


@pytest.fixture(name="any_space_evaluator_target")
# pylint: disable-next=unused-argument
def any_space_evaluator_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create a polymorphic evaluator interface on an ``any_space``.

    Metadata cannot say how many components a basis on an ``any_space`` has,
    which is PSyclone's issue #461, so the interface the matcher builds cannot
    state the basis's first extent. That extent is never compared, so it must
    not be what stops the question being asked.
    """
    return _invoke(
        tmp_path, "any_scale",
        _polymorphic_evaluator_algorithm(
            "any_scale", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("any_scale", "r_double", "r_single",
                                      space="any_space_2",
                                      target="any_space_1", dim=1))


@pytest.fixture(name="ambiguous_evaluator_target")
# pylint: disable-next=unused-argument
def ambiguous_evaluator_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an evaluator interface every member of which matches.

    ``precision_map`` gives r_single and r_solver the same 4 bytes. The point
    of asking it again over evaluator metadata is that the ambiguity must
    survive the interface becoming buildable: a matcher that now answers must
    answer "both" here rather than "the first one".
    """
    return _invoke(
        tmp_path, "dual_eval",
        _polymorphic_evaluator_algorithm(
            "dual_eval", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("dual_eval", "r_single", "r_solver"))


@pytest.fixture(name="unmodelled_target")
# pylint: disable-next=unused-argument
def unmodelled_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface the matcher cannot be asked about.

    The precisions are the matching fixture's, so nothing about the kinds
    stops selection. What stops it is the stencil: ``KernelInterface`` builds
    the interface the metadata implies before comparing anything, and issue
    #928 leaves stencils among the parts it does not build.
    """
    return _invoke(
        tmp_path, "stencil_scale",
        _polymorphic_algorithm(
            "stencil_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver", stencil=True),
        _polymorphic_kernel("stencil_scale", "r_double", "r_single",
                            stencil=True))


def test_lfric_kokkos_trans_selects_an_intergrid_schedule(
        polymorphic_intergrid_target):
    """A kind-polymorphic inter-grid kernel is selected, not refused.

    Selecting between an interface's members means building the argument list
    the metadata implies and comparing it with each member's, and PSyclone's
    issue #928 left the two inter-grid callbacks of that builder unwritten.
    Every polymorphic inter-grid kernel in the model was therefore refused for
    a reason that had nothing to do with what a region can express -- the
    question could not be asked. Asking it is what this asserts, and the
    r_single answer is what says the comparison really ran rather than
    returning the first member.
    """
    _, loop, kernel = polymorphic_intergrid_target
    assert kernel.is_intergrid
    assert len(kernel.get_callees()) == 2

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "restrict_poly_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    code = LFRicKokkosTrans().apply(loop)
    assert "double" not in code
    assert "float *coarse_field_data" in code
    assert "const float *fine_field_data" in code
    assert "const int *cell_map_data" in code


def test_lfric_kokkos_trans_selects_the_matching_schedule(polymorphic_target):
    """The member the algorithm's precisions pick is the one captured.

    Before stage 3 this refused with "LFRicKokkosTrans requires exactly one
    kernel schedule."

    The interface declares its r_double member first, so returning
    ``schedules[0]`` would pass every test that only checked a region was
    generated. The assertions below are therefore about the *width*: the
    algorithm passes r_solver fields, which ``precision_map`` gives 4 bytes,
    so the match is r_single and the region is float throughout.
    """
    _, loop, kernel = polymorphic_target
    assert len(kernel.get_callees()) == 2
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "tri_scale_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    cpp = LFRicKokkosTrans().apply(loop)
    assert "double" not in cpp
    assert "Kokkos::View<float*" in cpp
    assert "const float scaling" in cpp


def test_lfric_kokkos_trans_names_the_region_after_the_schedule(
        polymorphic_target):
    """One interface's members generate two differently named regions.

    What this protects is a build in which one interface is invoked at two
    precisions: naming both regions after the kernel would emit two launch
    symbols called ``tri_scale_kokkos`` whose arguments are float in one
    translation unit and double in the other.
    """
    _, loop, _ = polymorphic_target
    cpp = LFRicKokkosTrans().apply(loop)
    assert "tri_scale_r_single_kokkos" in cpp
    assert "tri_scale_kokkos" not in cpp


def test_lfric_kokkos_trans_refuses_when_no_schedule_matches(unmatched_target):
    """An interface no member of which the algorithm could have called.

    r_single is 4 bytes and r_quad is 16, against a ``field_type`` algorithm
    argument at r_def's 8. Nothing is monkeypatched: the widths are the ones
    ``psyclone.cfg`` records, so this is a case the matcher really rejects
    rather than one arranged to look rejected.
    """
    _, loop, _ = unmatched_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found no implementation of 'quad_scale_code'" in str(error.value)
    assert "out of 2" in str(error.value)


def test_lfric_kokkos_trans_refuses_an_ambiguous_interface(ambiguous_target):
    """An interface every member of which the algorithm could have called.

    ``precision_map`` gives r_single and r_solver the same 4 bytes, so the
    matcher cannot separate them; Fortran can, because it resolves by name.
    Refusing is deliberate. Picking either would be right only by coincidence,
    and the survey in psy-ir-aidev measures whether any GungHo kernel reaches
    this at all.
    """
    _, loop, _ = ambiguous_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found 2 implementations of 'dual_scale_code'" in str(error.value)
    assert "dual_scale_code_r_single, dual_scale_code_r_solver" in str(
        error.value)
    assert "will not choose between them" in str(error.value)


def test_lfric_kokkos_trans_accepts_a_polymorphic_kernel(
        polymorphic_evaluator_target):
    """An interface asking for an evaluator is selected between, not refused.

    Until this task the matcher could not build the interface an evaluator
    implies -- PSyclone's issue #928 -- so every member of such an interface
    was refused together, whatever its precisions were. The two members here
    differ in exactly the way ``polymorphic_target``'s do, so the answer is
    the same answer; the only new thing is that it can now be reached.

    The assertions are about the width rather than about a region existing,
    because the r_double member is declared first: returning ``schedules[0]``
    would generate a region that compiles and is wrong. The basis stays
    ``double`` in both members, which is what LFRic writes -- the PSy layer
    tabulates it once at r_def -- so the region carries a float field beside
    a double basis and that mixture is itself the evidence that the formals
    were read rather than assumed.
    """
    _, loop, kernel = polymorphic_evaluator_target
    assert kernel.eval_shapes == ["gh_evaluator"]
    assert len(kernel.get_callees()) == 2

    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "eval_scale_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    cpp = LFRicKokkosTrans().apply(loop)
    assert "eval_scale_r_single_kokkos" in cpp
    assert "eval_scale_kokkos" not in cpp
    assert "Kokkos::View<float*" in cpp
    assert ("auto basis_w1_on_w3 = lfric_kokkos::stage<\n"
            "      Kokkos::View<const double***, Kokkos::LayoutLeft, "
            "MemorySpace, ReadOnly>>(\n"
            "      basis_w1_on_w3_data, lfric_kokkos::Role::transient, 3, "
            "ndf_w1, ndf_w3);") \
        in cpp
    assert "weights" not in cpp


def test_lfric_kokkos_trans_selects_a_member_on_an_any_space(
        any_space_evaluator_target):
    """A basis on an ``any_space`` does not stop the question being asked.

    Metadata fixes a basis's first extent for a named function space and
    cannot for an ``any_space``, which is PSyclone's issue #461. That extent
    is not one of the things the matcher compares -- it is a literal on both
    sides -- so failing to produce it must not be read as the members failing
    to match. Before this task it was: the interface builder raised, the
    refusal said "found no implementation matching the precisions", and eight
    GungHo loops were reported as having no matching member when their
    precisions had never been looked at.
    """
    _, _, kernel = any_space_evaluator_target
    assert len(kernel.get_callees()) == 2

    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "any_scale_code_r_single"


def test_lfric_kokkos_trans_refuses_an_ambiguous_evaluator_interface(
        ambiguous_evaluator_target):
    """Two members of an evaluator interface matching equally is a refusal.

    The companion of
    :py:func:`test_lfric_kokkos_trans_refuses_an_ambiguous_interface`, asked
    through the metadata this task makes askable. Widening what the matcher
    can be asked about must not turn a refusal into a first-match, so the
    refusal is asserted to name both members rather than to merely happen.
    """
    _, loop, _ = ambiguous_evaluator_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found 2 implementations of 'dual_eval_code'" in str(error.value)
    assert "dual_eval_code_r_single, dual_eval_code_r_solver" in str(
        error.value)
    assert "will not choose between them" in str(error.value)


def test_lfric_kokkos_trans_refuses_when_the_matcher_cannot_answer(
        unmodelled_target):
    """A matcher that cannot be asked has not answered "no".

    ``KernelInterface`` builds the interface the metadata implies before it
    compares any precision, and PSyclone's issue #928 leaves stencils,
    evaluator shapes, CMA and inter-grid kernels among the parts it does not
    build, raising ``NotImplementedError``. Letting that be read as a mismatch
    would report a kind disagreement about a kernel whose kinds were never
    looked at; letting it escape would make a private helper of a
    transformation raise something other than ``TransformationError``.

    This calls ``_schedule`` rather than ``validate`` on purpose.
    ``_validate_kernel_metadata`` refuses a stencil first, so through
    ``validate`` this case is unreachable today -- which is exactly why the
    guard cannot rely on it. The survey in psy-ir-aidev calls ``_schedule``
    with no metadata check in front of it, because it reports every blocker of
    a loop rather than stopping at the first, and it is what found this.
    """
    _, _, kernel = unmodelled_target
    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._schedule(kernel)
    assert ("cannot tell which of the 2 implementations of "
            "'stencil_scale_code'" in str(error.value))
    assert "outside what PSyclone's own matcher models" in str(error.value)
    assert "TODO #928" in str(error.value)
