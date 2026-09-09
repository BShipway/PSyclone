# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosContractMixin: the capture contract."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _ALGORITHM, _DEPENDENT_SECTION_KERNEL, _KERNEL, _LEVEL_ALGORITHM,
    _LEVEL_KERNEL, _LOCAL_ALGORITHM, _LOCAL_KERNEL, _SECTION_ALGORITHM,
    _SECTION_KERNEL, _coloured_inner, _invoke)

from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.lfric import LFRicArgStencil
from psyclone.psyir.nodes import (
    ArrayReference, CodeBlock, IntrinsicCall, Literal, Range, Reference)
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType
from psyclone.psyir.transformations import TransformationError
from psyclone.transformations import LFRicColourTrans


# The same kernel with the column floored at a constant imported from a module
# that is not on the search path, so its symbol stays unresolved. The lowering
# has to expand every reference in the statement to decide whether it is an
# array, and cannot do that for a symbol whose declaration it has never seen.
# The argument of an intrinsic is where the two answers part company:
# ArrayAssignment2LoopsTrans.validate() skips references a Call encloses, and
# its apply() expands them like any other. A kernel of this shape, in
# solver_moist_correction_alg_mod, is what showed the difference.
_UNRESOLVED_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "  use fs_continuity_mod, only : w3, w2v",
    "  use fs_continuity_mod, only : w3, w2v\n"
    "  use unresolvable_constants_mod, only : eps").replace(
    "difference(w3_idx : w3_idx + nl) = &",
    "difference(w3_idx : w3_idx + nl) = max(eps, &").replace(
    "- mass_flux(b_idx : b_idx + nl)",
    "- mass_flux(b_idx : b_idx + nl))")


# The same kernel with a scalar local named after one of the identifiers the
# team launch declares around the kernel body. The kernel's declaration would
# shadow the launch's and then be assigned to, which C++ accepts: the region
# would run with a team size the kernel had overwritten. That is a wrong answer
# rather than a compile error, which is why it is refused.
_TEAM_NAME_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k, team_size\n").replace(
    "    partial(1) = field_in(map_w3(1))",
    "    team_size = nlayers\n    partial(1) = field_in(map_w3(1))").replace(
    "    do k = 2, nlayers", "    do k = 2, team_size")


# The same kernel with a scalar local named 'team' and no automatic array. The
# hierarchical launch declares 'team' as its lambda parameter whether or not
# there is scratch to place, so the reserved names cannot be conditional on
# there being an automatic array as they were until this stage.
_TEAM_LEVEL_KERNEL = _LEVEL_KERNEL.replace(
    "    real(kind=r_def) :: scaling",
    "    integer(kind=i_def) :: team\n"
    "    real(kind=r_def) :: scaling").replace(
    "    scaling = 0.5_r_def",
    "    team = nlayers\n    scaling = 0.5_r_def").replace(
    "    do k = 1, nlayers - 1", "    do k = 1, team - 1")


# A kernel with no local arrays and a scalar local named 'ncells'. The cell
# count is declared by both launch shapes, not only the team one, so this
# refusal does not depend on there being scratch to place.
_NCELLS_LOCAL_KERNEL = _KERNEL.replace(
    "integer(kind=i_def) :: k, df",
    "integer(kind=i_def) :: k, df, ncells").replace(
    "    do k = 0, nlayers - 1",
    "    ncells = nlayers\n    do k = 0, ncells - 1")


# The same kernel with a formal named after a C++ keyword. Fortran reserves no
# words at all, so `const` is an ordinary dummy argument -- and it is the one
# project_eliminated_theta_q32_kernel_mod declares, whose region the compiler
# met as `const float const,`. The kernel's own dummy names are its own
# business, the PSy layer calling it positionally, so only the generated C++
# ever has to write them.
_KEYWORD_FORMAL_KERNEL = _LOCAL_KERNEL.replace("field_in", "const")


# The same again for a local, which the launch declares in the scope it
# generates the kernel body into. `new` is a C++ keyword and an ordinary
# Fortran name.
_KEYWORD_LOCAL_KERNEL = _LOCAL_KERNEL.replace("swept", "new")


_FACE_QUADRATURE_ALGORITHM = """
program kokkos_face_quadrature_test
  use field_mod, only : field_type
  use quadrature_face_mod, only : quadrature_face_type
  use face_weight_kernel_mod, only : face_weight_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(quadrature_face_type) :: qr
  call invoke(face_weight_kernel_type(out_field, in_field, qr))
end program kokkos_face_quadrature_test
"""


# Face quadrature is a third shape, with a point count and a face count of its
# own. It is outside what this capability models and is refused by name.
_FACE_QUADRATURE_KERNEL = """
module face_weight_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, cell_column, gh_quadrature_face
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: face_weight_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(func_type) :: meta_funcs(1) = (/                                  &
         func_type(w3, gh_basis) /)
    integer :: gh_shape = gh_quadrature_face
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: face_weight_code
  end type face_weight_kernel_type
contains
  subroutine face_weight_code(nlayers, field_out, field_in,                &
                              ndf_w3, undf_w3, map_w3, basis_w3,           &
                              nfaces, np_xyz, weights_xyz)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: nfaces, np_xyz
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_def), dimension(np_xyz,nfaces), intent(in) :: weights_xyz
    real(kind=r_def), dimension(1,ndf_w3,np_xyz,nfaces), intent(in) ::     &
                                                                 basis_w3
    integer(kind=i_def) :: k, df, qp, face
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      total = 0.0_r_def
      do df = 1, ndf_w3
        do face = 1, nfaces
          do qp = 1, np_xyz
            total = total + weights_xyz(qp,face)                           &
                  * basis_w3(1,df,qp,face) * field_in(map_w3(df) + k)
          end do
        end do
      end do
      field_out(map_w3(1) + k) = total
    end do
  end subroutine face_weight_code
end module face_weight_kernel_mod
"""


@pytest.fixture(name="face_quadrature_target")
# pylint: disable-next=unused-argument
def face_quadrature_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads face-quadrature basis data."""
    return _invoke(
        tmp_path, "face_weight", _FACE_QUADRATURE_ALGORITHM,
        _FACE_QUADRATURE_KERNEL)


@pytest.fixture(name="team_name_target")
# pylint: disable-next=unused-argument
def team_name_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'team_size'."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _TEAM_NAME_KERNEL)


@pytest.fixture(name="team_level_target")
# pylint: disable-next=unused-argument
def team_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'team'."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _TEAM_LEVEL_KERNEL)


@pytest.fixture(name="ncells_local_target")
# pylint: disable-next=unused-argument
def ncells_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'ncells'."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _NCELLS_LOCAL_KERNEL)


@pytest.fixture(name="keyword_formal_target")
# pylint: disable-next=unused-argument
def keyword_formal_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel names a formal after a C++ keyword."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _KEYWORD_FORMAL_KERNEL)


@pytest.fixture(name="keyword_local_target")
# pylint: disable-next=unused-argument
def keyword_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel names a local after a C++ keyword."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _KEYWORD_LOCAL_KERNEL)


def test_lfric_kokkos_trans_refuses_an_unlowerable_section(
        tmp_path, clear_module_manager_instance):
    """A section the lowering will not touch stays a refusal.

    Widening a capture is only safe if the thing it delegates to keeps its
    own refusals, so this asserts that PSyclone's reason is carried out
    rather than swallowed.
    """
    # pylint: disable=unused-argument
    # pylint: disable-next=unused-variable
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _DEPENDENT_SECTION_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot lower an array section to a loop" in str(error.value)
    assert "loop-carried dependencies" in str(error.value)


def test_lfric_kokkos_trans_section_check_predicts_the_lowering(
        tmp_path, clear_module_manager_instance):
    """The section check answers for the lowering, not for its validate.

    ArrayAssignment2LoopsTrans.validate() accepts this assignment and its
    apply() then refuses it: validate() skips the references a Call encloses
    and apply() expands them like any other. Asking the weaker question left
    _validate_sections() promising a prediction it did not make, so a caller
    using the two helpers in turn -- as the coverage survey does -- got an
    exception out of the lowering instead of a refusal.

    Asserted through the public validate(), which reports the section reason
    because the section check is complete on its own. Other checks would
    refuse this kernel too, for its unresolved import; that they run later is
    what makes the message evidence about this one.
    """
    # pylint: disable=unused-argument
    # pylint: disable-next=unused-variable
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _UNRESOLVED_SECTION_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot lower an array section to a loop" in str(error.value)
    assert "eps" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_section_outside_an_assignment(
        section_target):
    """Only a whole-column assignment is within reach of the lowering.

    An intrinsic called as a statement can hold a section that no assignment
    encloses, which the lowering has no way to rewrite. Validation says so
    itself rather than letting the backend fail later on a bare Range.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]
    field = schedule.symbol_table.lookup("difference")
    integer = ScalarType.integer_type()
    section = ArrayReference.create(field, [Range.create(
        Literal("1", integer), Literal("2", integer))])
    schedule.addchild(IntrinsicCall.create(
        IntrinsicCall.Intrinsic.RANDOM_NUMBER, [section]))

    with pytest.raises(TransformationError, match="outside an assignment"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_refuses_an_unhandled_eval_shape(
        face_quadrature_target):
    """A shape outside the two modelled ones is refused by name.

    Face and edge quadrature carry a face count and a single point count
    rather than the XYoZ pair. Nothing here has been measured against the
    model for them, so the refusal stays and says which shape it is about
    rather than reporting quadrature as a whole as unsupported.
    """
    _, loop, kernel = face_quadrature_target
    assert kernel.eval_shapes == ["gh_quadrature_face"]

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the 'gh_quadrature_face' "
            "evaluator shape." in str(error.value))


def test_lfric_kokkos_trans_rejects_a_columnwise_assembly(target):
    """The CMA refusal is by operation, ahead of the argument-type walk.

    A CMA kernel is refused for what it does rather than for what it takes:
    an assembly kernel builds a banded matrix from an LMA one, so its
    arguments alone would now pass. The two refusals are therefore both
    needed and are asserted apart.
    """
    _, loop, kernel = target
    kernel._cma_operation = "assembly"
    with pytest.raises(TransformationError, match="CMA operators"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_a_columnwise_operator(target):
    """A CMA operator is refused where an LMA one is now accepted.

    The two are not variants of one capability. An LMA operator is a rank-3
    array the kernel slices by cell, which the region describes as a View like
    any other; a CMA operator is a banded matrix with its own bandwidth and
    indexing arguments, and none of that machinery exists here.
    """
    _, loop, kernel = target
    kernel.arguments.args[1]._argument_type = "gh_columnwise_operator"
    with pytest.raises(TransformationError, match="gh_columnwise_operator"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_declares_the_cell_position(operator_target):
    """The kernel's cell argument is declared, not taken across the ABI.

    LFRic passes its loop counter for that argument, and the counter is
    lowered away when the loop becomes a launch. Taking it would therefore
    read an unassigned variable on every cell -- a wrong answer rather than a
    build failure, which is why this is asserted from both sides.
    """
    _, loop, _ = operator_target
    code = LFRicKokkosTrans().apply(loop)

    # The launch renames its own index, because the kernel has taken 'cell'.
    assert "const int cell = cell_1 + 1;" in code

    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1] for parameter in signature.split(",")
        if parameter.strip().split()
    ]
    assert "cell" not in parameters
    assert "ncells" in parameters


def test_lfric_kokkos_trans_operator_cell_actual_must_be_the_counter(
        operator_target, monkeypatch):
    """A cell actual that is not the loop's own variable is refused.

    The transformation reads the position of the cell argument from
    ``has_operator()``, mirroring ``ArgOrdering.generate``. If those two ever
    part company the actual at that index stops being the loop variable, and
    the region would drop the wrong argument -- so the assumption is checked
    rather than trusted.
    """
    _, loop, _ = operator_target
    original = KernCallArgList.generate

    def _shuffled(self, var_accesses=None):
        original(self, var_accesses=var_accesses)
        # Put something that is not the loop variable where the cell is.
        self._psyir_arglist[0] = self._psyir_arglist[1].copy()

    monkeypatch.setattr(KernCallArgList, "generate", _shuffled)
    with pytest.raises(TransformationError, match="cell index"):
        LFRicKokkosTrans().apply(loop)


def test_lfric_kokkos_trans_refuses_an_unsupported_stencil_type(target):
    """Stencil storage and halo requirements are not silently captured.

    The refusal is shape-specific: 'cross', 'cross2d' and 'region' are
    accepted, and every other shape -- 'xory1d' here, which has a direction
    argument on top of a 1-D size -- is refused by a message naming both the
    shape it found and the whole set it would have taken.
    """
    _, loop, kernel = target
    kernel.arguments.args[1].stencil = LFRicArgStencil(name="xory1d")

    with pytest.raises(TransformationError, match="stencil") as error:
        LFRicKokkosTrans().validate(loop)

    assert "'xory1d'" in str(error.value)
    for shape in LFRicKokkosTrans._SUPPORTED_STENCILS:
        assert shape in str(error.value)


def test_lfric_kokkos_trans_rejects_multiple_kernels(target, monkeypatch):
    """The launch body must correspond to exactly one kernel schedule."""
    _, loop, kernel = target
    monkeypatch.setattr(loop, "kernels", lambda: [kernel, kernel])
    with pytest.raises(TransformationError, match="exactly one kernel"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_codeblock(target):
    """Opaque kernel statements cannot cross the backend boundary."""
    _, loop, kernel = target
    kernel.get_callees()[0].addchild(
        CodeBlock([], structure=CodeBlock.Structure.STATEMENT))
    with pytest.raises(TransformationError, match="CodeBlock"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_refuses_a_local_named_after_the_launch(
        team_name_target):
    """A local shadowing a name the team launch declares is refused.

    The launch index is renamed around such a collision instead, because a
    lambda parameter and a body declaration are a compile error and the fix
    is one name in two places. These seven are threaded through two launch
    shapes and through the scratch sizing, so they are refused rather than
    renamed; no GungHo kernel declares any of them.
    """
    _, loop, _ = team_name_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "generated launch declares 'team_size'" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_local_named_ncells(
        ncells_local_target):
    """A local named for the cell count is refused whatever the launch shape.

    ``_validate_formals`` already refuses a *formal* of this name. The cell
    count is declared by the range launch as well as the team one, so this
    kernel has no local arrays: the refusal must not be conditional on there
    being scratch to place.
    """
    _, loop, _ = ncells_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "generated launch declares 'ncells'" in str(error.value)


def test_lfric_kokkos_trans_refuses_an_unsizable_local(unsized_local_target):
    """A local sized by a module constant is refused, with the extent named.

    ``_constants`` could import ``n_moist`` and pass it into the region, so
    the refusal is a deliberate narrowing rather than an inability: the
    scratch size is computed by the launch, outside the region that constant
    would be passed to.
    """
    _, loop, _ = unsized_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "n_moist" in str(error.value)
    assert "kernel-local array 'swept'" in str(error.value)
    assert "kernel argument" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_local_of_an_unmapped_kind(
        unmapped_local_target):
    """A local array of a logical kind is refused, with the kind named."""
    _, loop, _ = unmapped_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "kernel-local array kinds" in str(error.value)
    assert "'rising'" in str(error.value)
    assert "'l_def'" in str(error.value)


def test_lfric_kokkos_trans_rejects_a_cxx_keyword_formal(
        keyword_formal_target):
    """A formal Fortran allows and C++ reserves is refused by name.

    ``project_eliminated_theta_q32_kernel_mod`` declares a dummy argument
    called ``const``, which is an ordinary Fortran name and a C++ keyword: the
    region generated for it said ``const float const,`` and the model build
    stopped there. That loop reaches the transformation at all because this
    branch released the halo bound, so the collision is this branch's to
    refuse. A rename is not attempted: it would have to reach every place the
    backend writes a name.
    """
    _, loop, kernel = keyword_formal_target
    # The transformation reaches the kernel schedule the same way.
    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)

    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._validate_formals(schedule)

    assert ("cannot name the kernel formal 'const' in the generated region, "
            "because it is a C++ keyword" in str(error.value))

    # And the loop as a whole is refused, rather than a region being written
    # that no compiler would take.
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'const'" in str(error.value)


def test_lfric_kokkos_trans_rejects_a_cxx_keyword_local(keyword_local_target):
    """A kernel-local of a C++ keyword's name is refused as a formal is.

    The launch declares a local in the scope it generates the kernel body
    into, where the name is no more writable than it is in the signature.
    """
    _, loop, kernel = keyword_local_target
    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)

    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._validate_locals(schedule)

    assert ("cannot name the kernel local 'new' in the generated region, "
            "because it is a C++ keyword" in str(error.value))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'new'" in str(error.value)


def test_team_is_reserved_for_a_hierarchical_kernel_without_scratch(
        team_level_target):
    """``team`` is refused for a kernel the loop selection alone reaches.

    Until this stage the seven generated names were reserved only for a
    kernel with a local array, because only scratch reached a launch that
    declares ``team``. The hierarchical launch declares it whether or not
    there is scratch, so the reservation follows the launch rather than the
    scratch.
    """
    _, loop, _ = team_level_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("generated launch declares 'team', but the kernel declares a "
            "local of that name" in str(error.value))


def test_lfric_kokkos_trans_evaluator_predicate_refuses_an_unmodelled_shape(
        target):
    """The evaluator-shape rule is askable on its own, shape by shape."""
    _, _, kernel = target
    kernel._basis_required = True
    kernel._eval_shapes = ["gh_quadrature_edge"]

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_evaluator(kernel)

    assert ("LFRicKokkosTrans does not support the 'gh_quadrature_edge' "
            "evaluator shape." in str(error.value))


def test_lfric_kokkos_trans_evaluator_predicate_accepts_modelled_shapes(
        target):
    """The two shapes the region models pass the rule, together or apart.

    Asked of each shape separately and of both at once, because a rule
    written to accept a single shape would refuse the kernels that ask for
    both -- and those are the ones the model has most of.
    """
    _, _, kernel = target
    for shapes in (["gh_quadrature_xyoz"], ["gh_evaluator"],
                   ["gh_quadrature_xyoz", "gh_evaluator"]):
        kernel._eval_shapes = shapes
        LFRicKokkosTrans._validate_evaluator(kernel)


def test_lfric_kokkos_trans_field_type_predicate_refuses_a_logical_field(
        target):
    """The field-type rule walks every field argument on its own.

    LFRic's metadata admits ``gh_real`` and ``gh_integer`` fields and no
    third intrinsic, and the ABI now carries both, so the witness has to be
    made rather than found. Made rather than deleted, because the rule is
    stated against what a View's elements may be and not against today's
    metadata: a field intrinsic added to LFRic would be refused by name
    instead of reaching the backend as a type it has no row for.
    """
    _, _, kernel = target
    kernel.arguments.args[1]._intrinsic_type = "logical"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_field_types(kernel)

    assert ("LFRicKokkosTrans supports only integer and real fields, but "
            "'mr' is logical." in str(error.value))


def test_lfric_kokkos_trans_predicates_accept_a_supported_loop(target):
    """Each of the four returns for a loop that does not fail it.

    A predicate that raised for everything would report every pattern as
    blocked, which is the failure mode a survey cannot see from the inside.
    """
    _, loop, kernel = target

    LFRicKokkosTrans._validate_iteration_space(loop)
    LFRicKokkosTrans._validate_halo_depth(loop)
    LFRicKokkosTrans._validate_evaluator(kernel)
    LFRicKokkosTrans._validate_field_types(kernel)


def test_lfric_kokkos_trans_field_type_predicate_ignores_the_shape(target):
    """A predicate answers for its own rule, not for the first blocker.

    This is the whole point of naming them. A kernel that asks for an
    unmodelled evaluator shape *and* carries a field of an intrinsic no View
    holds is two blocked patterns, and asking through 'validate' would only
    ever name the shape because it is checked first.
    """
    _, _, kernel = target
    kernel._eval_shapes = ["gh_quadrature_face"]
    kernel.arguments.args[0]._intrinsic_type = "logical"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_field_types(kernel)
    assert "'moist_dyn' is logical" in str(error.value)

    with pytest.raises(TransformationError) as second:
        LFRicKokkosTrans._validate_evaluator(kernel)
    assert "does not support the 'gh_quadrature_face' evaluator shape" in str(
        second.value)


def test_lfric_kokkos_trans_metadata_refuses_by_argument_order(target):
    """The bundled check still refuses in argument order, not rule order.

    The first argument is a field of an intrinsic no View holds and the
    second is not an argument type the region can describe at all. Walking
    the arguments -- which is what the transformation has always done --
    reports the first argument's blocker; running the two rules as separate
    passes over all the arguments would report the second argument's
    instead, because the argument-type rule comes first within an argument.
    The named predicate shares the per-argument helper with this loop so
    that only one answer exists.
    """
    _, _, kernel = target
    kernel.arguments.args[0]._intrinsic_type = "logical"
    kernel.arguments.args[1]._argument_type = "gh_columnwise_operator"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_kernel_metadata(kernel)

    assert "'moist_dyn' is logical" in str(error.value)
    assert "gh_columnwise_operator" not in str(error.value)


def test_lfric_kokkos_trans_field_predicate_passes_over_a_non_field(
        operator_target):
    """The field rule walks the whole argument list and skips the rest.

    Asked through 'validate' it only ever sees an argument the walk has
    already accepted as a field. Asked on its own -- which is how the survey
    asks it -- it meets the scalars and operators too, and a rule that read
    an intrinsic off an LMA operator would raise something other than a
    refusal.
    """
    _, _, kernel = operator_target
    assert any(argument.argument_type != "gh_field"
               for argument in kernel.arguments.args)

    LFRicKokkosTrans._validate_field_types(kernel)


def test_lfric_kokkos_trans_refuses_an_unmodelled_access(
        shared_write_target):
    """An access neither arm answers is refused by name.

    'gh_inc' and 'gh_readinc' are read-modify-writes of a field at a shared
    dof, which is what an atomic update and a colouring both answer. A
    reduction is not: every cell accumulates into one value, and where that
    value lives -- a PSy-layer scalar the invoke finishes with a global sum --
    is somewhere the region has no View for.

    There is no such kernel to write. PSyclone's own metadata parser refuses
    'gh_reduction' on a user-supplied kernel outright, so a reduction reaches
    a coded LFRic loop by no route at all, and the access has to be put on the
    argument here to ask the question. That is the point of the test: the
    refusal is a claim about what this transformation does with an access it
    does not model, and it should not depend on which accesses LFRic happens
    to admit this year.

    """
    _, loop, kernel = shared_write_target
    kernel.arguments.args[0]._access = AccessType.REDUCTION

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "'REDUCTION'" in str(err.value)
    assert "'acc'" in str(err.value)
    # And it names what is modelled, so the reader is not left to infer it
    # from the absence of their access.
    assert "gh_readinc" in str(err.value)


def test_lfric_kokkos_trans_refuses_a_cell_index_it_cannot_place(
        shared_write_operator_target, monkeypatch):
    """An actual that is neither the loop's cell nor the map's is refused.

    ``ArgOrdering`` decides what the first actual of an operator kernel is,
    and this transformation reads that decision rather than trusting it: an
    entry dropped in the wrong place is a region that compiles, runs and
    reads the wrong data. The refusal is kept reachable by naming the loop's
    variable something the actual is not.

    """
    psy, loop, _ = shared_write_operator_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)
    inner = _coloured_inner(schedule)
    monkeypatch.setattr(inner, "_variable",
                        schedule.symbol_table.lookup("colour"))

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().apply(inner)
    assert "cell index as the first argument" in str(err.value)
    assert "cmap(colour,cell)" in str(err.value)


def test_lfric_kokkos_trans_knows_which_first_actual_is_a_cell(
        shared_write_operator_target):
    """Only the two spellings of the loop's cell are read as one.

    An operator kernel takes the cell index as its first argument, and the
    generated region has to name the launch index there instead. Two
    spellings mean the cell -- the loop's own variable, or its entry in the
    colour map -- and anything else has to be refused rather than renamed,
    because renaming it would silently pass the launch index where the
    kernel expected something else. Neither of the other shapes reaches
    here from LFRic today, so they are asked of the classifier directly.
    """
    _, loop, kernel = shared_write_operator_target
    integer = ScalarType.integer_type()
    unrelated = DataSymbol("lookup", ArrayType(integer, [2, 2]))

    assert LFRicKokkosTrans._is_the_loops_cell(
        kernel, loop, Reference(loop.variable))
    assert not LFRicKokkosTrans._is_the_loops_cell(
        kernel, loop, Literal("1", integer))
    assert not LFRicKokkosTrans._is_the_loops_cell(
        kernel, loop, ArrayReference.create(
            unrelated, [Literal("1", integer), Literal("1", integer)]))


@pytest.mark.parametrize("bound", ["ncells", "ndofs", "first_cell"])
def test_lfric_kokkos_trans_refuses_a_formal_colliding_with_a_bound(
        tmp_path, bound):
    """The bound formals are reserved whatever bounds the loop needs.

    The generated signature carries the launch's count and, for a loop that
    does not begin at the first cell, the cell it does begin at. A kernel
    declaring one of those names would have the launch's value written over
    its own, so all three names are refused for every kernel rather than for
    the loops that happen to use them: which two a capture uses depends on
    the loop it is asked of rather than on the kernel, and a kernel that
    captured from one invoke and not from another would be worse than one
    that never captures.
    """
    kernel = _KERNEL.replace("nlayers", bound)
    _, loop, _ = _invoke(tmp_path, "moist_dyn_gas", _ALGORITHM, kernel)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert (f"LFRicKokkosTrans adds '{bound}' to the generated signature, "
            "but the kernel already declares it." in str(error.value))
