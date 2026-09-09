# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosBoundsMixin: declared bounds and enquiries."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _LITERAL_LOCAL_KERNEL, _LOCAL_ALGORITHM, _SECTION_ALGORITHM,
    _SECTION_KERNEL, _UNRENDERABLE_ORIGIN_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.transformations import TransformationError


_IMPLICIT_ALGORITHM = """
program kokkos_implicit_test
  use field_mod, only : field_type
  use normals_sum_kernel_mod, only : normals_sum_kernel_type
  implicit none
  type(field_type) :: field_out, field_in
  call invoke(normals_sum_kernel_type(field_out, field_in))
end program kokkos_implicit_test
"""


# The boundary-condition shape: a reference-element property and a mesh
# property, each declared by the kernel with the extent left out. Every one of
# the six loops the catalogue counts under "implicit extent" is this --
# 'weighted_div_bd_code' writes 'real(kind=r_def), intent(in) ::
# outward_normals(:,:)' beside 'integer(kind=i_def), intent(in) ::
# adjacent_face(:)' -- and the extent the declaration leaves out is not
# unknown: the PSy layer holds the array it passes and can measure it.
#
# The two are here together because they are measured differently. The
# reference-element array is passed whole, so the region's View has the two
# extents the actual has; the mesh property is passed one cell's column at a
# time, so the region takes it whole and the extent measured is the one the
# slice leaves. 'nfaces_re_h' is declared and unused, as LFRic's own kernels
# declare it: the argument order is the metadata's rather than the body's.
_IMPLICIT_KERNEL = """
module normals_sum_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column, reference_element_data_type,       &
                           mesh_data_type, adjacent_face,                  &
                           outward_normals_to_horizontal_faces
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: normals_sum_kernel_type
    type(arg_type) :: meta_args(2) = (/                                &
         arg_type(gh_field, gh_real, gh_write, w3),                    &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(reference_element_data_type) :: meta_reference_element(1) = (/ &
         reference_element_data_type(                                   &
             outward_normals_to_horizontal_faces) /)
    type(mesh_data_type) :: meta_mesh(1) = (/                           &
         mesh_data_type(adjacent_face) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: normals_sum_code
  end type normals_sum_kernel_type
contains
  subroutine normals_sum_code(nlayers, field_out, field_in,   &
                              ndf_w3, undf_w3, map_w3,        &
                              nfaces_re_h, outward_normals,   &
                              adjacent_face)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: nfaces_re_h
    real(kind=r_def), intent(in) :: outward_normals(:,:)
    integer(kind=i_def), intent(in) :: adjacent_face(:)
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df, face
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = field_in(map_w3(df) + k)
        do face = 1, size(adjacent_face, 1)
          field_out(map_w3(df) + k) = field_out(map_w3(df) + k) + &
              outward_normals(1, face) *                         &
              outward_normals(2, adjacent_face(face))
        end do
      end do
    end do
  end subroutine normals_sum_code
end module normals_sum_kernel_mod
"""


# The same kernel asking for the bounds themselves rather than for the size.
# Fortran fixes the lower bound of an assumed-shape dummy at 1 whatever the
# actual was declared with, so 'lbound' is the literal 1 and 'ubound' is the
# measured extent -- the one place where the declared origin A3 reads is the
# dummy's own and not the actual's.
_IMPLICIT_BOUND_KERNEL = _IMPLICIT_KERNEL.replace(
    "do face = 1, size(adjacent_face, 1)",
    "do face = lbound(adjacent_face, 1), ubound(adjacent_face, 1)")


# An assumed shape whose declaration states its lower bound. Fortran allows
# it, and it is the one assumed shape that stays refused: the extent would be
# taken from the actual and the origin from the declaration, so the View's
# shape would be read out of two places at once.
_IMPLICIT_ORIGIN_KERNEL = _IMPLICIT_KERNEL.replace(
    "intent(in) :: adjacent_face(:)",
    "intent(in) :: adjacent_face(0:)")


# An assumed shape that is not a kernel argument, and so has no call to be
# measured through. The declaration is not legal Fortran outside a dummy
# argument list, which is the point: there is no route by which a local can
# acquire a shape from a caller, so the region has nothing to size a scratch
# View by and refuses rather than guessing one.
_IMPLICIT_LOCAL_KERNEL = _IMPLICIT_KERNEL.replace(
    "    integer(kind=i_def) :: k, df, face\n",
    "    integer(kind=i_def) :: k, df, face\n"
    "    real(kind=r_def), dimension(:) :: loose\n")


# SIZE in a dimension other than the first, of a rank-2 local. The literal
# bound is the answer, so the substitution has to reach the second entry of
# the declared shape rather than assuming the first.
_RANK_TWO_BOUND_KERNEL = _LITERAL_LOCAL_KERNEL.replace(
    "      swept(k,1) = swept(k + 1,1) - partial(k)",
    "      swept(k,1) = swept(k + 1,1) - partial(k) * size(swept, 2)")


# UBOUND of an array whose declared origin cannot be written. The bounds
# grammar already refuses that declaration, and the enquiry inherits the
# refusal rather than paraphrasing it, because the answer depends on the same
# bounds.
_LOWER_BOUND_ENQUIRY_KERNEL = _UNRENDERABLE_ORIGIN_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)",
    "    swept(ubound(swept, 1)) = partial(nlayers)")


# A whole-array assignment, which is where these enquiries mostly come from:
# the lowering writes LBOUND and UBOUND into the loop bounds of every
# full-extent section it rewrites, so they are produced by the transformation
# rather than by the kernel author.
_FULL_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "    difference(w3_idx : w3_idx + nl) = &\n"
    "        mass_flux(b_idx + 1 : b_idx + nl + 1) "
    "- mass_flux(b_idx : b_idx + nl)",
    "    difference(:) = difference(:) + (b_idx + nl)")


@pytest.fixture(name="implicit_target")
# pylint: disable-next=unused-argument
def implicit_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel leaves two formals' extents out."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_KERNEL)


@pytest.fixture(name="implicit_bound_target")
# pylint: disable-next=unused-argument
def implicit_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel asks an assumed shape for its bounds."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_BOUND_KERNEL)


@pytest.fixture(name="implicit_origin_target")
# pylint: disable-next=unused-argument
def implicit_origin_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel states one bound of an assumed shape."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_ORIGIN_KERNEL)


@pytest.fixture(name="implicit_local_target")
# pylint: disable-next=unused-argument
def implicit_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local of no stated shape."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_LOCAL_KERNEL)


@pytest.fixture(name="rank_two_bound_target")
# pylint: disable-next=unused-argument
def rank_two_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking the SIZE of a rank-2 local's second axis."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RANK_TWO_BOUND_KERNEL)


@pytest.fixture(name="lower_bound_enquiry_target")
# pylint: disable-next=unused-argument
def lower_bound_enquiry_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke asking the UBOUND of an array based other than at 1."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _LOWER_BOUND_ENQUIRY_KERNEL)


@pytest.fixture(name="full_section_target")
# pylint: disable-next=unused-argument
def full_section_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns a whole array at once."""
    return _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _FULL_SECTION_KERNEL)


def test_lfric_kokkos_trans_sizes_an_assumed_shape_from_the_actual(
        implicit_target):
    """A formal declared '(:)' is sized by the array the PSy layer passes.

    The declaration states no extent, but the extent is not unknown: Fortran
    takes it from the actual at the call, and the PSy layer holds that array.
    So the region carries the measurement -- 'SIZE' of the actual -- as a
    scalar of its own and sizes the View from that, which is the same route
    C2 measures a stencil dofmap's storage extent by.

    The kernel's own 'size(adjacent_face, 1)' has to resolve to that same
    scalar. A View sized by one reading of the shape and a loop bounded by
    another is a wrong answer rather than a refusal, which is why the two are
    asserted together.
    """
    psy, loop, _ = implicit_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> adjacent_face(adjacent_face_data, "
            "adjacent_face_extent_1, ncells);" in cpp)
    assert "const int adjacent_face_extent_1" in cpp
    assert "for(face=1; face<=adjacent_face_extent_1; face+=1)" in cpp

    fortran = str(psy.gen)
    assert "integer(c_int), value :: adjacent_face_extent_1" in fortran
    assert "SIZE(adjacent_face, dim=1)" in fortran


def test_lfric_kokkos_trans_sizes_a_rank_2_assumed_shape(implicit_target):
    """Every dimension left out of the declaration is measured, in order.

    A rank-2 assumed shape carries two extents and both are the actual's, so
    the region takes two scalars and the PSy layer measures the same array
    twice. The call is asserted whole because the scalars are positional: two
    measurements of one array in the wrong order size the View wrongly in a
    way that still compiles.
    """
    psy, loop, _ = implicit_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::View<const double**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> outward_normals(outward_normals_data, "
            "outward_normals_extent_1, outward_normals_extent_2);" in cpp)
    assert "const int outward_normals_extent_1" in cpp
    assert "const int outward_normals_extent_2" in cpp

    fortran = str(psy.gen)
    assert ("call normals_sum_kokkos(nlayers_field_out, field_out_data, "
            "field_in_data, ndf_w3, undf_w3, map_w3, nfaces_re_h, "
            "out_normals_to_horiz_faces, adjacent_face, "
            "SIZE(out_normals_to_horiz_faces, dim=1), "
            "SIZE(out_normals_to_horiz_faces, dim=2), "
            "SIZE(adjacent_face, dim=1), loop0_stop)" in fortran)


def test_lfric_kokkos_trans_refuses_an_assumed_shape_with_no_actual(
        implicit_local_target):
    """An assumed shape is only measurable where there is a call to measure.

    A kernel-local array declared '(:)' has no actual anywhere, so there is
    nothing to read its extent from and the region refuses it rather than
    choosing one. The refusal names the array and says which of the two
    shapeless declarations it carries, so a reader knows whether to look at
    the kernel or at its caller.
    """
    _, loop, _ = implicit_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'loose' to be declared with explicit bounds, but it is "
            "declared with an assumed shape" in str(error.value))


def test_lfric_kokkos_trans_refuses_an_assumed_shape_with_a_stated_origin(
        implicit_origin_target):
    """A shape half declared and half measured is not read from two places.

    'dimension(0:)' takes its extent from the actual and its origin from the
    declaration. Honouring both would give the View a shape assembled out of
    the caller and the callee at once, which is the confusion the declared
    bounds exist to avoid, so the measurement is not attempted and the
    refusal says which of the two it found.
    """
    _, loop, _ = implicit_origin_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'adjacent_face' to be declared with explicit bounds, "
            "but it is declared with an assumed shape whose lower bound its "
            "declaration states, so its origin and its size would be read "
            "from two different places" in str(error.value))


def test_lfric_kokkos_trans_assumed_shape_lower_bound_is_one(
        implicit_bound_target):
    """An assumed-shape formal is 1-based whatever the actual was declared as.

    Fortran gives the dummy the actual's *extent* and its own lower bound,
    which is 1 unless the dummy states otherwise. The origin A3 shifts every
    subscript by is therefore the declaration's own and must not be taken from
    the actual: 'lbound' is the literal 1, 'ubound' is the measured extent,
    and the subscripts are shifted by one like any other 1-based array's.
    """
    _, loop, _ = implicit_bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "for(face=1; face<=adjacent_face_extent_1; face+=1)" in cpp
    assert "adjacent_face((face - 1), cell)" in cpp


def test_lfric_kokkos_trans_resolves_bounds_from_the_declaration(
        bound_target):
    """LBOUND, UBOUND and SIZE become the bounds the kernel declared.

    Each is answered symbolically, against the symbol table, so the generated
    region carries the declared bound as an expression rather than a call.
    The loop is the evidence: its start is ``LBOUND(partial, 1) + 1`` and its
    end is ``UBOUND(partial, 1)``, and both have to have gone before the
    backend sees them.
    """
    _, loop, _ = bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "for(k=(1 + 1); k<=nlayers; k+=1)" in cpp
    # SIZE of a formal declared dimension(undf_w3), read in an expression
    # rather than as a bound.
    assert "(swept((k - 1)) + undf_w3)" in cpp
    # Nothing of the enquiries survives into the region. A bare "size(" would
    # match shmem_size, team_size and league_size, so the one call the kernel
    # made is named instead.
    for name in ("LBOUND", "UBOUND", "SIZE", "lbound(", "ubound(",
                 "size(field_in)"):
        assert name not in cpp


def test_lfric_kokkos_trans_resolves_a_bound_in_a_later_dimension(
        rank_two_bound_target):
    """``SIZE(swept, 2)`` takes the second entry of the declared shape.

    A rank-2 local declared ``dimension(nlayers,4)`` answers its second
    dimension with a literal, so the substitution has to index the shape
    rather than assume the first entry.
    """
    _, loop, _ = rank_two_bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(partial((k - 1)) * 4)" in cpp
    assert "swept_scratch_t::shmem_size(nlayers, 4)" in cpp


def test_lfric_kokkos_trans_resolves_a_lowered_full_section(
        full_section_target):
    """The bounds the lowering itself writes are substituted too.

    ``difference(:)`` is rewritten to an explicit loop by
    ``ArrayAssignment2LoopsTrans``, which writes ``LBOUND`` and ``UBOUND``
    into that loop's bounds. They are produced by the transformation rather
    than by the kernel author, which is why the substitution runs after the
    lowering and not before it.
    """
    _, loop, _ = full_section_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::parallel_for(Kokkos::TeamVectorRange"
            "(team, 1, undf_w3 + 1)," in cpp)
    assert "difference((idx - 1)) = (difference((idx - 1)) + (b_idx + nl));" \
        in cpp


def test_lfric_kokkos_trans_refuses_a_bound_of_a_scalar(scalar_bound_target):
    """``SIZE`` of a scalar is refused, naming the symbol.

    fparser2 parses it, the check that would refuse it being semantic rather
    than syntactic, so the transformation is the first thing in the chain
    that can notice.
    """
    _, loop, _ = scalar_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot resolve 'SIZE' of 'nlayers', which is not declared as an "
            "array" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_bound_of_an_element(
        element_bound_target):
    """``UBOUND(map_w3(1), 1)`` asks about an element, not about the array.

    An ``ArrayReference`` is a ``Reference``, so a test that accepted any
    reference would read this as the array's bound and generate the wrong
    answer silently.
    """
    _, loop, _ = element_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the first argument of 'UBOUND' to be a plain reference "
            "to a declared array" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_variable_dimension(
        variable_bound_target):
    """A dimension given by a variable cannot be resolved from the table."""
    _, loop, _ = variable_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the dimension of 'UBOUND' of 'partial' to be an "
            "integer literal" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_dimension_past_the_rank(
        out_of_range_bound_target):
    """A dimension outside the declared rank is refused rather than indexed."""
    _, loop, _ = out_of_range_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot resolve 'UBOUND' of 'partial' in dimension 2, since it is "
            "declared with rank 1" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_whole_size_above_rank_one(
        whole_size_target):
    """``SIZE(a)`` on a rank-2 array is a count, not a bound.

    It is valid Fortran and has a well-defined value, so the refusal is about
    what the region can express rather than about the source being wrong: no
    one entry of the declared shape answers it.
    """
    _, loop, _ = whole_size_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'SIZE' of 'swept' to name a dimension, since 'swept' is "
            "declared with rank 2" in str(error.value))


def test_lfric_kokkos_trans_inherits_the_bounds_grammar_refusal(
        lower_bound_enquiry_target):
    """A bound of an array with an unwritable origin gets that message.

    ``_bounds`` already refuses that declaration, and the enquiry depends on
    the same bounds, so its refusal is passed through rather than paraphrased
    -- a reader gets the sentence that says which rule was broken.
    """
    _, loop, _ = lower_bound_enquiry_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the declared origin of 'swept' to be an integer "
            "expression over named sizes, but found 'pow(nlayers, nlayers)'"
            in str(error.value))
