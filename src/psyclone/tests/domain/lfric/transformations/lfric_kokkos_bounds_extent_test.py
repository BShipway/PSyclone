# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the extents and origins a local declaration states."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _LOCAL_ALGORITHM, _LOCAL_KERNEL, _ZERO_BASED_LOCAL_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.transformations import TransformationError


# The same array asked for all three of its shape enquiries. Each is answered
# from the declaration, and each answer moves with the origin: LBOUND is no
# longer the constant 1 and SIZE is no longer the upper bound.
_ZERO_BASED_ENQUIRY_KERNEL = _ZERO_BASED_LOCAL_KERNEL.replace(
    "    do k = 2, nlayers",
    "    do k = lbound(u_e, 1) + 2, ubound(u_e, 1)").replace(
    "      field_out(map_w3(1) + k - 1) = u_e(k)",
    "      field_out(map_w3(1) + k - 1) = u_e(k) + size(u_e, 1)")


# A local whose shape its declaration does not carry. Every one of the fifteen
# rows the catalogue counts under "explicit bounds" is this: an allocatable
# whose size is stated by an ALLOCATE in the body, over values the kernel
# computes for itself. A scratch size is computed before the launch enters the
# region, so there is nowhere for such a size to come from.
_SHAPELESS_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "    real(kind=r_def), dimension(nlayers) :: swept",
    "    real(kind=r_def), allocatable, dimension(:) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    allocate( swept(nlayers) )\n"
    "    swept(nlayers) = partial(nlayers)").replace(
    "  end subroutine column_solve_code",
    "    deallocate( swept )\n"
    "  end subroutine column_solve_code")


# An extent that is not an integer expression over named sizes. The origin is
# the Fortran default here, so this is the extent grammar's own refusal rather
# than the origin's, asserted apart from it because the two are checked in
# order and the first to fire hides the second. The exponent is a name rather
# than a literal because a literal one is not unrenderable any more: the C
# writer expands `nlayers**2` into `(nlayers * nlayers)`, which is an integer
# expression over named sizes and is accepted. `nlayers**nlayers` still has to
# be written `pow(nlayers, nlayers)`, and the comma is what a `shmem_size`
# argument may not carry.
_UNRENDERABLE_EXTENT_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers**nlayers) :: swept")


# The same declaration with a literal exponent, which is not unrenderable.
# `nlayers**2` was refused as long as the C writer wrote an integer power as
# `pow`; it is written as a product now, and a product over named sizes is
# exactly what a `shmem_size` argument may be. The pair is here so that the
# refusal above is read as being about the comma rather than about the power.
_SQUARED_EXTENT_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers**2) :: swept")


@pytest.fixture(name="zero_based_enquiry_target")
# pylint: disable-next=unused-argument
def zero_based_enquiry_target_fixture(tmp_path,
                                      clear_module_manager_instance):
    """Create an invoke asking all three enquiries of a zero-based local."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _ZERO_BASED_ENQUIRY_KERNEL)


@pytest.fixture(name="unrenderable_extent_target")
# pylint: disable-next=unused-argument
def unrenderable_extent_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel squares in a declared upper bound."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _UNRENDERABLE_EXTENT_KERNEL)


@pytest.fixture(name="squared_extent_target")
# pylint: disable-next=unused-argument
def squared_extent_target_fixture(tmp_path,
                                  clear_module_manager_instance):
    """Create an invoke whose kernel squares a name in a declared extent."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _SQUARED_EXTENT_KERNEL)


@pytest.fixture(name="shapeless_local_target")
# pylint: disable-next=unused-argument
def shapeless_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates a local in its body."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SHAPELESS_LOCAL_KERNEL)


def test_lfric_kokkos_trans_takes_a_literal_extent(literal_local_target):
    """A local declared ``dimension(nlayers,4)`` is captured, not refused.

    This is ``apply_helmholtz_operator_code``'s ``coeff`` in miniature, and
    the reason this widening exists: a literal bound refuses 179 of the 183
    loops the catalogue's ``local-array`` row counts, and that kernel is one
    of them.
    """
    psy, loop, kernel = literal_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("nlayers", "4")
    # The literal is not a kernel argument and must not be asked to be one.
    assert LFRicKokkosTrans._extent_names(
        schedule.symbol_table.lookup("swept")) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t::shmem_size(nlayers, 4)" in cpp
    assert "swept_scratch_t swept(team.team_scratch(0), nlayers, 4);" in cpp
    assert ("using swept_scratch_t = Kokkos::View<double**, "
            "Kokkos::LayoutLeft, ScratchSpace, Unmanaged>;" in cpp)
    # Both indices lose their Fortran base, not just the first.
    assert "swept((k - 1), (1 - 1))" in cpp
    # Nothing about it reaches the ABI, as for any other scratch array.
    assert "swept" not in str(psy.gen)


def test_lfric_kokkos_trans_takes_an_arithmetic_extent(
        arithmetic_local_target):
    """A local declared ``dimension(nlayers+1)`` is captured."""
    _, loop, kernel = arithmetic_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("swept")

    assert LFRicKokkosTrans._extents(symbol) == ("(nlayers + 1)",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t::shmem_size((nlayers + 1))" in cpp
    assert ("swept_scratch_t swept(team.team_scratch(0), (nlayers + 1));"
            in cpp)


def test_lfric_kokkos_trans_takes_an_explicit_lower_bound_of_one(
        explicit_one_local_target):
    """``dimension(1:nlayers)`` is accepted; the rule is about the value.

    A rule written against the source text would refuse this, since it is
    spelt like the ``dimension(0:nlayers-1)`` case that is out of reach. The
    lower bound is rendered and compared instead.
    """
    _, loop, kernel = explicit_one_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("nlayers",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t swept(team.team_scratch(0), nlayers);" in cpp


def test_lfric_kokkos_trans_accepts_a_zero_based_local(
        zero_based_local_target):
    """``dimension(0:nlayers)`` is captured, sized and shifted by its origin.

    This is the shape the catalogue's ``array-bound`` row counts. The View is
    one element longer than the upper bound, because the extent is
    ``ub - lb + 1`` and not ``ub``; and every subscript of it is shifted by
    the declared origin rather than by the Fortran default of 1.

    The shift is emitted even though it is zero. ``u_e(k - 0)`` is the origin
    stated in the generated code, and a subscript that read ``u_e(k)`` would
    be indistinguishable from one the offset had never reached -- which is
    the failure this capability is at risk of, since it is a wrong answer
    rather than a refusal.
    """
    psy, loop, kernel = zero_based_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_e")

    assert LFRicKokkosTrans._extents(symbol) == ("(nlayers + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("0",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "u_e_scratch_t::shmem_size((nlayers + 1))" in cpp
    assert "u_e_scratch_t u_e(team.team_scratch(0), (nlayers + 1));" in cpp
    # Every subscript of it, not only the one the assignment writes.
    assert "u_e((k - 0))" in cpp
    assert "u_e((nlayers - 0))" in cpp
    assert "u_e(((k + 1) - 0))" in cpp
    # A subscript that never met the offset would read exactly this.
    assert "u_e(k)" not in cpp
    # Scratch reaches no interface, as for any other kernel-local array.
    assert "u_e" not in str(psy.gen)


def test_lfric_kokkos_trans_zero_based_bounds_enquiries(
        zero_based_enquiry_target):
    """LBOUND, UBOUND and SIZE of a zero-based array move with its origin.

    Each is answered from the declaration, so each has to be answered from
    *both* declared bounds: ``LBOUND`` is the origin rather than the constant
    1, and ``SIZE`` is the extent rather than the upper bound. Getting either
    from the old assumption gives an answer that is wrong by one.
    """
    _, loop, _ = zero_based_enquiry_target

    cpp = LFRicKokkosTrans().apply(loop)

    # LBOUND(u_e, 1) is 0 and UBOUND(u_e, 1) is nlayers, from the
    # declaration; the loop the kernel wrote them into is the evidence.
    assert "for(k=(0 + 2); k<=nlayers; k+=1)" in cpp
    # SIZE(u_e, 1) is the extent, which is one more than the upper bound.
    assert "(u_e((k - 0)) + (nlayers + 1))" in cpp
    for name in ("LBOUND", "UBOUND", "SIZE", "lbound(", "ubound(", "size(u_e"):
        assert name not in cpp


def test_lfric_kokkos_trans_accepts_a_negative_lower_bound(
        negative_origin_local_target):
    """``dimension(-nlayers:nlayers)`` is sized and shifted symbolically.

    The origin is an expression rather than a literal, which is the case
    where the extent and the origin are genuinely different readings of the
    same declaration: a region that sized the View correctly and shifted its
    subscripts by 1 would index a View of the right size from the wrong
    place.

    The extent is rendered as the writer builds it, ``((nlayers -
    (-nlayers)) + 1)``, which is ``2 * nlayers + 1`` unsimplified; the
    back-end emits an extent verbatim rather than folding it, so what is
    asserted is what is compiled.
    """
    _, loop, kernel = negative_origin_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_e")

    assert LFRicKokkosTrans._extents(symbol) == (
        "((nlayers - (-nlayers)) + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("(-nlayers)",)
    # The origin is named in the generated C++ too, so it has to be reachable
    # there as well as at the launch.
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("u_e_scratch_t u_e(team.team_scratch(0), "
            "((nlayers - (-nlayers)) + 1));" in cpp)
    assert "u_e((k - (-nlayers)))" in cpp
    assert "u_e((nlayers - (-nlayers)))" in cpp


def test_lfric_kokkos_trans_rejects_an_unrenderable_extent(
        unrenderable_extent_target):
    """``dimension(nlayers**nlayers)`` is refused, naming the extent.

    The origin is the Fortran default here, so the refusal is the extent
    grammar's own. The two checks run in order and the first to fire hides
    the second, which is why the origin refusal is asserted over a separate
    declaration rather than over this one.

    The exponent is a name because a literal one is written as a product now
    and is accepted; ``pow`` reaches an extent only through an exponent whose
    value is not known where the launch is written.
    """
    _, loop, _ = unrenderable_extent_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "extents of 'swept'" in str(error.value)
    assert "found 'pow(nlayers, nlayers)'" in str(error.value)


def test_lfric_kokkos_trans_sizes_scratch_from_a_squared_extent(
        squared_extent_target):
    """``dimension(nlayers**2)`` sizes the scratch from a product.

    The extent grammar refuses a call because a `shmem_size` argument is the
    text the launch writes and nothing rewrites it. An integer power with a
    literal exponent is no longer a call, so this declaration crossed from
    the refusal above into the region without the grammar being touched.
    """
    _, loop, kernel = squared_extent_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("(nlayers * nlayers)",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(nlayers * nlayers)" in cpp
    assert "pow(" not in cpp


def test_lfric_kokkos_trans_rejects_an_unrenderable_lower_bound(
        unrenderable_origin_target):
    """A declared origin that is not an integer expression is still refused.

    Widening the origin from "must be 1" to "any integer expression over
    named sizes" is not a widening to anything at all.
    ``dimension(nlayers**nlayers:nlayers)`` is written
    ``pow(nlayers, nlayers)``, which the grammar refuses in an origin exactly
    as it refuses it in an extent -- and here the consequence is a subscript
    shifted by a value the launch cannot evaluate rather than a wrongly sized
    View.
    """
    _, loop, _ = unrenderable_origin_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'swept'" in str(error.value)
    assert "declared origin" in str(error.value)
    assert "found 'pow(nlayers, nlayers)'" in str(error.value)


def test_lfric_kokkos_trans_moves_an_origin_and_an_extent_together(
        lower_bound_local_target):
    """``dimension(0:nlayers-1)`` moves the origin and computes the extent.

    Both answers come out of one reading of one declaration, and this is the
    shape where reading them apart would disagree: the upper bound is itself
    an expression, so an extent taken as ``ub`` and an origin taken as 1 are
    each wrong by one and in opposite directions.
    """
    _, loop, kernel = lower_bound_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("swept")

    assert LFRicKokkosTrans._extents(symbol) == ("((nlayers - 1) + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("0",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("swept_scratch_t swept(team.team_scratch(0), "
            "((nlayers - 1) + 1));" in cpp)
    assert "swept((k - 0))" in cpp


def test_lfric_kokkos_trans_refuses_an_unsizable_expression(
        unsized_expression_target):
    """``dimension(n_moist+1)`` is refused, naming the module constant.

    Accepting arithmetic widened what an extent may be shaped like, not what
    may appear in one: the scratch size is still computed by the launch,
    where only kernel arguments are in scope.
    """
    _, loop, _ = unsized_expression_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "n_moist" in str(error.value)
    assert "kernel-local array 'swept'" in str(error.value)
    assert "kernel argument" in str(error.value)


def test_lfric_kokkos_trans_places_a_local_sized_by_a_division(
        divided_local_target):
    """``dimension((nlayers + 1)/2)`` is carried, and the truncation is said.

    A quotient is the last shape the catalogue's ``local-array`` row counted
    that the extent grammar refused outright, and refusing it was
    conservative rather than correct: Fortran and C++ both truncate an
    integer quotient toward zero, so the extent computed at the launch is the
    extent the kernel declared.

    What neither language defines is an allocation of negative size, which a
    division can reach where subtraction is in the numerator. So the launch
    states the rule it is relying on in a comment and stops a negative extent
    rather than requesting it -- the guard is emitted because this extent
    divides, and no region without a division gains one.
    """
    _, loop, kernel = divided_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_local")

    assert LFRicKokkosTrans._extents(symbol) == ("((nlayers + 1) / 2)",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "u_local_scratch_t::shmem_size(((nlayers + 1) / 2))" in cpp
    assert ("u_local_scratch_t u_local(team.team_scratch(0), "
            "((nlayers + 1) / 2));" in cpp)
    # The guard is over the extent that divides, and it aborts rather than
    # allocating.
    assert "if ((((nlayers + 1) / 2)) < 0) {" in cpp
    assert "Kokkos::abort(" in cpp
    # The array that does not divide gains no guard of its own.
    assert "if ((nlayers) < 0)" not in cpp


def test_lfric_kokkos_trans_rejects_a_shapeless_local(shapeless_local_target):
    """A declaration with no shape is refused, saying whose statement its is.

    The rule reads declarations, and a deferred shape is not one: the message
    says which of the two it is, and points at the statement that carries the
    size instead. What has changed since it was written is that the statement
    is now read. The declaration is rewritten from the ALLOCATE before this
    rule is asked, so a kernel whose extents are values the region holds at
    entry reaches it as an ordinary automatic array and is captured; the rule
    still refuses the declaration it is asked about here, which is the one the
    kernel wrote rather than the one the conversion leaves behind.
    """
    _, loop, kernel = shapeless_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._validate_locals(schedule)

    assert "'swept' to be declared with explicit bounds" in str(error.value)
    assert "a deferred shape" in str(error.value)
    assert "ALLOCATE in the kernel body" in str(error.value)

    # And the loop as a whole is now accepted, the ALLOCATE stating a size
    # the launch can compute where it reserves its scratch.
    LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_reports_an_unwritable_shape(
        unwritable_shape_target):
    """A declared shape the C writer cannot render is refused, not raised.

    ``dimension(max(nlayers,1))`` is what ffsl_flux_z_nirvana_kernel_mod
    declares, and MAX has no C operator: the writer's own failure is a
    ``VisitorError``, which ``validate`` may not raise, so a caller asking
    whether the loop was capturable got an exception of the wrong type from
    inside the backend instead of an answer. The refusal names the array and
    carries the writer's reason.
    """
    _, loop, _ = unwritable_shape_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "declared shape of 'swept' as C" in str(error.value)
