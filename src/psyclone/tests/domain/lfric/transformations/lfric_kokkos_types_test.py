# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosTypesMixin: LFRic kinds on the C ABI."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _LOCAL_ALGORITHM, _LOCAL_KERNEL, _SOLVER_ALGORITHM, _SOLVER_KERNEL,
    _invoke)

from psyclone.configuration import Config
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.symbols import ScalarType
from psyclone.psyir.transformations import TransformationError


# average_w3_to_w0_code's shape: every one of the seven integers LFRic's
# argument ordering supplies -- the layer count, and a dof count, a dof total
# and a dofmap for each of the two function spaces -- declared with no kind at
# all. That is not a contrivance; the kernel is written that way in
# lfric_apps, and it is the whole of what stands between it and the ABI.
#
# Unlike a logical, an integer does cross the ABI as a width, and the default
# integer's width is stated in no kind parameter the precision map could be
# asked about. It is asserted instead, against the compiler that will build
# the generated code: see the assertion this fixture's test looks for.
_DEFAULT_INTEGER_ALGORITHM = """
program kokkos_default_integer_test
  use field_mod, only : field_type
  use average_w3_to_w0_kernel_mod, only : average_w3_to_w0_kernel_type
  implicit none
  type(field_type) :: coarse_field, fine_field
  call invoke(average_w3_to_w0_kernel_type(coarse_field, fine_field))
end program kokkos_default_integer_test
"""


_DEFAULT_INTEGER_KERNEL = """
module average_w3_to_w0_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : r_def
  use fs_continuity_mod, only : w0, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: average_w3_to_w0_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w0) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: average_w3_to_w0_code
  end type average_w3_to_w0_kernel_type
contains
  subroutine average_w3_to_w0_code(nlayers, field_w3, field_w0, &
                                   ndf_w3, undf_w3, map_w3,     &
                                   ndf_w0, undf_w0, map_w0)
    integer, intent(in) :: nlayers, ndf_w3, undf_w3, ndf_w0, undf_w0
    integer, dimension(ndf_w3), intent(in) :: map_w3
    integer, dimension(ndf_w0), intent(in) :: map_w0
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_w3
    real(kind=r_def), dimension(undf_w0), intent(in) :: field_w0
    integer :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w0
        field_w3(map_w3(1) + k) = field_w3(map_w3(1) + k) + &
                                  field_w0(map_w0(df) + k)
      end do
    end do
  end subroutine average_w3_to_w0_code
end module average_w3_to_w0_kernel_mod
"""


# The single-precision fixture's kernel with the kind name taken off its real.
# A real is refused where an integer and a logical are admitted, and the
# asymmetry is deliberate: LFRic names a kind on every real it means -- r_def,
# r_solver, r_single and r_tran are all in use and all different -- so a real
# declared with no kind is more likely an oversight than a default, and the
# default a compiler picks for it is the one width nobody wrote down.
_DEFAULT_REAL_KERNEL = _SOLVER_KERNEL.replace(
    "  use constants_mod, only : i_def, r_single, r_solver",
    "  use constants_mod, only : i_def, r_solver").replace(
    "    real(kind=r_single), intent(in) :: scaling",
    "    real, intent(in) :: scaling")


# The same, casting to a kind the LFRic precision map does not carry. The map
# names the kinds the model computes in; `r_native` is the compiler's own
# default, and there is no width to record for it.
_UNMAPPED_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_native").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = real(swept(k), r_native)")


# A field whose data is integer, and the same kernel over real data beside it.
# LFRic's integer_field_type has field_type's proxy shape and differs in one
# thing, the intrinsic of the data array, so both are instantiated from one
# template: what the generated region does differently for an integer field is
# then read off against a region differing in nothing else. The name is a
# parameter because the two live in one directory and the kernel source is
# reparsed by module name when the transformation resolves the body.
_RATIO_ALGORITHM = """
program kokkos_{name}_test
  use {module}, only : {field}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field}) :: mask_out, mask_in, mask_div
  call invoke({name}_kernel_type(mask_out, mask_in, mask_div))
end program kokkos_{name}_test
"""


_RATIO_KERNEL = """
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, {intrinsic}, gh_write, &
                           gh_read, cell_column
  use constants_mod, only : i_def, i_native, r_def
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field, {intrinsic}, gh_write, w3),              &
         arg_type(gh_field, {intrinsic}, gh_read,  wtheta),          &
         arg_type(gh_field, {intrinsic}, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: {name}_code
  end type {name}_kernel_type
contains
  subroutine {name}_code(nlayers, mask_out, mask_in, mask_div, &
                         ndf_w3, undf_w3, map_w3, &
                         ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    {data}, dimension(undf_w3), intent(inout) :: mask_out
    {data}, dimension(undf_wtheta), intent(in) :: mask_in
    {data}, dimension(undf_wtheta), intent(in) :: mask_div
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        mask_out(map_w3(df) + k) = mask_in(map_wtheta(df) + k) &
                                 / mask_div(map_wtheta(df) + k)
      end do
    end do
  end subroutine {name}_code
end module {name}_kernel_mod
"""


_INTEGER_FIELD_ALGORITHM = _RATIO_ALGORITHM.format(
    name="int_ratio", module="integer_field_mod", field="integer_field_type")


_INTEGER_FIELD_KERNEL = _RATIO_KERNEL.format(
    name="int_ratio", intrinsic="gh_integer", data="integer(kind=i_def)")


_REAL_FIELD_ALGORITHM = _RATIO_ALGORITHM.format(
    name="real_ratio", module="field_mod", field="field_type")


_REAL_FIELD_KERNEL = _RATIO_KERNEL.format(
    name="real_ratio", intrinsic="gh_real", data="real(kind=r_def)")


# A field whose data is integer at a kind the ABI has no width for. i_native
# is a real LFRic kind and is deliberately absent from the precision map
# psyclone.cfg carries, so nothing here is monkeypatched: the element type
# following the argument's intrinsic must not become a licence to write 'int'
# for an integer of any width at all.
_OFF_ABI_ALGORITHM = _RATIO_ALGORITHM.format(
    name="native_ratio", module="integer_field_mod",
    field="integer_field_type")


_OFF_ABI_KERNEL = _RATIO_KERNEL.format(
    name="native_ratio", intrinsic="gh_integer",
    data="integer(kind=i_native)")


# One invoke carrying both kinds of field, which is how LFRic actually uses an
# integer one: a mask or an index array read beside the real data it selects.
# The two field types come from two modules, and the region has to give each
# formal its own element type without disturbing the order the PSy layer
# passes them in.
_MIXED_ALGORITHM = """
program kokkos_mixed_test
  use field_mod, only : field_type
  use integer_field_mod, only : integer_field_type
  use masked_copy_kernel_mod, only : masked_copy_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(integer_field_type) :: mask
  call invoke(masked_copy_kernel_type(out_field, in_field, mask))
end program kokkos_mixed_test
"""


_MIXED_KERNEL = """
module masked_copy_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_integer, &
                           gh_write, gh_read, cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: masked_copy_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field, gh_real,    gh_write, w3),               &
         arg_type(gh_field, gh_real,    gh_read,  w3),               &
         arg_type(gh_field, gh_integer, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: masked_copy_code
  end type masked_copy_kernel_type
contains
  subroutine masked_copy_code(nlayers, field_out, field_in, mask, &
                              ndf_w3, undf_w3, map_w3, &
                              ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(undf_wtheta), intent(in) :: mask
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = field_in(map_w3(df) + k) &
                                  * mask(map_wtheta(df) + k)
      end do
    end do
  end subroutine masked_copy_code
end module masked_copy_kernel_mod
"""


@pytest.fixture(name="integer_field_target")
# pylint: disable-next=unused-argument
def integer_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose every field carries integer data."""
    return _invoke(
        tmp_path, "int_ratio", _INTEGER_FIELD_ALGORITHM,
        _INTEGER_FIELD_KERNEL)


@pytest.fixture(name="real_field_target")
# pylint: disable-next=unused-argument
def real_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create the same invoke over real fields, to read the other against."""
    return _invoke(
        tmp_path, "real_ratio", _REAL_FIELD_ALGORITHM, _REAL_FIELD_KERNEL)


@pytest.fixture(name="off_abi_field_target")
# pylint: disable-next=unused-argument
def off_abi_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose integer fields are of a kind off the ABI."""
    return _invoke(
        tmp_path, "native_ratio", _OFF_ABI_ALGORITHM, _OFF_ABI_KERNEL)


@pytest.fixture(name="mixed_field_target")
# pylint: disable-next=unused-argument
def mixed_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke taking real fields and an integer one together."""
    return _invoke(
        tmp_path, "masked_copy", _MIXED_ALGORITHM, _MIXED_KERNEL)


@pytest.fixture(name="default_integer_target")
# pylint: disable-next=unused-argument
def default_integer_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes seven default-kind integers."""
    return _invoke(
        tmp_path, "average_w3_to_w0", _DEFAULT_INTEGER_ALGORITHM,
        _DEFAULT_INTEGER_KERNEL)


@pytest.fixture(name="default_real_target")
# pylint: disable-next=unused-argument
def default_real_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes a real declared with no kind."""
    return _invoke(
        tmp_path, "scaled_solver", _SOLVER_ALGORITHM, _DEFAULT_REAL_KERNEL)


@pytest.fixture(name="unmapped_cast_kind_target")
# pylint: disable-next=unused-argument
def unmapped_cast_kind_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke casting to a kind the precision map does not carry."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNMAPPED_CAST_KIND_KERNEL)


def test_lfric_kokkos_trans_rejects_unsupported_kind(target):
    """The fixed C ABI must fail closed if a Fortran kind changes.

    The witness is an integer whose declaration states a width rather than a
    kind name -- 'integer*8', as PSyIR records it. Stage 5 admits an integer
    that states neither, reading it as the default kind and asserting that
    width against the compiler; this one is not that, because the declaration
    did say which width it wanted and it is not the ABI's. Reading the two the
    same way would drop the top four bytes of every value in silence, so the
    pair is tested rather than only the half that is admitted.
    """
    _, loop, kernel = target
    schedule = kernel.get_callees()[0]
    schedule.symbol_table.lookup("nlayers").datatype = ScalarType(
        ScalarType.Intrinsic.INTEGER, 8)
    with pytest.raises(TransformationError, match="argument kinds"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_types_kinds_by_width(target, monkeypatch):
    """A kind reaches the ABI on its width, not on being called r_def.

    Transport kernels are written in r_tran and dynamics kernels in r_def,
    which is a distinction about what the model may reconfigure rather than
    about what reaches C. Narrowing r_def here is the same question asked from
    the other side: the name is unchanged, and what the ABI carries follows the
    width rather than the name.
    """
    _, loop, _ = target
    api_config = Config.get().api_conf("lfric")
    narrowed = dict(api_config.precision_map)
    narrowed["r_def"] = 4
    monkeypatch.setattr(api_config, "_precision_map", narrowed)
    cpp = LFRicKokkosTrans().apply(loop)
    assert "float *moist_dyn_gas_data" in cpp
    assert "const float *mr_v_data" in cpp
    assert "const float recip_epsilon" in cpp


def test_lfric_kokkos_trans_fails_closed_on_an_unsupported_width(
        target, monkeypatch):
    """Widening the ABI to float did not make it accept every kind.

    r_quad is a 16-byte real and is in the LFRic precision map, so the map
    resolves it and the C type table then has no entry for it. Asked through
    the same route as the test above, so that the two read as one question
    with two answers.
    """
    _, loop, _ = target
    api_config = Config.get().api_conf("lfric")
    widened = dict(api_config.precision_map)
    widened["r_def"] = widened["r_quad"]
    monkeypatch.setattr(api_config, "_precision_map", widened)
    with pytest.raises(TransformationError, match="argument kinds") as err:
        LFRicKokkosTrans().validate(loop)
    assert "4-byte integer, 4-byte real and 8-byte real" in str(err.value)


def test_lfric_kokkos_trans_accepts_default_kind_integers(
        default_integer_target):
    """Default-kind integers cross as int, at a width the compiler checks.

    All seven of the integers LFRic's argument ordering supplies are declared
    with no kind here, as average_w3_to_w0_code declares them. An integer does
    cross the ABI as a width, so unlike the logical above this one cannot be
    admitted by ignoring the question: it is admitted by asking the compiler
    instead of the precision map, which has no entry to be asked about because
    a default kind is precisely the one with no name to key it by.

    The assertion is that question. It compares storage_size of a default
    literal against storage_size of a c_int one, in the generated interface,
    so a build whose default integer is not c_int fails to compile rather than
    reading four bytes where the Fortran wrote eight.
    """
    psy, loop, _ = default_integer_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int nlayers" in cpp
    assert "const int ndf_w0" in cpp
    assert "const int *map_w0_data" in cpp
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft" in cpp
    for name in ("nlayers", "ndf_w3", "undf_w3", "ndf_w0", "undf_w0"):
        assert f"integer(c_int), value :: {name}" in fortran
    for name in ("map_w3", "map_w0"):
        assert f"integer(c_int), dimension(*), intent(in) :: {name}" in fortran

    # The width is measured rather than assumed, and this is the measurement.
    # The probe is the bare literal, because the kind it is about is the one a
    # literal has when nothing is said about it.
    assert ("integer(kind=merge(4, -1, storage_size(1) == &\n"
            "        storage_size(1_c_int))), parameter :: "
            "assert_kind_default_integer = 0") in fortran
    # constants_mod is asked only for the kind the kernel still names. The
    # default kind is declared nowhere -- being unnamed is what makes it the
    # default -- so importing it by that name would not compile.
    assert "use constants_mod, only : r_def\n" in fortran
    assert "only : default_integer" not in fortran


def test_lfric_kokkos_trans_rejects_a_default_kind_real(default_real_target):
    """A real declared with no kind stays off the ABI.

    The integer above is admitted because LFRic has one default integer and
    means it. It has no default real: r_def, r_solver, r_single and r_tran are
    all in use and all different, so the kind is the whole of what a real
    declaration says about its width. Reading a missing one as 'whatever the
    compiler picks' would put a promotion or a truncation on the ABI silently,
    which is the failure every other refusal here exists to prevent.
    """
    _, loop, _ = default_real_target

    with pytest.raises(TransformationError, match="argument kinds") as error:
        LFRicKokkosTrans().validate(loop)

    assert "'scaling'" in str(error.value)


def test_lfric_kokkos_trans_carries_single_precision_to_c(solver_target):
    """An r_solver kernel reaches C as float on both sides of the ABI.

    The fields are r_solver and the scalar is r_single -- two different 4-byte
    kinds -- so a widening that hardcoded one kind name would not generate
    this. What matters is that neither side promotes: the C++ says float and
    the bind(C) interface says real(c_float), so a single-precision build is
    honoured rather than silently widened to double.
    """
    psy, loop, _ = solver_target
    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "float *field_out_data" in cpp
    assert "const float *field_in_data" in cpp
    assert "const float scaling" in cpp
    assert "Kokkos::View<float*, Kokkos::LayoutLeft" in cpp
    assert "Kokkos::View<const float*, Kokkos::LayoutLeft" in cpp

    # Nothing but these two lines decides what precision the region computes
    # in: the local and the literal cross no interface, so the compiler
    # cannot check them and a promotion here would be silent.
    assert "float scaled;" in cpp
    assert "2.0f" in cpp

    assert "use iso_c_binding, only : c_int, c_float" in fortran
    assert ("real(c_float), dimension(*), intent(inout) :: field_out"
            in fortran)
    assert "real(c_float), dimension(*), intent(in) :: field_in" in fortran
    assert "real(c_float), value :: scaling" in fortran
    assert "c_double" not in fortran

    # The compiler checks the arguments, because the interface names a C kind
    # and the PSy layer names an LFRic one. It cannot check what the body
    # assumed, so the interface says so itself.
    assert "use constants_mod, only : i_def, r_single, r_solver" in fortran
    assert ("storage_size(1.0_r_solver) == &\n        "
            "storage_size(1.0_c_float))), parameter :: assert_kind_r_solver "
            "= 0" in fortran)
    assert ("storage_size(1.0_r_single) == &\n        "
            "storage_size(1.0_c_float))), parameter :: assert_kind_r_single "
            "= 0" in fortran)


def test_lfric_kokkos_trans_leaves_out_a_kind_the_map_does_not_carry(
        unmapped_cast_kind_target):
    """A cast kind with no width recorded is left out, not written in as none.

    ``r_native`` is the compiler's own default rather than one of the widths
    the LFRic precision map names, so there is nothing to record. Storing the
    failed lookup would be worse than leaving it out: the table is keyed by
    kind name, and a declaration may already have resolved the same name.
    """
    _, loop, kernel = unmapped_cast_kind_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    kinds = dict(LFRicKokkosTrans._kind_types(schedule))

    assert "r_native" not in kinds
    assert kinds["r_def"] == "double"
    assert "r_native" not in LFRicKokkosTrans().apply(loop)


def test_lfric_kokkos_trans_accepts_an_integer_field(
        integer_field_target, real_field_target):
    """A field whose data is integer crosses the ABI as a View of int.

    The element type follows the argument, and nothing else about the field
    does: the dofmap, the ndf and undf formals and the launch belong to the
    function space rather than to the intrinsic. They are asserted to be the
    real kernel's to the character, which the two regions being generated
    from one template makes a statement rather than a coincidence.
    """
    psy, loop, _ = integer_field_target
    _, real_loop, _ = real_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)
    real_cpp = LFRicKokkosTrans().apply(real_loop)

    assert "int *mask_out_data," in cpp
    assert "const int *mask_in_data," in cpp
    assert ("Kokkos::View<int*, Kokkos::LayoutLeft, MemorySpace, Unmanaged> "
            "mask_out(mask_out_data, undf_w3);" in cpp)
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_in(mask_in_data, undf_wtheta);" in cpp)
    # The dofmap and the sizes are what they are for a real field.
    assert "const int undf_w3," in cpp
    assert "const int undf_wtheta," in cpp
    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> map_w3(map_w3_data, ndf_w3, ncells);" in cpp)
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp

    # The width the C++ body assumed for i_def, asserted where the generated
    # Fortran and the generated C++ meet. A region carrying integer data and
    # nothing else names no real kind at all, so this is the only assertion
    # it writes and c_double never reaches its 'use' line.
    assert "use iso_c_binding, only : c_int" in fortran
    assert "use constants_mod, only : i_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert "c_double" not in fortran
    assert "integer(c_int), dimension(*), intent(inout) :: mask_out" in fortran
    assert "call int_ratio_kokkos(" in fortran
    assert "call mask_out_proxy%set_dirty()" in fortran

    # Everything the intrinsic does not decide is the real kernel's: putting
    # the element type and the kernel's name back gives that region exactly.
    assert real_cpp.replace("double", "int").replace(
        "real_ratio", "int_ratio") == cpp
    assert "double" not in cpp


def test_lfric_kokkos_trans_accepts_a_mixed_field_kernel(mixed_field_target):
    """Each field formal takes its own element type, in the order given.

    The intrinsic is per argument rather than per kernel, and a real field
    beside an integer one is what separates the two readings. The order is
    asserted as well as the types, because the generated signature and the
    actuals the PSy layer passes carry their correspondence in nothing but
    their shared indices: a formal given the wrong element type is a region
    that compiles and reads the wrong storage.
    """
    psy, loop, _ = mixed_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert ("    double *field_out_data,\n"
            "    const double *field_in_data,\n"
            "    const int *mask_data,\n" in cpp)
    assert ("Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged> field_out(field_out_data, undf_w3);" in cpp)
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask(mask_data, undf_wtheta);" in cpp)

    # One interface carrying both widths, and both kinds asserted on it.
    assert "use iso_c_binding, only : c_int, c_double" in fortran
    assert "use constants_mod, only : i_def, r_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert ("storage_size(1.0_r_def) == &\n        "
            "storage_size(1.0_c_double))), parameter :: assert_kind_r_def = 0"
            in fortran)
    assert ("real(c_double), dimension(*), intent(inout) :: field_out\n"
            "    real(c_double), dimension(*), intent(in) :: field_in\n"
            "    integer(c_int), dimension(*), intent(in) :: mask\n"
            in fortran)
    assert ("call masked_copy_kokkos(nlayers_out_field, out_field_data, "
            "in_field_data, mask_data, ndf_w3, undf_w3, map_w3, ndf_wtheta, "
            "undf_wtheta, map_wtheta, loop0_stop)" in fortran)


def test_lfric_kokkos_trans_integer_field_arithmetic_is_integer(
        integer_field_target, real_field_target):
    """A quotient of two integer field values is an integer quotient.

    C++ takes '/' from its operands, so this is a statement about the Views
    the region declares rather than about the expression: the writer emits
    the same text for both kernels and the element type is the whole of what
    makes one of them truncate. Worth asserting because a promotion anywhere
    on the way -- a cast written round a field read, a double View over
    integer storage -- would change the answer rather than fail to compile.
    """
    _, loop, _ = integer_field_target
    _, real_loop, _ = real_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    real_cpp = LFRicKokkosTrans().apply(real_loop)

    quotient = ("(mask_in(((map_wtheta((df - 1), cell) + k) - 1)) / "
                "mask_div(((map_wtheta((df - 1), cell) + k) - 1)))")
    assert quotient in cpp
    assert quotient in real_cpp
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_div(mask_div_data, undf_wtheta);" in cpp)
    assert ("Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_div(mask_div_data, undf_wtheta);" in real_cpp)
    # Nothing widens the operands on the way to the division.
    assert "double" not in cpp
    assert "float" not in cpp
    assert "static_cast" not in cpp


def test_lfric_kokkos_trans_refuses_a_field_kind_not_on_the_abi(
        off_abi_field_target):
    """Following the argument's intrinsic did not retire the width check.

    ``i_native`` is an LFRic integer kind that ``psyclone.cfg``'s precision
    map does not carry, so there is no width to put on the ABI. Writing
    ``int`` for it because the argument is an integer field would drop or
    invent bytes without saying so, so it is refused naming the field and the
    kind. Nothing here is monkeypatched: the kind really is absent.
    """
    _, loop, _ = off_abi_field_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'mask_out' has 'i_native'" in str(error.value)
    assert "4-byte integer, 4-byte real and 8-byte real" in str(error.value)
