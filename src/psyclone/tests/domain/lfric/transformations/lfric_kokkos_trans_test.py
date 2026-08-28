# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately narrow LFRic-to-Kokkos transformation."""

# pylint: disable=protected-access

import pytest

from psyclone.configuration import Config
from psyclone.core import AccessType
from psyclone.domain.lfric import LFRicLoop
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.parse import ModuleManager
from psyclone.parse.algorithm import parse
from psyclone.psyGen import PSyFactory
from psyclone.psyir.nodes import CodeBlock, Literal
from psyclone.psyir.symbols import ScalarType
from psyclone.psyir.transformations import TransformationError


_ALGORITHM = """
program kokkos_test
  use field_mod, only : field_type
  use moist_dyn_gas_kernel_mod, only : moist_dyn_gas_kernel_type
  implicit none
  type(field_type) :: moist_dyn, mr
  call invoke(moist_dyn_gas_kernel_type(moist_dyn, mr))
end program kokkos_test
"""


_KERNEL = """
module moist_dyn_gas_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : wtheta
  use kernel_mod, only : kernel_type
  use planet_config_mod, only : recip_epsilon
  implicit none
  type, public, extends(kernel_type) :: moist_dyn_gas_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, wtheta),              &
         arg_type(gh_field, gh_real, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: moist_dyn_gas_code
  end type moist_dyn_gas_kernel_type
contains
  subroutine moist_dyn_gas_code(nlayers, moist_dyn_gas, mr_v, &
                                ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_wtheta), intent(inout) :: moist_dyn_gas
    real(kind=r_def), dimension(undf_wtheta), intent(in) :: mr_v
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    real(kind=r_def) :: mr_v_at_dof
    do k = 0, nlayers - 1
      do df = 1, ndf_wtheta
        mr_v_at_dof = mr_v(map_wtheta(df) + k)
        moist_dyn_gas(map_wtheta(df) + k) = &
            1.0_r_def + recip_epsilon * mr_v_at_dof
      end do
    end do
  end subroutine moist_dyn_gas_code
end module moist_dyn_gas_kernel_mod
"""


_SECOND_ALGORITHM = """
program kokkos_second_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling))
end program kokkos_second_test
"""


# A second, deliberately unrelated shape: a different region name, an r_tran
# scalar argument, two function spaces and so two dofmaps, and no module
# constant. Nothing about it is special-cased, so it only generates if the
# transformation really does derive its contract from the kernel.
_SECOND_KERNEL = """
module scaled_copy_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, gh_write, &
                           gh_read, cell_column
  use constants_mod, only : i_def, r_def, r_tran
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: scaled_copy_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real, gh_write, w3),                 &
         arg_type(gh_field,  gh_real, gh_read,  wtheta),             &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: scaled_copy_code
  end type scaled_copy_kernel_type
contains
  subroutine scaled_copy_code(nlayers, field_out, field_in, scaling, &
                              ndf_w3, undf_w3, map_w3, &
                              ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_wtheta), intent(in) :: field_in
    real(kind=r_tran), intent(in) :: scaling
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = &
            scaling * field_in(map_wtheta(df) + k)
      end do
    end do
  end subroutine scaled_copy_code
end module scaled_copy_kernel_mod
"""


# The same kernel twice in one invoke, so that the captured loop has a sibling
# still counting with the PSy layer's cell counter.
_PAIRED_ALGORITHM = """
program kokkos_paired_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  implicit none
  type(field_type) :: out_field, other_field, in_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling),   &
              scaled_copy_kernel_type(other_field, in_field, scaling))
end program kokkos_paired_test
"""


# The kind of a module constant is stated only in its own module, so the
# transformation reads it there rather than guessing. A real run reaches it
# because generate() puts the kernel search path on the ModuleManager; these
# tests drive parse() and PSyFactory directly, so they say so themselves.
_PLANET_CONFIG = """
module planet_config_mod
  use constants_mod, only : r_def
  implicit none
  real(kind=r_def), public, protected :: recip_epsilon = 1.0_r_def
end module planet_config_mod
"""


def _invoke(tmp_path, name, algorithm_source, kernel_source):
    """Build a one-invoke LFRic PSy layer from the given sources."""
    Config.get().api = "lfric"
    algorithm = tmp_path / f"{name}_alg.f90"
    kernel = tmp_path / f"{name}_kernel_mod.f90"
    algorithm.write_text(algorithm_source, encoding="utf-8")
    kernel.write_text(kernel_source, encoding="utf-8")
    (tmp_path / "planet_config_mod.f90").write_text(
        _PLANET_CONFIG, encoding="utf-8")
    ModuleManager.get().add_search_path(str(tmp_path))
    _, invoke_info = parse(
        str(algorithm), api="lfric", kernel_paths=[str(tmp_path)])
    psy = PSyFactory("lfric", distributed_memory=True).create(invoke_info)
    loop = psy.invokes.invoke_list[0].schedule.walk(LFRicLoop)[0]
    return psy, loop, loop.kernels()[0]


# pylint: disable-next=unused-argument
@pytest.fixture(name="target")
def target_fixture(tmp_path, clear_module_manager_instance):
    """Create the production metadata/body in a minimal LFRic invoke."""
    return _invoke(tmp_path, "moist_dyn_gas", _ALGORITHM, _KERNEL)


# pylint: disable-next=unused-argument
@pytest.fixture(name="second_target")
def second_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an unrelated supported kernel in a minimal LFRic invoke."""
    return _invoke(
        tmp_path, "scaled_copy", _SECOND_ALGORITHM, _SECOND_KERNEL)


# pylint: disable-next=unused-argument
@pytest.fixture(name="paired_target")
def paired_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose captured loop has a cell loop beside it."""
    return _invoke(
        tmp_path, "scaled_copy", _PAIRED_ALGORITHM, _SECOND_KERNEL)


def test_lfric_kokkos_trans_splices_one_loop(target):
    """The selected loop becomes one typed call and retains halo updates."""
    psy, loop, _ = target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void moist_dyn_gas_kokkos(' in cpp
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp
    assert "map_wtheta((df - 1), cell)" in cpp
    assert "const int *map_wtheta_data" in cpp
    assert "map_wtheta(map_wtheta_data, ndf_wtheta, ncells)" in cpp
    assert "const double recip_epsilon" in cpp

    assert "subroutine moist_dyn_gas_kokkos(" in fortran
    assert "bind(C)" in fortran
    assert "dimension(*), intent(inout) :: moist_dyn_gas" in fortran
    assert "call moist_dyn_gas_kokkos(" in fortran
    assert "map_wtheta, loop0_stop, recip_epsilon)" in fortran
    assert "call moist_dyn_proxy%set_dirty()" in fortran
    assert fortran.index("call moist_dyn_gas_kokkos(") < fortran.index(
        "call moist_dyn_proxy%set_dirty()")
    assert "call moist_dyn_gas_code(" not in fortran
    assert str(psy.gen) == fortran


def test_lfric_kokkos_trans_derives_a_second_kernel_contract(second_target):
    """Nothing about the region is specific to the first kernel captured."""
    psy, loop, _ = second_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # The region is named after the kernel, and its r_tran scalar is on the
    # ABI as a double even though the kernel the transformation was written
    # against has no scalar at all and never mentions r_tran.
    assert 'extern "C" void scaled_copy_kokkos(' in cpp
    assert "const double scaling" in cpp

    # Two function spaces, so two dofmaps, each given the cell extent the
    # PSy layer would otherwise have sliced away.
    assert "map_w3(map_w3_data, ndf_w3, ncells)" in cpp
    assert "map_wtheta(map_wtheta_data, ndf_wtheta, ncells)" in cpp
    assert "field_out(((map_w3((df - 1), cell) + k) - 1)) = " in cpp
    assert "(scaling * field_in(((map_wtheta((df - 1), cell) + k) - 1)))" \
        in cpp

    assert "subroutine scaled_copy_kokkos(" in fortran
    assert "real(c_double), value :: scaling" in fortran
    assert "dimension(*), intent(inout) :: field_out" in fortran
    assert "call scaled_copy_kokkos(" in fortran
    # No module constant, so the cell count is the last actual argument.
    assert "map_wtheta, loop0_stop)" in fortran
    assert "call scaled_copy_code(" not in fortran


def test_lfric_kokkos_trans_drops_an_orphaned_counter(second_target):
    """Capturing an invoke's only cell loop undeclares its counter.

    LFRic compiles the PSy layer with -Werror=unused-variable, so a counter
    left declared by the loop that has just been replaced is a build failure
    rather than untidiness.

    """
    psy, loop, _ = second_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "call scaled_copy_kokkos(" in fortran
    assert ":: cell" not in fortran


def test_lfric_kokkos_trans_keeps_a_shared_counter(paired_target):
    """A sibling cell loop still counting keeps the declaration."""
    psy, loop, _ = paired_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # A Loop holds its counter as an attribute rather than as a Reference, so
    # the surviving loop is only visible to a check that asks the loops.
    assert "call scaled_copy_kokkos(" in fortran
    assert "call scaled_copy_code(" in fortran
    assert "integer(kind=i_def) :: cell" in fortran


def test_untransformed_invoke_can_be_generated_repeatedly(target):
    """Preparing a temporary generation copy must not mark the source tree."""
    psy, _, _ = target
    first = str(psy.gen)
    assert str(psy.gen) == first


def test_lfric_kokkos_trans_rejects_wrong_node():
    """Only an LFRic loop is a valid capture boundary."""
    with pytest.raises(TransformationError, match="LFRicLoop"):
        LFRicKokkosTrans().validate(CodeBlock(
            [], structure=CodeBlock.Structure.STATEMENT))


def test_lfric_kokkos_trans_rejects_halo_depth(target):
    """The prototype may not launch over a halo or redundant region."""
    _, loop, _ = target
    loop._upper_bound_halo_depth = Literal(
        "1", ScalarType(ScalarType.Intrinsic.INTEGER,
                        ScalarType.Precision.UNDEFINED))
    with pytest.raises(TransformationError, match="halo depth"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_continuous_write(target):
    """A continuous-space write would require colouring or atomics."""
    _, loop, kernel = target
    kernel.arguments.args[0].function_space._orig_name = "w0"
    with pytest.raises(TransformationError, match="discontinuous space"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_permits_a_continuous_read(target):
    """Only the written spaces decide whether cells may run in parallel."""
    _, loop, kernel = target
    kernel.arguments.args[1].function_space._orig_name = "w0"
    LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_quadrature(target):
    """Quadrature arrays are outside the first region contract."""
    _, loop, kernel = target
    kernel._basis_required = True
    kernel._qr_rules["gh_quadrature_xyoz"] = object()
    with pytest.raises(TransformationError, match="quadrature"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_operator(target):
    """Operator arguments are outside the first region contract."""
    _, loop, kernel = target
    kernel.arguments.args[1]._argument_type = "gh_operator"
    with pytest.raises(TransformationError, match="gh_operator"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_incremented_field(target):
    """An incremented field needs colouring or atomics, not a bare launch."""
    _, loop, kernel = target
    kernel.arguments.args[0]._access = AccessType.INC
    with pytest.raises(TransformationError, match="colouring or atomics"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_stencil(target):
    """Stencil storage and halo requirements are not silently captured."""
    _, loop, kernel = target
    kernel.arguments.args[1].stencil = {"type": "xory1d"}
    with pytest.raises(TransformationError, match="stencil"):
        LFRicKokkosTrans().validate(loop)


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


def test_lfric_kokkos_trans_rejects_unsupported_kind(target):
    """The fixed C ABI must fail closed if a Fortran kind changes."""
    _, loop, kernel = target
    schedule = kernel.get_callees()[0]
    schedule.symbol_table.lookup("nlayers").datatype = (
        ScalarType.integer_type())
    with pytest.raises(TransformationError, match="argument kinds"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_types_kinds_by_width(target, monkeypatch):
    """A kind reaches the ABI on its width, not on being called r_def.

    Transport kernels are written in r_tran and dynamics kernels in r_def,
    which is a distinction about what the model may reconfigure rather than
    about what reaches C. Narrowing r_def here is the same question asked
    from the other side: the name is unchanged and the capture must stop.
    """
    _, loop, _ = target
    api_config = Config.get().api_conf("lfric")
    narrowed = dict(api_config.precision_map)
    narrowed["r_def"] = 4
    monkeypatch.setattr(api_config, "_precision_map", narrowed)
    with pytest.raises(TransformationError, match="argument kinds"):
        LFRicKokkosTrans().validate(loop)
