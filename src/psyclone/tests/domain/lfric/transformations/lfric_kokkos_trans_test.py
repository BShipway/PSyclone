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
from psyclone.domain.lfric import LFRicLoop
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
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


@pytest.fixture(name="target")
def target_fixture(tmp_path):
    """Create the production metadata/body in a minimal LFRic invoke."""
    Config.get().api = "lfric"
    algorithm = tmp_path / "kokkos_test.f90"
    kernel = tmp_path / "moist_dyn_gas_kernel_mod.f90"
    algorithm.write_text(_ALGORITHM, encoding="utf-8")
    kernel.write_text(_KERNEL, encoding="utf-8")
    _, invoke_info = parse(
        str(algorithm), api="lfric", kernel_paths=[str(tmp_path)])
    psy = PSyFactory("lfric", distributed_memory=True).create(invoke_info)
    loop = psy.invokes.invoke_list[0].schedule.walk(LFRicLoop)[0]
    return psy, loop, loop.kernels()[0]


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
    with pytest.raises(TransformationError, match="discontinuous Wtheta"):
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
    with pytest.raises(TransformationError, match="field arguments"):
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
