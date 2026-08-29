# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately small Kokkos backend."""

from dataclasses import FrozenInstanceError, replace

import pytest

from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosView, KokkosWriter)
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import CodeBlock, KernelSchedule, Routine


def _kernel_schedule():
    """Create the PSyIR body used by the first LFRic Kokkos region."""
    source = """
subroutine moist_dyn_gas_code(nlayers, moist_dyn_gas, mr_v, &
                              ndf_wtheta, undf_wtheta, map_wtheta)
  use constants_mod, only : i_def, r_def
  use planet_config_mod, only : recip_epsilon
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
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "moist_dyn_gas_code", symbol_table=symbol_table, children=children)


def _region():
    """Return the explicit launch and argument contract for the test body."""
    return KokkosRegion(
        name="moist_dyn_gas_kokkos",
        schedule=_kernel_schedule(),
        cell_count="ncells",
        arguments=(
            KokkosScalar("nlayers", "int"),
            KokkosView(
                "moist_dyn_gas", "moist_dyn_gas_data", "double",
                ("undf_wtheta",), index_offsets=(1,)),
            KokkosView(
                "mr_v", "mr_v_data", "double", ("undf_wtheta",),
                index_offsets=(1,), read_only=True, random_access=True),
            KokkosScalar("ndf_wtheta", "int"),
            KokkosScalar("undf_wtheta", "int"),
            KokkosView(
                "map_wtheta", "map_wtheta_data", "int",
                ("ndf_wtheta", "ncells"), index_offsets=(1,),
                extra_indices=("cell",), read_only=True,
                random_access=True),
            KokkosScalar("ncells", "int"),
            KokkosScalar("recip_epsilon", "double"),
        ))


def test_kokkos_writer_translation_unit():
    """The writer emits the complete CPU translation-unit boundary."""
    code = KokkosWriter()(_region())

    assert code.startswith("#include <Kokkos_Core.hpp>\n")
    assert 'extern "C" void moist_dyn_gas_kokkos(' in code
    assert "double *moist_dyn_gas_data" in code
    assert "const double *mr_v_data" in code
    assert "const int *map_wtheta_data" in code
    assert "const int ncells" in code
    assert "const double recip_epsilon" in code

    assert "using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;" in code
    assert "Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, " \
        "Unmanaged> moist_dyn_gas(moist_dyn_gas_data, undf_wtheta);" in code
    assert "Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> mr_v(mr_v_data, undf_wtheta);" in code
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> map_wtheta(map_wtheta_data, ndf_wtheta, ncells);" in code
    assert 'Kokkos::parallel_for("moist_dyn_gas_kokkos"' in code
    assert "Kokkos::RangePolicy<>(0, ncells)" in code
    assert "KOKKOS_LAMBDA(const int cell)" in code
    assert "Kokkos::fence();" in code

    # A View constructed with a string label owns an allocation. No View may be
    # given one; the generated string literals are the parallel-for label and
    # the uninitialised-runtime message, neither of which constructs a View.
    assert 'moist_dyn_gas("' not in code
    assert 'mr_v("' not in code
    assert 'map_wtheta("' not in code


def test_kokkos_writer_requires_an_initialised_runtime():
    """A region entered before Kokkos::initialize() stops, and says which.

    Kokkos itself does not treat this as an error: the region warns on stderr
    and then runs correctly but single-threaded. Both the build and any
    answer-based test would therefore pass, leaving only lost performance to
    give it away, so the generated code has to raise the alarm itself.
    """
    code = KokkosWriter()(_region())

    guard = code.index("if (!Kokkos::is_initialized()) {")
    assert 'Kokkos::abort("moist_dyn_gas_kokkos: Kokkos region entered ' \
        'before Kokkos::initialize()");' in code

    # Before the parallel dispatch, or the diagnosis arrives after the damage.
    assert guard < code.index("Kokkos::parallel_for(")


def test_kokkos_writer_indices_and_imported_constant():
    """Fortran indices become zero-based View calls and globals are scalars."""
    code = KokkosWriter()(_region())

    assert "for(k=0; k<=(nlayers - 1); k+=1)" in code
    assert "for(df=1; df<=ndf_wtheta; df+=1)" in code
    assert "map_wtheta((df - 1), cell)" in code
    expected_index = "((map_wtheta((df - 1), cell) + k) - 1)"
    assert f"mr_v_at_dof = mr_v({expected_index});" in code
    assert f"moist_dyn_gas({expected_index}) = " in code
    assert "(1.0 + (recip_epsilon * mr_v_at_dof))" in code


def test_kokkos_region_descriptions_are_immutable():
    """Captured pre-lowering semantics cannot drift during generation."""
    region = _region()
    with pytest.raises(FrozenInstanceError):
        region.cell_count = "other"
    with pytest.raises(FrozenInstanceError):
        region.arguments[1].read_only = True


def test_kokkos_writer_rejects_managed_views():
    """Generated regions may not allocate or own LFRic storage."""
    region = _region()
    arguments = list(region.arguments)
    arguments[1] = replace(arguments[1], managed=True)
    with pytest.raises(ValueError, match="must be unmanaged"):
        KokkosWriter()(replace(region, arguments=tuple(arguments)))


def test_kokkos_writer_rejects_unsupported_type():
    """The prototype fails closed when no C ABI mapping exists."""
    region = _region()
    arguments = region.arguments + (KokkosScalar("unsupported", "complex"),)
    with pytest.raises(TypeError, match="unsupported C type 'complex'"):
        KokkosWriter()(replace(region, arguments=arguments))


def test_kokkos_writer_rejects_a_widened_neighbour_type():
    """Admitting float did not admit every C type that resembles one.

    ``complex`` is refused above because nothing in the prototype could ever
    produce it; ``long`` is the harder case, being an ordinary C type of a
    kind LFRic really has, and it stays refused because the ABI names what it
    carries rather than excluding what it does not.
    """
    region = _region()
    arguments = region.arguments + (KokkosScalar("wide_count", "long"),)
    with pytest.raises(TypeError, match="unsupported C type 'long'"):
        KokkosWriter()(replace(region, arguments=arguments))


def test_kokkos_writer_rejects_codeblocks():
    """Opaque Fortran cannot silently enter generated Kokkos code."""
    region = _region()
    region.schedule.addchild(
        CodeBlock([], structure=CodeBlock.Structure.STATEMENT))
    with pytest.raises(ValueError, match="CodeBlock"):
        KokkosWriter()(region)
