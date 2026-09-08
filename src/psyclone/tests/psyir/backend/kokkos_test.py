# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately small Kokkos backend."""

# pylint: disable=protected-access

from dataclasses import FrozenInstanceError, replace

import pytest

from psyclone.psyir.backend.kokkos import (
    KokkosConstant, KokkosRegion, KokkosScalar, KokkosScratch, KokkosView,
    KokkosWriter, extent_names, is_extent, is_offset)
from psyclone.psyir.backend.kokkos_launch import (
    hierarchical_launch, range_launch, team_launch)
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import (
    Assignment, CodeBlock, IntrinsicCall, KernelSchedule, Literal, Loop,
    Reference, Routine)
from psyclone.psyir.symbols import (
    ArgumentInterface, ArrayType, DataSymbol, ScalarType)


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


def _region(cell_count="ncells"):
    """Return the explicit launch and argument contract for the test body.

    :param str cell_count: the formal the launch is bounded by, which is also
        the last extent of every per-cell View. It is a parameter because the
        count a region covers is whatever bound the loop it came from
        carried; the default is only the commonest of those.

    :returns: the region the writer is asked to generate.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    return KokkosRegion(
        name="moist_dyn_gas_kokkos",
        schedule=_kernel_schedule(),
        cell_count=cell_count,
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
                ("ndf_wtheta", cell_count), index_offsets=(1,),
                extra_indices=("cell",), read_only=True,
                random_access=True),
            KokkosScalar(cell_count, "int"),
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


def test_kokkos_launch_takes_an_upper_bound_formal():
    """Every launch shape is bounded by the region's count, whatever it is.

    A region covers the cells its loop covered, and an LFRic loop is not
    always bounded by the owned cells: an operator is assembled redundantly
    into the first halo, and a dof loop may run to the last annexed dof. The
    count therefore crosses the ABI as a formal the launch reads rather than
    as anything the generated source names for itself, and every per-cell
    View is sized by that same formal -- a View still sliced by the owned
    count would be read past its end by the very cells the wider bound added.

    ``ncells`` is the commonest bound and so the one every other test here
    uses; asking for a different name is what tells the two apart.
    """
    bound = "last_halo_cell"

    code = KokkosWriter()(_region(cell_count=bound))

    assert f"const int {bound}" in code
    assert f"Kokkos::RangePolicy<>(0, {bound})" in code
    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            f"ReadOnly> map_wtheta(map_wtheta_data, ndf_wtheta, {bound});"
            in code)
    assert "ncells" not in code

    # The two team shapes read the same formal, in the league they size and
    # in the guard that stops a rank with no cell of its own.
    flat = team_launch(_scratch_region(cell_count=bound), "", "")
    assert f"if (cell >= {bound}) {{" in flat
    assert (f"const int league_size = ({bound} + team_size - 1) / team_size;"
            in flat)
    assert "ncells" not in flat

    hierarchical = hierarchical_launch(_level_region(cell_count=bound), "", "")
    assert f"TeamPolicy({bound}, Kokkos::AUTO)" in hierarchical
    assert "ncells" not in hierarchical


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


def test_kokkos_writer_accepts_a_bool_argument():
    """``bool`` is on the ABI, and is written as an ordinary scalar.

    It is the one supported type carrying no width, which is what lets the
    driving transformation admit a logical whatever ``l_def`` measures. The
    writer needs to know nothing about that: a ``bool`` scalar is declared and
    passed like an ``int`` one, and the conversion happens on the Fortran
    side.
    """
    region = _region()
    arguments = region.arguments + (KokkosScalar("flag", "bool"),)

    code = KokkosWriter()(replace(region, arguments=arguments))

    assert "const bool flag" in code


def test_kokkos_writer_rejects_codeblocks():
    """Opaque Fortran cannot silently enter generated Kokkos code."""
    region = _region()
    region.schedule.addchild(
        CodeBlock([], structure=CodeBlock.Structure.STATEMENT))
    with pytest.raises(ValueError, match="CodeBlock"):
        KokkosWriter()(region)


def test_kokkos_writer_generates_locals_and_literals_at_their_kind():
    """A described kind decides the width of a local and of a literal.

    Nothing outside the generated file constrains either: the local crosses no
    interface and the literal is not an argument, so without this the C
    writer's own default would silently promote a single-precision body to
    double.
    """
    region = replace(_region(), kind_types=(("i_def", "int"),
                                            ("r_def", "float")))
    cpp = KokkosWriter()(region)
    assert "float mr_v_at_dof;" in cpp
    assert "1.0f" in cpp
    assert "int k;" in cpp
    # The table governs the body alone. Each argument still carries its own
    # C type, so a region whose kinds and whose ABI disagree generates the
    # disagreement rather than hiding it.
    assert "double *moist_dyn_gas_data" in cpp


def test_kokkos_writer_leaves_an_undescribed_kind_to_the_c_writer():
    """An empty kind table generates exactly what it did before it existed."""
    assert KokkosWriter()(_region()) == KokkosWriter()(
        replace(_region(), kind_types=()))
    cpp = KokkosWriter()(_region())
    assert "double mr_v_at_dof;" in cpp
    assert "1.0" in cpp and "1.0f" not in cpp


def test_kokkos_writer_rejects_an_invalid_kind_name():
    """A kind name reaches generated C++ verbatim, so it must be one."""
    region = replace(_region(), kind_types=(("r_def kind", "double"),))
    with pytest.raises(ValueError, match="'r_def kind' is not a C\\+\\+"):
        KokkosWriter()(region)


def test_kokkos_writer_rejects_an_unsupported_kind_type():
    """The kind table fails closed on the same list the ABI does."""
    region = replace(_region(), kind_types=(("r_quad", "long double"),))
    with pytest.raises(TypeError, match="unsupported C type 'long double'"):
        KokkosWriter()(region)


def test_kokkos_writer_declares_an_array_local_at_its_element_kind():
    """A local array's kind is its element's, as it is for a kernel formal.

    ``gen_declaration`` is reached through ``gen_local_variable`` for every
    automatic symbol, so it is exercised here directly: a region whose body
    indexed such an array would be refused later by ``arrayreference_node``,
    which has no View for it, and the declaration would never be seen.
    """
    source = """
subroutine local_array_code(nlayers)
  use constants_mod, only : i_def, r_def
  integer(kind=i_def), intent(in) :: nlayers
  real(kind=r_def), dimension(3) :: column
  integer(kind=i_def) :: k
  do k = 1, 3
    nlayers = nlayers
  end do
end subroutine local_array_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    writer = KokkosWriter()
    writer._kind_types = {"r_def": "float", "i_def": "int"}
    declarations = [
        writer.gen_declaration(symbol)
        for symbol in routine.symbol_table.automatic_datasymbols]
    assert "float * restrict column" in declarations
    assert "int k" in declarations


def _scratch_schedule():
    """Create a two-sweep body carrying its state in kernel-local arrays.

    This is ``tri_solve_code``'s shape in miniature: a forward sweep filling
    two automatic arrays sized by ``nlayers``, and a backward sweep reading
    them out. Nothing smaller exercises scratch, because an array that is
    written and read within one sweep would not need to survive it.
    """
    source = """
subroutine tri_solve_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only : i_def, r_double
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
  real(kind=r_double), dimension(nlayers) :: x_new, tri_plus_new
  do k = 1, nlayers
    x_new(k) = x(map(1) + k - 1)
    tri_plus_new(k) = x_new(k) * 2.0_r_double
  end do
  do k = nlayers, 1, -1
    y(map(1) + k - 1) = tri_plus_new(k)
  end do
end subroutine tri_solve_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "tri_solve_code", symbol_table=symbol_table, children=children)


def _scratch_region(**overrides):
    """Return a region whose kernel-local arrays live in team scratch."""
    region = KokkosRegion(
        name="tri_solve_kokkos",
        schedule=_scratch_schedule(),
        cell_count="ncells",
        arguments=(
            KokkosScalar("nlayers", "int"),
            KokkosView("y", "y_data", "double", ("undf",), index_offsets=(1,)),
            KokkosView(
                "x", "x_data", "double", ("undf",), index_offsets=(1,),
                read_only=True, random_access=True),
            KokkosScalar("ndf", "int"),
            KokkosScalar("undf", "int"),
            KokkosView(
                "map", "map_data", "int", ("ndf", "ncells"),
                index_offsets=(1,), extra_indices=("cell",), read_only=True,
                random_access=True),
            KokkosScalar("ncells", "int"),
        ),
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(
            KokkosScratch("x_new", "double", ("nlayers",), index_offsets=(1,)),
            KokkosScratch(
                "tri_plus_new", "double", ("nlayers",), index_offsets=(1,)),
        ))
    return replace(region, **overrides) if overrides else region


def _level_schedule():
    """Create a body with one parallel level loop and one boundary write.

    This is the shape of the five regions the hierarchical launch takes, in
    miniature: a scalar the whole team computes redundantly, a level loop
    whose iterations are independent, and a single-element write outside it.
    Nothing smaller exercises all three of the statement kinds the writer
    renders differently, since a body with only a loop would never need
    ``Kokkos::single``.
    """
    source = """
subroutine inject_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only: r_double, i_def
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
  real(kind=r_double) :: scale
  scale = 0.5_r_double
  do k = 1, nlayers
    y(map(1) + k - 1) = scale * x(map(1) + k - 1)
  end do
  y(map(1) + nlayers) = x(map(1) + nlayers)
end subroutine inject_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "inject_code", symbol_table=symbol_table, children=children)


def _level_region(**overrides):
    """Return a region whose level loop is spread over the team.

    Its ABI is ``_scratch_region``'s, so that the two shapes differ in the
    launch and not in what they are handed. The parallel loop is taken from
    the schedule the region itself carries: the writer matches the loops by
    identity, so a node from a second parse of the same source would be
    refused.
    """
    schedule = _level_schedule()
    region = KokkosRegion(
        name="inject_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_scratch_region().arguments,
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(),
        parallel_loops=(schedule.walk(Loop)[0],))
    return replace(region, **overrides) if overrides else region


def test_kokkos_writer_places_locals_in_team_scratch():
    """A region with kernel-local arrays launches over teams, not a range."""
    code = KokkosWriter()(_scratch_region())

    assert "using TeamPolicy = Kokkos::TeamPolicy<>;" in code
    assert "using TeamMember = TeamPolicy::member_type;" in code
    assert "using ScratchSpace = " \
        "Kokkos::DefaultExecutionSpace::scratch_memory_space;" in code
    assert "Kokkos::RangePolicy<>" not in code

    # One cell per team rank, and the tail of the last team does nothing: the
    # league is sized by rounding up, so without this the body would run for
    # cells past the end of every View.
    assert "const int cell = team.league_rank() * team.team_size() + rank;" \
        in code
    assert "if (cell >= ncells) {" in code

    # The team the backend recommends for this functor, not the largest the
    # scratch allows: on OpenMP the recommendation is one thread, so the
    # leagues rather than the ranks carry the parallelism. The probe policy
    # carries the scratch request, since a backend other than OpenMP may
    # answer differently for a launch that asks for none.
    probe = code.index("TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO)")
    assert ".set_scratch_size(0, Kokkos::PerThread(scratch_bytes));" in \
        code[probe:]
    assert "const int team_size = probe.team_size_recommended(body, " \
        "Kokkos::ParallelForTag());" in code
    assert "const int league_size = (ncells + team_size - 1) / team_size;" \
        in code
    assert code.index("const int team_size") < code.index(
        'Kokkos::parallel_for("tri_solve_kokkos"')

    # The runtime guard and the fence belong to both shapes.
    assert "if (!Kokkos::is_initialized()) {" in code
    assert code.rstrip().endswith("Kokkos::fence();\n}")


def _renamed(region):
    """Return ``region`` with its cell index, and every slice, called cell_1.

    :param region: the region to rename the launch index of.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the same region indexed by ``cell_1`` rather than ``cell``.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    arguments = tuple(
        replace(argument, extra_indices=("cell_1",))
        if isinstance(argument, KokkosView) and argument.extra_indices
        else argument
        for argument in region.arguments)
    return replace(region, cell_index="cell_1", arguments=arguments)


def test_kokkos_region_renames_a_colliding_cell_index():
    """A range launch names its index from the region, not from a literal.

    A kernel that declares ``cell`` itself shares a scope with the lambda
    parameter, so C++ rejects the translation unit outright rather than
    quietly reading the wrong index. The region carries the name so that its
    caller can choose one the kernel has not taken.
    """
    code = KokkosWriter()(_renamed(_region()))

    assert "KOKKOS_LAMBDA(const int cell_1)" in code
    assert "KOKKOS_LAMBDA(const int cell)" not in code

    # Every per-cell View is sliced by the same renamed index.
    assert ", cell_1)" in code
    assert ", cell)" not in code


def test_kokkos_team_region_renames_a_colliding_cell_index():
    """A team launch renames the index it computes from the league rank."""
    code = KokkosWriter()(_renamed(_scratch_region()))

    assert "const int cell_1 = team.league_rank() * team.team_size() + rank;" \
        in code
    assert "if (cell_1 >= ncells) {" in code
    assert ", cell_1)" in code
    assert "const int cell =" not in code
    assert "if (cell >= ncells) {" not in code


def test_kokkos_writer_sizes_and_builds_every_scratch_array():
    """Two scratch arrays are both sized into the request and both built."""
    code = KokkosWriter()(_scratch_region())

    for name in ("x_new", "tri_plus_new"):
        assert f"using {name}_scratch_t = Kokkos::View<double*, " \
            "Kokkos::LayoutLeft, ScratchSpace, Unmanaged>;" in code
        assert f"{name}_scratch_t {name}(team.thread_scratch(0), nlayers);" \
            in code

    # Summed, not counted once: a request covering one of two arrays hands
    # the second one memory the first is already using.
    assert "const size_t scratch_bytes = " \
        "x_new_scratch_t::shmem_size(nlayers)\n" \
        "      + tri_plus_new_scratch_t::shmem_size(nlayers);" in code

    # Per rank, not per team. PerTeam scratch with cells tiled across ranks
    # would give every cell in a team the same array.
    assert "Kokkos::PerTeam(" not in code
    assert "team.team_scratch(0)" not in code


def test_kokkos_writer_does_not_declare_a_scratch_symbol():
    """A scratch array is its View, and must not also be a local pointer.

    ``gen_declaration`` renders an array local as ``double * restrict x_new``,
    a pointer to nothing. It compiles, so nothing downstream would object; it
    would simply shadow the View and be dereferenced.
    """
    code = KokkosWriter()(_scratch_region())

    assert "restrict x_new" not in code
    assert "restrict tri_plus_new" not in code
    # The scalar local is still declared, so the skip is by name and not by
    # dropping local declarations altogether.
    assert "int k;" in code


def test_kokkos_writer_indexes_scratch_like_a_view():
    """Scratch resolves through the same table, with Fortran bounds removed."""
    code = KokkosWriter()(_scratch_region())

    assert "x_new((k - 1)) = " in code
    assert "tri_plus_new((k - 1)) = (x_new((k - 1)) * 2.0);" in code
    assert "= tri_plus_new((k - 1));" in code


def test_kokkos_writer_counts_a_backward_sweep_down():
    """The substitution sweep runs, rather than being tested out of existence.

    A Fortran ``do k = nlayers, 1, -1`` generated with C's ``k<=1`` compiles,
    links and runs zero iterations for any column deeper than one layer. Only
    the answer gives it away, which is why it is asserted here rather than
    left to the model.
    """
    code = KokkosWriter()(_scratch_region())

    assert "for(k=nlayers; k>=1; k+=(-1))" in code
    assert "for(k=1; k<=nlayers; k+=1)" in code


def test_kokkos_writer_without_scratch_keeps_the_range_launch():
    """A region with no local arrays generates exactly what it always did.

    The captures already in the model are gated on whole-model checksums and
    on assertions over this text, so widening the writer must not rewrite
    them.
    """
    code = KokkosWriter()(_region())

    assert 'Kokkos::parallel_for("moist_dyn_gas_kokkos", ' \
        "Kokkos::RangePolicy<>(0, ncells),\n" \
        "      KOKKOS_LAMBDA(const int cell) {" in code
    for absent in ("TeamPolicy", "TeamMember", "ScratchSpace",
                   "scratch_bytes", "thread_scratch",
                   "team_size_recommended"):
        assert absent not in code


def test_kokkos_team_launch_never_asks_team_size_max():
    """``team_size_max`` is the whole thread pool, and is never asked for.

    On the OpenMP backend it returns the pool size whatever the scratch
    request, which put the model on one team running the whole league with a
    rendezvous between consecutive cells. The query is wrong rather than
    merely suboptimal, so its absence is asserted and not just the presence
    of its replacement.
    """
    assert "team_size_max" not in KokkosWriter()(_scratch_region())


def test_kokkos_launch_module_renders_both_existing_shapes():
    """The launch shapes are rendered by ``kokkos_launch``, not the writer.

    The writer selects a shape and the module renders it. Pinning the two
    existing shapes to the module's own output is what makes the split
    checkable: the strings below are the first line each renderer emits.
    """
    code = KokkosWriter()(_scratch_region())
    assert team_launch(_scratch_region(), "", "").splitlines()[0] == (
        "  using x_new_scratch_t = Kokkos::View<double*, Kokkos::LayoutLeft,"
        " ScratchSpace, Unmanaged>;")
    assert team_launch(_scratch_region(), "", "").splitlines()[0] in code

    code = KokkosWriter()(_region())
    assert range_launch(_region(), "", "").splitlines()[0] == (
        '  Kokkos::parallel_for("moist_dyn_gas_kokkos", '
        "Kokkos::RangePolicy<>(0, ncells),")
    assert range_launch(_region(), "", "").splitlines()[0] in code


def test_kokkos_hierarchical_region_launches_one_team_per_cell():
    """A region naming a parallel loop puts one team on each cell.

    The league carries the cells, so nothing computes a cell from a rank and
    no team runs a second cell after the first; the shape's parallelism is
    the level loops inside the body rather than the ranks around it.
    """
    code = KokkosWriter()(_level_region())

    assert "TeamPolicy(ncells, Kokkos::AUTO)," in code
    assert "KOKKOS_LAMBDA(const TeamMember &team) {" in code
    assert "const int cell = team.league_rank();" in code
    assert "using TeamPolicy = Kokkos::TeamPolicy<>;" in code
    assert "using TeamMember = TeamPolicy::member_type;" in code
    # The scratch alias is not emitted, because this region has no scratch.
    # It rode along with the other two until this stage, when the loop
    # selection made a team launch without scratch reachable for the first
    # time and left the alias naming a space nothing is placed in.
    assert "ScratchSpace" not in code

    # Neither of the flat shapes' fingerprints: no range launch, and none of
    # the team-size arithmetic that tiles cells across a team.
    for absent in ("Kokkos::RangePolicy<>", "team_size_recommended",
                   "league_size"):
        assert absent not in code


def test_kokkos_hierarchical_region_spreads_the_loop_over_the_team():
    """A chosen loop becomes a ``TeamVectorRange``, not a serial ``for``."""
    code = KokkosWriter()(_level_region())

    assert "    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1, " \
        "nlayers + 1),\n        [&](const int k) {\n" in code
    # The Fortran bound is inclusive and the Kokkos range is half-open, so
    # the stop expression gains the ``+ 1`` above and the loop is gone.
    assert "for(k=1;" not in code

    # Every parallel loop is followed by a barrier: a later statement may
    # read what it wrote, and deciding whether one does is a second analysis
    # this writer does not attempt.
    closed = code.index("        [&](const int k) {")
    assert "    });\n    team.team_barrier();\n" in code[closed:]


def test_kokkos_hierarchical_region_wraps_team_level_array_writes():
    """An array written outside a parallel loop is written by one member."""
    code = KokkosWriter()(_level_region())

    assert "    Kokkos::single(Kokkos::PerTeam(team), [&]() {\n" \
        "      y(((map((1 - 1), cell) + nlayers) - 1)) = " \
        "x(((map((1 - 1), cell) + nlayers) - 1));\n" \
        "    });\n" \
        "    team.team_barrier();\n" in code

    # A scalar is a per-member local, so every member writing its own copy
    # races with nothing and needs no wrapper.
    assert "    scale = 0.5;\n" in code


def test_kokkos_hierarchical_region_places_scratch_per_team():
    """Team-shared scratch, because a team is now a cell rather than a rank."""
    code = KokkosWriter()(_level_region(scratch=(
        KokkosScratch("x_new", "double", ("nlayers",), index_offsets=(1,)),)))

    assert ".set_scratch_size(0, Kokkos::PerTeam(scratch_bytes))" in code
    assert "x_new_scratch_t x_new(team.team_scratch(0), nlayers);" in code
    for absent in ("PerThread", "thread_scratch"):
        assert absent not in code


def test_kokkos_hierarchical_region_takes_a_team_size():
    """A forced team size is a literal in the policy, replacing ``AUTO``."""
    code = KokkosWriter()(_level_region(team_size=4))

    assert "TeamPolicy(ncells, 4)," in code
    assert "Kokkos::AUTO" not in code


def test_kokkos_flat_shapes_ignore_team_size():
    """Neither flat shape reads ``team_size``; both size their own team.

    The range launch has no team at all and the flat team launch takes the
    size the backend recommends for its functor, so the option is inert on
    both rather than quietly overriding a measured answer.
    """
    for region in (_region(), _scratch_region()):
        assert KokkosWriter()(replace(region, team_size=4)) == \
            KokkosWriter()(region)


def _private_schedule():
    """Create a body whose spread loop keeps its working values in scalars.

    Both kinds of loop-private scalar are here, because both are declared by
    the same rule and a body with only one of them would leave the other to
    an assertion rather than to the generator: ``weight`` is written and read
    back within one level, and ``df`` is the counter of a loop nested inside
    that level. Neither is read before the level writes it and neither is
    read once the level loop has finished, so each level owns its own.
    """
    source = """
subroutine private_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only: r_double, i_def
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k, df
  real(kind=r_double) :: weight
  do k = 1, nlayers
    weight = 0.5_r_double * x(map(1) + k - 1)
    do df = 1, ndf
      y(map(df) + k - 1) = weight + y(map(df) + k - 1)
    end do
  end do
end subroutine private_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "private_code", symbol_table=symbol_table, children=children)


def _private_region(**overrides):
    """Return a region whose spread loop owns every scalar it writes.

    Its ABI is ``_scratch_region``'s, as ``_level_region``'s is.

    :param overrides: fields to replace on the region.
    :type overrides: dict

    :returns: the region to hand the writer.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    schedule = _private_schedule()
    region = KokkosRegion(
        name="private_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_scratch_region().arguments,
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(),
        parallel_loops=(schedule.walk(Loop)[0],))
    return replace(region, **overrides) if overrides else region


def test_kokkos_spread_loop_declares_its_scalars_inside_the_lambda():
    """A scalar the spread loop writes is declared by each iteration.

    Declared at the top of the team body it is one object per member, and
    the levels are handed out between the members: when the loop ends it
    holds the last level *that member* ran, which is a value no statement
    should be able to read. Moving the declaration inside the lambda gives
    every iteration its own and takes the name out of scope afterwards.
    """
    code = KokkosWriter()(_private_region())

    opened = code.index("        [&](const int k) {\n")
    closed = code.index("    });\n", opened)
    assert "      double weight;\n" in code[opened:closed]
    # Not both: the region-scope declaration is removed rather than
    # shadowed, so that a member reading the outer one is a compile error
    # here and not a wrong answer on the machine this is generated for.
    assert "\n    double weight;\n" not in code


def test_kokkos_spread_loop_declares_its_inner_counters_inside_the_lambda():
    """The counter of a loop inside the spread loop is private too.

    It is written by the inner loop's own initialisation before anything
    reads it and nothing reads it afterwards, which is the same rule the
    working scalars meet; the counter is worth its own test because it is
    the one the kernel author never declared and so never thinks about.
    """
    code = KokkosWriter()(_private_region())

    opened = code.index("        [&](const int k) {\n")
    closed = code.index("    });\n", opened)
    assert "      int df;\n" in code[opened:closed]
    assert "\n    int df;\n" not in code
    # The loop itself is still the serial one each member runs: only where
    # its counter is declared has changed.
    assert "for(df=1; df<=ndf; df+=1)" in code[opened:closed]


def _live_out_schedule():
    """Create a body reading a scalar after the loop that wrote it.

    The level loop leaves ``running`` holding the last level's value and the
    statement after it reads that value, which is a loop-carried use: the
    loop cannot be spread over the team at all, whatever the writer does
    with the declaration. A second loop follows so that the same body can
    also be generated with a different loop chosen.
    """
    source = """
subroutine live_out_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only: r_double, i_def
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k, df
  real(kind=r_double) :: running
  do k = 1, nlayers
    running = x(map(1) + k - 1) * 2.0_r_double
    y(map(1) + k - 1) = running
  end do
  do df = 1, ndf
    y(map(df) + nlayers) = x(map(df) + nlayers)
  end do
  y(map(1) + nlayers) = running
end subroutine live_out_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "live_out_code", symbol_table=symbol_table, children=children)


def _live_out_region(loop=0):
    """Return a region spreading one of the two loops of that body.

    :param loop: which of the body's loops to spread, in source order.
    :type loop: int

    :returns: the region to hand the writer.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    schedule = _live_out_schedule()
    return KokkosRegion(
        name="live_out_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_scratch_region().arguments,
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(),
        parallel_loops=(schedule.walk(Loop)[loop],))


def test_kokkos_scalar_read_after_the_spread_loop_stays_outside():
    """A scalar the body reads after the loop is not hidden in the lambda.

    Moving it inside would silence the C++ compiler, which is the whole
    danger: the statement after the loop would then read a differently named
    object and the wrong answer would arrive with no diagnostic anywhere.
    The value it wants belongs to the last level, and after a spread no
    member holds it, so the region is refused instead.

    With the other loop chosen the same scalar is generated exactly as it
    was: nothing is moved out of a loop that was not spread.
    """
    with pytest.raises(ValueError) as err:
        KokkosWriter()(_live_out_region())
    message = str(err.value)
    assert "running" in message
    assert "live_out_kokkos" in message

    code = KokkosWriter()(_live_out_region(loop=1))
    assert "        [&](const int df) {\n" in code
    assert "\n    double running;\n" in code
    assert "\n      double running;\n" not in code


def _spread_region(body, loop=0):
    """Return a region spreading one loop of a body given as Fortran.

    The declarations are fixed and the statements are not, so that a rule
    about scalars can be exercised by writing the statements that break it
    rather than by building a schedule by hand.

    :param body: the executable part of the kernel, indented as Fortran.
    :type body: str
    :param loop: which of the body's loops to spread, in source order; a
        tuple to spread more than one.
    :type loop: int | Tuple[int]

    :returns: the region to hand the writer.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    source = f"""
subroutine spread_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only: r_double, i_def
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k, df
  real(kind=r_double) :: weight, total
{body}
end subroutine spread_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    schedule = KernelSchedule.create(
        "spread_code", symbol_table=symbol_table, children=children)
    indices = (loop,) if isinstance(loop, int) else loop
    return KokkosRegion(
        name="spread_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_scratch_region().arguments,
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(),
        parallel_loops=tuple(
            schedule.walk(Loop)[index] for index in indices))


def test_kokkos_spread_loop_takes_a_scalar_every_branch_writes():
    """A scalar written by both arms of an ``if`` is written by the level.

    Whichever arm runs leaves the iteration's own value in it, so it is
    private on the same terms as one written unconditionally. This is the
    case
    :py:meth:`~psyclone.psyir.tools.dependency_tools.DependencyTools._is_scalar_parallelisable`
    is not asked about: it takes the first access to a scalar anywhere in the
    loop and does not ask which branches reach it.
    """
    code = KokkosWriter()(_spread_region("""
  do k = 1, nlayers
    if (x(map(1) + k - 1) > 0.0_r_double) then
      weight = 1.0_r_double
    else
      weight = 2.0_r_double
    end if
    y(map(1) + k - 1) = weight
  end do
"""))

    opened = code.index("        [&](const int k) {\n")
    closed = code.index("    });\n", opened)
    assert "      double weight;\n" in code[opened:closed]
    assert "\n    double weight;\n" not in code


@pytest.mark.parametrize("case, body", [
    ("one branch", """
  do k = 1, nlayers
    if (x(map(1) + k - 1) > 0.0_r_double) then
      weight = 1.0_r_double
    end if
    y(map(1) + k - 1) = weight
  end do
"""),
    ("accumulated", """
  do k = 1, nlayers
    weight = weight + x(map(1) + k - 1)
    y(map(1) + k - 1) = weight
  end do
"""),
    ("written in a while", """
  do k = 1, nlayers
    do while (weight < x(map(1) + k - 1))
      weight = weight + 1.0_r_double
    end do
    y(map(1) + k - 1) = weight
  end do
"""),
])
def test_kokkos_spread_loop_refuses_a_scalar_an_iteration_reads_first(
        case, body):
    """A level reading a scalar it has not yet written cannot be spread.

    Each of these carries a value from one iteration into the next, which is
    what a spread destroys: an ``if`` with no ``else`` leaves the previous
    level's value in place, an accumulation reads it on purpose, and a
    ``while`` is not analysed at all, so what it writes is not known to have
    been written. The third is refused for being unknown rather than for
    being wrong, which is the safe direction: the answer costs a loop its
    spread and never a wrong number.
    """
    with pytest.raises(ValueError) as err:
        KokkosWriter()(_spread_region(body))
    assert "reads before it writes" in str(err.value), case
    assert "weight" in str(err.value)


def test_kokkos_spread_loop_refuses_a_scalar_on_the_interface():
    """A formal the loop writes is refused: it cannot be moved anywhere.

    A kernel argument is a parameter of the generated function and part of
    the region's ABI, so there is no declaration to move inside the lambda
    and no way for a member to have its own. LFRic's kernels do not write
    their scalar arguments today, and the check is here because the writer
    must not depend on that.
    """
    region = _spread_region("""
  do k = 1, nlayers
    weight = x(map(1) + k - 1)
    y(map(1) + k - 1) = weight
  end do
""")
    table = region.schedule.symbol_table
    formals = list(table.argument_list)
    symbol = table.lookup("weight")
    symbol.interface = ArgumentInterface(ArgumentInterface.Access.READWRITE)
    table.specify_argument_list(formals + [symbol])
    # Described as well as declared, or the region is refused a step earlier
    # for an argument it does not account for.
    region = replace(
        region,
        arguments=region.arguments + (KokkosScalar("weight", "double"),))

    with pytest.raises(ValueError) as err:
        KokkosWriter()(region)
    assert "not a local of the kernel but part of its interface" in \
        str(err.value)


def test_kokkos_spread_loop_shadows_a_scalar_the_body_also_uses():
    """A scalar used at team level as well keeps both declarations.

    ``weight`` is a working value in the spread loop and a working value
    again in the statements after it, and every use writes it before reading
    it. The loop's members each need their own; the statements after it need
    something to name. So the private declaration shadows the region-scope
    one rather than replacing it, which is what an LFRic kernel reusing one
    temporary through a long body needs -- the alternative refuses the region
    over a name.
    """
    code = KokkosWriter()(_spread_region("""
  do k = 1, nlayers
    weight = x(map(1) + k - 1)
    y(map(1) + k - 1) = weight
  end do
  weight = x(map(1)) * 2.0_r_double
  y(map(1)) = weight
"""))

    opened = code.index("        [&](const int k) {\n")
    closed = code.index("    });\n", opened)
    assert "      double weight;\n" in code[opened:closed]
    assert "\n    double weight;\n" in code


def test_kokkos_spread_loop_refuses_a_scalar_the_enclosing_loop_carries():
    """A serial loop around the spread loop runs its earlier statements again.

    The read before the spread loop is a read after it as well, on every
    iteration of the enclosing loop but the last, so the value the spread
    destroyed is wanted after all. Liveness that only looked forwards from
    the loop would miss this and generate a wrong answer silently.
    """
    with pytest.raises(ValueError) as err:
        KokkosWriter()(_spread_region("""
  do df = 1, ndf
    y(map(df)) = weight
    do k = 1, nlayers
      weight = x(map(df) + k - 1)
    end do
  end do
""", loop=1))
    assert "reads after the loop without writing it first" in str(err.value)
    assert "weight" in str(err.value)


def test_kokkos_spread_loop_refuses_a_scalar_a_while_condition_reads():
    """The condition of an enclosing ``while`` is a read after the loop.

    It is evaluated once the body has run, on the value the last iteration
    left behind, which after a spread no member holds. The condition is not
    a statement of the body and is counted separately for that reason.
    """
    with pytest.raises(ValueError) as err:
        KokkosWriter()(_spread_region("""
  weight = 0.0_r_double
  do while (weight < 10.0_r_double)
    do k = 1, nlayers
      weight = x(map(1) + k - 1)
    end do
  end do
"""))
    assert "reads after the loop without writing it first" in str(err.value)
    assert "weight" in str(err.value)


def test_kokkos_spread_loop_declares_only_what_it_writes_itself():
    """A name a second spread loop only reads is not declared in its lambda.

    ``df`` counts a serial loop inside the first spread loop, which makes it
    that loop's own, and is read by the second, which does not write it at
    all. A declaration in the second loop's lambda would shadow the value
    the body left there with an uninitialised object of the same name --
    which compiles, and which no compiler will say a word about. Privacy is
    therefore per loop and follows what each loop writes, not what it names.
    """
    code = KokkosWriter()(_spread_region("""
  do k = 1, nlayers
    do df = 1, ndf
      y(map(df) + k - 1) = x(map(df) + k - 1)
    end do
  end do
  df = 2
  do k = 1, nlayers
    y(map(df) + k - 1) = x(map(1) + k - 1) * 2.0_r_double
  end do
""", loop=(0, 2)))

    first = code.index("        [&](const int k) {\n")
    second = code.index("        [&](const int k) {\n", first + 1)
    assert "      int df;\n" in code[first:second]
    assert "      int df;\n" not in code[second:]
    # The team-level declaration stays: the statements between the two loops
    # write it and the second loop reads what they wrote.
    assert "\n    int df;\n" in code


def _with_cell_position(region, name="cell"):
    """Return ``region`` with ``name`` prepended to its schedule's formals.

    LFRic gives every kernel taking an operator a leading cell argument, which
    the region declares from its own launch index rather than taking across the
    ABI. This builds that state directly: the formal is in the schedule's
    argument list and in none of ``region.arguments``.

    :param region: the region to add a cell position to.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param name: the formal's name, defaulting to the one LFRic uses.
    :type name: str

    :returns: the same region, with the formal added and named as its cell
        position.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    table = region.schedule.symbol_table
    # Read the existing formals before adding: the property revalidates the
    # table, and between the add and the specify there is an argument symbol
    # the list does not carry.
    formals = list(table.argument_list)
    symbol = DataSymbol(
        name, ScalarType.integer_type(),
        interface=ArgumentInterface(ArgumentInterface.Access.READ))
    table.add(symbol)
    table.specify_argument_list([symbol] + formals)
    return replace(region, cell_position=name)


def test_kokkos_cell_position_declared_in_range_launch():
    """The cell position is declared from the launch index, not passed.

    The kernel computes its own offset into an operator from ``cell``, so the
    value has to be the cell the iteration is on. Taking it as a parameter
    would give every cell whatever the caller passed once, which is why it is
    declared inside the lambda instead.
    """
    code = KokkosWriter()(_with_cell_position(_renamed(_region())))

    assert ("KOKKOS_LAMBDA(const int cell_1) {\n"
            "    const int cell = cell_1 + 1;\n") in code

    # And it is nowhere on the ABI: the caller neither has the value nor
    # needs it.
    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1] for parameter in signature.split(",")
        if parameter.strip().split()
    ]
    assert "cell" not in parameters
    assert "ncells" in parameters


def test_kokkos_cell_position_declared_in_team_launch():
    """The flat team launch declares it after computing its own index."""
    code = KokkosWriter()(_with_cell_position(_renamed(_scratch_region())))

    assert "\n      const int cell = cell_1 + 1;\n" in code
    assert code.index("const int cell_1 = team.league_rank()") < \
        code.index("const int cell = cell_1 + 1;")


def test_kokkos_cell_position_declared_in_hierarchical_launch():
    """The hierarchical launch declares it straight from the league rank."""
    code = KokkosWriter()(_with_cell_position(_renamed(_level_region())))

    assert ("    const int cell_1 = team.league_rank();\n"
            "    const int cell = cell_1 + 1;\n") in code


def test_kokkos_cell_position_absent_declares_nothing():
    """The field is inert on every region that does not set it.

    Every capture in the model was generated before it existed, and each is
    gated on assertions over its exact text, so the declaration has to be
    conditional rather than empty-when-unset.
    """
    for region in (_region(), _scratch_region(), _level_region()):
        code = KokkosWriter()(region)
        assert "= cell + 1;" not in code
        assert "= cell_1 + 1;" not in code


@pytest.mark.parametrize("name, message", [
    ("not an identifier", "is not a C++ identifier"),
    ("ik", "is not a kernel argument"),
    ("nlayers", "is also described as a region argument"),
])
def test_kokkos_writer_rejects_an_invalid_cell_position(name, message):
    """Three ways the field can name something it cannot declare.

    None announces itself downstream: an unknown name generates a declaration
    initialised from nothing, and a name the region also passes generates a
    parameter and a local of the same name in one scope.
    """
    with pytest.raises(ValueError) as error:
        KokkosWriter()(replace(_region(), cell_position=name))

    assert message in str(error.value)
    assert name in str(error.value)


def test_kokkos_writer_rejects_a_cell_position_shadowing_the_index():
    """A cell position equal to the launch index initialises from itself.

    ``const int cell = cell + 1;`` reads the object being declared, which C++
    permits and no compiler is obliged to diagnose. The transformation renames
    the index away from a kernel's own names, so this is unreachable from
    there; it is checked because the consequence of reaching it is a value
    rather than an error.
    """
    region = _with_cell_position(_region())

    with pytest.raises(ValueError) as error:
        KokkosWriter()(region)

    assert "'cell' is also the launch's cell index" in str(error.value)


def _nested_schedule():
    """Create a body whose three loops are one inside the next.

    Three rather than two, so that a chosen loop can be given a parent that
    was not chosen and a grandparent that was: the enclosure check walks up
    to the schedule and is not satisfied by looking at the parent alone.
    """
    source = """
subroutine nested_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only: r_double, i_def
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k, df, i
  do df = 1, ndf
    do k = 1, nlayers
      do i = 1, 4
        y(map(df) + k - 1) = x(map(df) + k - 1)
      end do
    end do
  end do
end subroutine nested_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "nested_code", symbol_table=symbol_table, children=children)


def _invalid_parallel_loops(case):
    """Return a region whose ``parallel_loops`` breaks one of the four rules.

    :param case: which rule to break.
    :type case: str

    :returns: the region to hand the writer.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    if case == "not-a-loop":
        return _level_region(parallel_loops=("k",))
    if case == "foreign":
        return _level_region(
            parallel_loops=(_scratch_schedule().walk(Loop)[0],))
    if case in ("nested", "nested-deep"):
        schedule = _nested_schedule()
        loops = schedule.walk(Loop)
        return _level_region(
            schedule=schedule,
            parallel_loops=(
                (loops[0], loops[1]) if case == "nested"
                else (loops[0], loops[2])))
    region = _level_region()
    region.parallel_loops[0].step_expr.replace_with(
        Literal("2", ScalarType.integer_type()))
    return region


@pytest.mark.parametrize("case, error, message", [
    ("not-a-loop", TypeError, "parallel_loops"),
    ("foreign", ValueError, "not in the region's schedule"),
    ("nested", ValueError, "nested"),
    ("nested-deep", ValueError, "nested"),
    ("stepped", ValueError, "step"),
])
def test_kokkos_writer_rejects_invalid_parallel_loops(case, error, message):
    """The loops the transformation chose are checked, not trusted.

    A loop from another schedule would be visited by identity and never
    matched, so the region would silently generate a serial ``for``; the
    other three would generate C++ that does not mean what the Fortran did.
    """
    with pytest.raises(error) as err:
        KokkosWriter()(_invalid_parallel_loops(case))
    assert message in str(err.value)


@pytest.mark.parametrize("team_size, error", [
    ("4", TypeError),
    (4.0, TypeError),
    (True, TypeError),
    (0, ValueError),
    (-1, ValueError),
])
def test_kokkos_writer_rejects_invalid_team_size(team_size, error):
    """``team_size`` is rendered into the policy, so it is checked first.

    ``True`` is refused although it is an ``int`` in Python: a team of one
    written as ``TeamPolicy(ncells, True)`` compiles and runs, so nothing
    downstream would report it.
    """
    with pytest.raises(error) as err:
        KokkosWriter()(_level_region(team_size=team_size))
    assert "team_size" in str(err.value)


def test_kokkos_writer_views_are_never_managed():
    """No View owns storage, in either launch shape.

    ``managed`` exists so that a View claiming ownership is refused rather
    than silently double-freeing LFRic's memory. Team scratch was chosen over
    an allocated slab partly to keep that unqualified, so it is asserted
    rather than left to the absence of a caller setting it.
    """
    for region in (_region(), _scratch_region()):
        for argument in region.arguments:
            if isinstance(argument, KokkosView):
                assert argument.managed is False


@pytest.mark.parametrize("scratch, message", [
    (KokkosScratch("x new", "double", ("nlayers",), index_offsets=(1,)),
     "Kokkos scratch name 'x new' is invalid."),
    (KokkosScratch("x_new", "half", ("nlayers",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has unsupported C type 'half'."),
    (KokkosScratch("x_new", "double", ()),
     "Kokkos scratch 'x_new' must have extents."),
    (KokkosScratch("x_new", "double", ("nrows",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent 'nrows', which is sized from 'nrows' "
     "rather than from a scalar argument."),
    (KokkosScratch("x_new", "double", ("y",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent 'y', which is sized from 'y' rather "
     "than from a scalar argument."),
    (KokkosScratch("x_new", "double", ("nrows + 1",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent 'nrows + 1', which is sized from "
     "'nrows' rather than from a scalar argument."),
    (KokkosScratch("x_new", "double", ("max(nlayers, 1)",),
                   index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent 'max(nlayers, 1)' which is not an "
     "integer expression over named sizes."),
    (KokkosScratch("x_new", "double", ("(nlayers",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent '(nlayers' which is not an integer "
     "expression over named sizes."),
    (KokkosScratch("x_new", "double", ("4nlayers",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent '4nlayers' which is not an integer "
     "expression over named sizes."),
    (KokkosScratch("x_new", "double", ("nlayers",), index_offsets=(1, 1)),
     "Kokkos scratch 'x_new' dimensions do not match its kernel indices."),
    (KokkosScratch("x_new", "double", ("nlayers",), index_offsets=(1.5,)),
     "Kokkos scratch 'x_new' index offsets must be integers or integer "
     "expressions over named sizes."),
    (KokkosScratch("x_new", "double", ("nlayers",),
                   index_offsets=("max(nlayers, 1)",)),
     "Kokkos scratch 'x_new' index offsets must be integers or integer "
     "expressions over named sizes."),
    (KokkosScratch("y", "double", ("nlayers",), index_offsets=(1,)),
     "Kokkos scratch 'y' collides with an existing region name."),
    (KokkosScratch("nlayers", "double", ("nlayers",), index_offsets=(1,)),
     "Kokkos scratch 'nlayers' collides with an existing region name."),
    ("x_new", "KokkosRegion scratch must be KokkosScratch instances, found "
     "'str'."),
])
def test_kokkos_writer_rejects_invalid_scratch(scratch, message):
    """Every way of describing scratch wrongly is refused by name."""
    region = _scratch_region(scratch=(scratch,))
    with pytest.raises((ValueError, TypeError)) as error:
        KokkosWriter()(region)
    assert message in str(error.value)


def _constant(**kwargs):
    """Return a one-dimensional carried-constant description."""
    integer = ScalarType(ScalarType.Intrinsic.INTEGER,
                         ScalarType.Precision.UNDEFINED)
    fields = {"name": "x_dofs", "c_type": "int",
              "values": (Literal("1", integer), Literal("3", integer)),
              "index_offsets": (1,)}
    fields.update(kwargs)
    return KokkosConstant(**fields)


def test_kokkos_writer_declares_a_constant_inside_the_body():
    """A constant array is declared among the body's locals, and not taken.

    Not at file scope, which is what a Fortran ``parameter`` beside a kernel
    most resembles: a namespace-scope array is host data, and nvcc rejects a
    device lambda that subscripts one. Inside the body it is a local of
    whichever execution space the launch runs in, and that is the only
    spelling both accept.

    The values are generated rather than held as text, so they cross the same
    literal path the body's own literals do; and nothing about the constant
    reaches the ABI, which is the point of describing it here rather than as
    an argument.
    """
    region = replace(_region(), constants=(_constant(),))

    generated = KokkosWriter()(region)

    assert generated.startswith(
        "#include <Kokkos_Core.hpp>\n\n"
        'extern "C" void moist_dyn_gas_kokkos(')
    assert "static const" not in generated
    assert "    const int x_dofs[2] = {1, 3};\n" in generated
    assert "x_dofs" not in generated.split("(", 1)[1].split(") {", 1)[0]


def test_kokkos_writer_indexes_a_constant_with_brackets():
    """A constant is a C array, so its subscript is one too.

    The origin is removed exactly as a View's is: the constant is described
    in the same table and read by the same code, and only the punctuation of
    the access differs.
    """
    schedule = _kernel_schedule()
    integer = ScalarType(ScalarType.Intrinsic.INTEGER,
                         ScalarType.Precision.UNDEFINED)
    schedule.symbol_table.add(DataSymbol("x_dofs", ArrayType(integer, [2])))
    body = schedule.walk(Assignment)[0]
    body.rhs.replace_with(
        FortranReader().psyir_from_expression(
            "mr_v(x_dofs(df))", schedule.symbol_table))
    region = replace(_region(), schedule=schedule,
                     constants=(_constant(),))

    generated = KokkosWriter()(region)

    assert "mr_v((x_dofs[(df - 1)] - 1))" in generated


@pytest.mark.parametrize("constant, message", [
    ("x_dofs", "KokkosRegion constants must be KokkosConstant instances of a "
     "supported C type, found 'x_dofs'."),
    (_constant(c_type="quad"),
     "KokkosRegion constants must be KokkosConstant instances of a supported "
     "C type,"),
    (_constant(values=()),
     "Kokkos constant 'x_dofs' must have at least one value and exactly one "
     "index offset."),
    (_constant(index_offsets=(1, 1)),
     "Kokkos constant 'x_dofs' must have at least one value and exactly one "
     "index offset."),
])
def test_kokkos_writer_rejects_an_invalid_constant(constant, message):
    """Every way of describing a carried constant wrongly is refused."""
    region = replace(_region(), constants=(constant,))
    with pytest.raises((ValueError, TypeError)) as error:
        KokkosWriter()(region)
    assert message in str(error.value)


@pytest.mark.parametrize("field, message", [
    ("name", "Kokkos region name 'not a name' is not a C++ identifier."),
    ("cell_count", "Cell count 'not a name' is not a C++ identifier."),
    ("cell_index", "Cell index 'not a name' is not a C++ identifier."),
])
def test_kokkos_writer_rejects_a_name_that_is_not_an_identifier(
        field, message):
    """Each name the writer emits verbatim is checked before it is emitted.

    All three reach the generated source as a declaration or a label, so a
    value that is not a C++ identifier produces a translation unit that does
    not compile. The refusal names which of the three it was.
    """
    with pytest.raises(ValueError) as error:
        KokkosWriter()(replace(_region(), **{field: "not a name"}))
    assert message in str(error.value)


def test_kokkos_writer_takes_a_literal_extent():
    """A dimension the kernel author wrote as a number is emitted as one.

    A literal bound is what refuses 179 of the 183 loops the ``local-array``
    catalogue row counts, so a literal reaching the generated C++ unchanged is
    the whole of what this stage buys.
    """
    scratch = tuple(
        replace(item, extents=("4",)) if item.name == "x_new" else item
        for item in _scratch_region().scratch)
    code = KokkosWriter()(_scratch_region(scratch=scratch))

    assert "x_new_scratch_t::shmem_size(4)\n" in code
    assert "x_new_scratch_t x_new(team.thread_scratch(0), 4);" in code
    # The other array is still sized from a name, so the two forms coexist in
    # one request rather than the writer having switched mode.
    assert "+ tri_plus_new_scratch_t::shmem_size(nlayers);" in code


def test_kokkos_writer_takes_an_arithmetic_extent():
    """An extent may be an expression, and reaches both places verbatim."""
    scratch = tuple(
        replace(item, extents=("(nlayers + 1)",)) if item.name == "x_new"
        else item
        for item in _scratch_region().scratch)
    code = KokkosWriter()(_scratch_region(scratch=scratch))

    assert "x_new_scratch_t::shmem_size((nlayers + 1))\n" in code
    assert "x_new_scratch_t x_new(team.thread_scratch(0), (nlayers + 1));" \
        in code


def test_kokkos_view_takes_an_expression_extent():
    """A View sizes from an expression as scratch does.

    ``_extents`` serves array formals and kernel-local arrays alike, so
    widening it reaches Views whether or not an LFRic formal is ever declared
    with a bound that is not a name.
    """
    arguments = tuple(
        replace(argument, extents=("4", "ncells"))
        if getattr(argument, "name", None) == "map" else argument
        for argument in _scratch_region().arguments)
    code = KokkosWriter()(_scratch_region(arguments=arguments))

    assert "map(map_data, 4, ncells);" in code
    # Still rank 2: a literal extent must not change how the View is typed.
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft, " in code


@pytest.mark.parametrize("extent, accepted", [
    ("nlayers", True), ("4", True), ("(nlayers + 1)", True),
    ("nlayers + 1", True), ("2 * nlayers - 1", True), ("nlayers*4", True),
    ("nlayers / 2", True), ("((stencil_size + 1) / 2)", True),
    ("", False), ("   ", False), ("4nlayers", False),
    ("(nlayers", False), ("nlayers)", False), (")nlayers(", False),
    ("nlayers % 2", False), ("nlayers.size", False), (4, False), (None, False),
    ("pow(nlayers, 2)", False), ("max(nlayers, 1)", False),
])
def test_is_extent(extent, accepted):
    """The extent predicate accepts arithmetic and refuses everything else.

    Division is accepted: Fortran and C++ both truncate an integer quotient
    toward zero, so an extent that divides is the extent the kernel declared.
    A call is not, because the comma in one is what a generated
    ``shmem_size`` argument may not carry. ``)nlayers(`` is here because a
    depth count that only checked the total would accept it.
    """
    assert is_extent(extent) is accepted


@pytest.mark.parametrize("offset, accepted", [
    (1, True), (0, True), (-3, True), ("1", True), ("0", True),
    ("(-nlayers)", True), ("2 * nlayers", True), ("nlayers / 2", True),
    (1.5, False), ("max(nlayers, 1)", False), ("", False), (None, False),
])
def test_is_offset(offset, accepted):
    """An offset is an integer, or an extent expression standing in for one.

    The grammar is the extent grammar because an origin and an extent are two
    readings of one declaration: a shape admitted into the extent and refused
    in the origin would size a View correctly and index it from the wrong
    place. A negative integer is accepted, since an array centred on zero has
    a negative origin, and a division is accepted for the same reason it is
    accepted in an extent: an origin that rounded the other way would shift
    every subscript of the array by one.
    """
    assert is_offset(offset) is accepted


@pytest.mark.parametrize("extent, names", [
    ("nlayers", {"nlayers"}), ("4", set()), ("(nlayers + 1)", {"nlayers"}),
    ("2 * nrows - ncols", {"nrows", "ncols"}), (4, set()),
])
def test_extent_names(extent, names):
    """An extent reports the sizes it is built from, and only those."""
    assert extent_names(extent) == names


def test_kokkos_writer_rejects_two_scratch_arrays_sharing_a_name():
    """The second of two scratch arrays cannot reuse the first one's name."""
    duplicate = KokkosScratch("x_new", "double", ("nlayers",),
                              index_offsets=(1,))
    region = _scratch_region(scratch=(duplicate, duplicate))
    with pytest.raises(ValueError) as error:
        KokkosWriter()(region)
    assert "Kokkos scratch 'x_new' collides with an existing region name." \
        in str(error.value)


#: The three kinds the intrinsic tests describe, as a region would.
_INTRINSIC_KINDS = (("r_def", "double"), ("r_solver", "float"),
                    ("i_def", "int"))


def _written_expressions(body, kind_types=_INTRINSIC_KINDS):
    """Write the right-hand side of each assignment in ``body``.

    The writer is driven directly rather than through a region, because an
    intrinsic's spelling depends only on the kinds in force and on the
    argument types the frontend resolved -- neither of which the launch
    shape, the View descriptions or the ABI can change.

    :param str body: assignments, indented, over the names declared below.
    :param kind_types: the ``(kind name, C type)`` pairs in force.
    :type kind_types: tuple[tuple[str, str], ...]

    :returns: what the writer made of each right-hand side, in order.
    :rtype: list[str]
    """
    source = f"""
subroutine intrinsic_probe(i, j, a, b, c, s)
  use constants_mod, only : i_def, r_def, r_solver
  integer(kind=i_def) :: i, j
  real(kind=r_def) :: a, b, c
  real(kind=r_solver) :: s
{body}
end subroutine intrinsic_probe
"""
    writer = KokkosWriter()
    writer._kind_types = dict(kind_types)
    return [writer._visit(assignment.rhs)
            for assignment
            in FortranReader().psyir_from_source(source).walk(Assignment)]


def test_kokkos_writer_qualifies_intrinsic_functions():
    """Every maths function is ``Kokkos::``-qualified, one per shape.

    Qualification is what makes the call legal in device code, where an
    unqualified ``sqrt`` is a host function. One intrinsic is taken from each
    shape the C writer knows: a one-argument function, a two-argument one, a
    two-argument one whose C name differs from its Fortran name, and the two
    that need an integer cast around a floating-point function.
    """
    assert _written_expressions("""
  a = sqrt(a)
  a = atan2(a, b)
  a = sign(a, b)
  i = nint(a)
  i = floor(a)
""") == ["Kokkos::sqrt(a)", "Kokkos::atan2(a, b)", "Kokkos::copysign(a, b)",
         "(int)Kokkos::round(a)", "(int)Kokkos::floor(a)"]


def test_kokkos_writer_spells_an_intrinsic_by_its_argument_type():
    """``ABS`` and ``MOD`` are spelt by their argument's type.

    ``Kokkos::abs`` truncates a real and ``Kokkos::fabs`` returns a double
    for an integer, so neither is right for both. Integer ``MOD`` has no
    function at all: it is the ``%`` operator, which the C writer already
    writes and which needs no qualification.
    """
    assert _written_expressions("""
  a = abs(a)
  i = abs(i)
  a = mod(a, b)
  i = mod(i, j)
""") == ["Kokkos::fabs(a)", "Kokkos::abs(i)", "Kokkos::fmod(a, b)", "(i % j)"]


def test_kokkos_writer_folds_max_and_min():
    """``MAX`` and ``MIN`` fold right to left, for integers as well.

    ``Kokkos::max`` is type-generic where ``fmax`` is not, so the integer
    case the C writer refuses -- for want of a standard C spelling -- is
    generated here.
    """
    assert _written_expressions("""
  a = max(a, b, c)
  i = max(i, j)
  i = min(i, j)
""") == ["Kokkos::max(a, Kokkos::max(b, c))", "Kokkos::max(i, j)",
         "Kokkos::min(i, j)"]


def test_kokkos_writer_refuses_a_fold_over_one_argument():
    """A ``MAX`` with one argument is no Fortran, and is refused as such."""
    call = IntrinsicCall(IntrinsicCall.Intrinsic.MAX)
    call.addchild(Reference(DataSymbol("i", ScalarType.integer_type())))
    with pytest.raises(VisitorError) as error:
        KokkosWriter()._visit(call)
    assert ("The Kokkos back-end can only fold 'MAX' over 2 or more "
            "arguments, but found 1." in str(error.value))


def test_kokkos_writer_casts_at_the_kind_that_was_asked_for():
    """A cast takes its target from the kind the Fortran named."""
    assert _written_expressions("""
  a = real(i, r_def)
  s = real(a, r_solver)
  i = int(a, i_def)
""") == ["(double)i", "(float)a", "(int)a"]


def test_kokkos_writer_leaves_an_undescribed_cast_kind_to_the_c_writer():
    """A kind the region did not describe falls back rather than raising.

    The same ``real(a, r_solver)`` is a ``float`` where the region says
    ``r_solver`` is single precision and the C writer's ``double`` where the
    region says nothing about it. Raising instead would report a region
    unsupported for a kind that is merely undescribed, which is the normal
    case for the probe a caller uses to ask what is supported.
    """
    assert _written_expressions(
        "  s = real(a, r_solver)\n",
        (("r_solver", "float"),)) == ["(float)a"]
    assert _written_expressions(
        "  s = real(a, r_solver)\n", ()) == ["(double)a"]


def test_kokkos_writer_ignores_a_kind_belonging_to_another_intrinsic():
    """A kindless ``real(i)`` is a default real, whatever ``i``'s kind is.

    PSyIR infers the precision of ``real(i)`` from its argument, reporting
    ``Scalar<REAL, Reference['i_def']>``. Resolving that kind would cast to
    ``int`` and lose the value, so a kind whose C type does not belong to the
    intrinsic being cast to is treated as unresolved.
    """
    assert _written_expressions("  a = real(i)\n") == ["(double)i"]


def test_kokkos_writer_writes_epsilon_as_a_numeric_trait():
    """``EPSILON`` is a question about a type, answered by a Kokkos trait.

    Its argument is consumed for its type and never visited, so the trait
    carries the width of the argument's kind and the argument itself does not
    appear in the generated code.
    """
    assert _written_expressions("""
  a = epsilon(a)
  s = epsilon(s)
""") == ["Kokkos::Experimental::epsilon_v<double>",
         "Kokkos::Experimental::epsilon_v<float>"]


def test_kokkos_writer_refuses_epsilon_of_an_undescribed_kind():
    """``EPSILON`` has no answer to fall back on, so it refuses.

    Unlike a cast, there is no kind-blind spelling: the trait has to be
    instantiated at some width, and guessing one would be a wrong answer
    rather than a wide one.
    """
    with pytest.raises(VisitorError) as error:
        _written_expressions("  a = epsilon(a)\n", ())
    assert ("EPSILON needs the width of its argument's kind, which this "
            "region does not describe." in str(error.value))


def test_kokkos_writer_leaves_an_unknown_intrinsic_to_the_c_writer():
    """An intrinsic neither writer knows raises the C writer's own error."""
    with pytest.raises(VisitorError) as error:
        _written_expressions("  a = tiny(a)\n")
    assert "The C backend does not support the 'TINY' intrinsic." \
        in str(error.value)


# The kinds the array-expression probe below declares. A lowered nest
# generates its own loop index, which has no Fortran kind at all, so a region
# that described none would not distinguish "int, because the region says so"
# from "int, because nothing said otherwise".
_ARRAY_KINDS = (("i_def", "int"), ("r_def", "double"))


def _array_probe(body):
    """Return a kernel schedule holding array-valued statements.

    :param str body: the statements, indented, over the arrays declared here.

    :returns: the body, detached from the routine that parsed it.
    :rtype: :py:class:`psyclone.psyir.nodes.KernelSchedule`
    """
    source = f"""
subroutine array_probe(nlayers, df, a, b, m, jac, p, q, w, f, g)
  use constants_mod, only : i_def, r_def
  integer(kind=i_def), intent(in) :: nlayers, df
  real(kind=r_def), dimension(nlayers), intent(inout) :: a
  real(kind=r_def), dimension(nlayers), intent(in) :: b
  real(kind=r_def), dimension(3,nlayers), intent(inout) :: m
  real(kind=r_def), dimension(3,3,nlayers), intent(in) :: jac
  real(kind=r_def), dimension(3), intent(inout) :: p
  real(kind=r_def), dimension(3), intent(in) :: q
  real(kind=r_def), dimension(3,3), intent(in) :: w
  real(kind=r_def), dimension(6), intent(in) :: f
  real(kind=r_def), dimension(2,3), intent(inout) :: g
  real(kind=r_def), dimension(3,3) :: v
  real(kind=r_def) :: x
{body}
end subroutine array_probe
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "array_probe", symbol_table=symbol_table, children=children)


def _array_region(body):
    """Return a region over the array probe's body.

    :param str body: the statements, indented, over the probe's arrays.

    :returns: the region description the writer is given.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    return KokkosRegion(
        name="array_probe_kokkos",
        schedule=_array_probe(body),
        cell_count="ncells",
        kind_types=_ARRAY_KINDS,
        arguments=(
            KokkosScalar("nlayers", "int"),
            KokkosScalar("df", "int"),
            KokkosView("a", "a_data", "double", ("nlayers",),
                       index_offsets=(1,)),
            KokkosView("b", "b_data", "double", ("nlayers",),
                       index_offsets=(1,), read_only=True),
            KokkosView("m", "m_data", "double", ("3", "nlayers"),
                       index_offsets=(1, 1)),
            KokkosView("jac", "jac_data", "double", ("3", "3", "nlayers"),
                       index_offsets=(1, 1, 1), read_only=True),
            KokkosView("p", "p_data", "double", ("3",), index_offsets=(1,)),
            KokkosView("q", "q_data", "double", ("3",), index_offsets=(1,),
                       read_only=True),
            KokkosView("w", "w_data", "double", ("3", "3"),
                       index_offsets=(1, 1), read_only=True),
            KokkosView("f", "f_data", "double", ("6",), index_offsets=(1,),
                       read_only=True),
            KokkosView("g", "g_data", "double", ("2", "3"),
                       index_offsets=(1, 1)),
            KokkosScalar("ncells", "int"),
        ))


def _lowering(body):
    """Return a lowering over a writer holding the probe's arrays.

    Driven directly rather than through a region, as
    :py:func:`_written_expressions` is and for the same reason: a value the
    region asks for by name is lowered the same way whatever launch shape
    encloses it, and the paths a caller reaches by naming no destination are
    not reachable through a statement at all.

    :param str body: the statements, indented, over the probe's arrays.

    :returns: the lowering, and the statements it was built over.
    :rtype: tuple[
        :py:class:`psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpression`,
        list[:py:class:`psyclone.psyir.nodes.Assignment`]]
    """
    region = _array_region(body)
    writer = KokkosWriter()
    writer._views = {argument.name: argument
                     for argument in region.arguments
                     if isinstance(argument, KokkosView)}
    writer._kind_types = dict(region.kind_types)
    return writer.array_expressions, region.schedule.walk(Assignment)


def test_kae_lowers_a_whole_array_assignment():
    """``a(:) = b(:) + 1.0`` becomes one loop and allocates nothing.

    The extents of a whole dimension are the View's own, so the loop runs
    from the array's declared origin for its declared extent without the
    ``LBOUND`` and ``UBOUND`` calls the frontend wrote there, neither of
    which this back-end can generate.
    """
    code = KokkosWriter()(_array_region("  a(:) = b(:) + 1.0_r_def\n"))

    assert code.count("for (int _kae_i") == 1
    assert "for (int _kae_i0 = 1; _kae_i0 <= (1 + nlayers - 1); _kae_i0++)" \
        in code
    assert "a((_kae_i0 - 1)) = (b((_kae_i0 - 1)) + 1.0);" in code
    assert "_kae_tmp" not in code


def test_kae_lowers_a_rank_2_assignment():
    """A rank-2 assignment nests its loops in the layout's own fast order.

    Every View this back-end declares is ``Kokkos::LayoutLeft``, whose
    leading dimension is the contiguous one, so the leading subscript is the
    one that must vary fastest and therefore the innermost loop. A nest
    written the other way round computes exactly the same answer while
    striding across memory on every step, so no later gate would catch it:
    the order is asserted here or nowhere.
    """
    code = KokkosWriter()(_array_region("  m(:,:) = 0.0_r_def\n"))

    assert code.count("for (int _kae_i") == 2
    assert code.index("for (int _kae_i1") < code.index("for (int _kae_i0")
    assert "for (int _kae_i1 = 1; _kae_i1 <= (1 + nlayers - 1); _kae_i1++)" \
        in code
    assert "for (int _kae_i0 = 1; _kae_i0 <= (1 + 3 - 1); _kae_i0++)" in code
    assert "m((_kae_i0 - 1), (_kae_i1 - 1)) = 0.0;" in code


def test_kae_passes_a_contiguous_section_as_a_subview():
    """A slice taking the leading dimensions whole is a View, not a copy.

    ``Kokkos::subview`` over those dimensions of a ``LayoutLeft`` View names
    the same storage, so nothing is allocated and no element is read: the
    result is a rank-2 View that a callee's formal takes directly.
    """
    lowering, statements = _lowering("  m(:,1) = jac(:,1,df)\n")

    text = lowering.lower(statements[0].rhs)

    assert "auto _kae_sub0 = Kokkos::subview(jac, Kokkos::ALL, (1 - 1), " \
        "(df - 1));" in text
    assert "for (" not in text
    assert lowering.result == "_kae_sub0"
    assert not lowering.temporaries


def test_kae_copies_a_non_contiguous_section():
    """A slice that is not contiguous is gathered, and says so in the source.

    ``jac(df,:,:)`` skips the leading dimension, which is the contiguous one
    in ``LayoutLeft``, so no subview describes it. The elements are copied
    into a temporary instead, and the generated C++ carries the reason: a
    reader who sees a nest where the other section got a subview would
    otherwise have to read this back-end to find out why.
    """
    lowering, statements = _lowering("  v(:,:) = jac(df,:,:)\n")

    text = lowering.lower(statements[0].rhs)

    assert "is copied rather than passed as a Kokkos::subview" in text
    assert "contiguous in this View's LayoutLeft" in text
    # Named in the comment and nowhere else: no subview is taken.
    assert "auto _kae_sub" not in text
    assert lowering.result == "_kae_tmp0"
    assert text.count("for (int _kae_i") == 2
    assert "_kae_tmp0((_kae_i0 - 1), (_kae_i1 - 1)) = " \
        "jac((df - 1), (_kae_i0 - 1), (_kae_i1 - 1));" in text


def test_kae_allocates_a_temporary_for_an_unnamed_result():
    """A value with nowhere to go is written into scratch of its own shape.

    The temporary is a
    :py:class:`~psyclone.psyir.backend.kokkos_array_expression.KokkosScratch`
    rather than a declaration of this class's own, because that is what the
    launch asks the run time for: an array the body allocated behind the
    launch's back would not be in its ``shmem_size`` sum.
    """
    lowering, statements = _lowering("  a(:) = b(:) + 1.0_r_def\n")
    expression = statements[0].rhs

    text = lowering.lower(expression)

    assert lowering.result == "_kae_tmp0"
    assert len(lowering.temporaries) == 1
    temporary = lowering.temporaries[0]
    assert isinstance(temporary, KokkosScratch)
    assert temporary.name == "_kae_tmp0"
    assert temporary.c_type == "double"
    assert temporary.extents == lowering.ranks(expression) == ("nlayers",)
    assert temporary.index_offsets == ("1",)
    assert "_kae_tmp0((_kae_i0 - 1)) = (b((_kae_i0 - 1)) + 1.0);" in text


def test_kae_refuses_a_loop_carried_dependency():
    """A section reading what it writes elsewhere is refused, by name.

    Fortran evaluates the whole right-hand side before assigning any of it,
    and a nest does not, so there is no order of the generated loops that
    means what the Fortran meant. The refusal names the array and both of the
    subscripts, because "a loop-carried dependency" alone leaves the reader
    to find which of the statement's arrays carries it.
    """
    lowering, statements = _lowering(
        "  a(2:nlayers) = a(1:nlayers - 1)\n")
    assignment = statements[0]

    with pytest.raises(VisitorError) as error:
        lowering.lower(assignment.rhs, assignment.lhs)

    assert "'a' is written at 'a(2:)' and read at 'a(:nlayers - 1)'" \
        in str(error.value)
    assert "dependence from one iteration to the next" in str(error.value)


def test_kae_refuses_an_expression_that_is_not_array_valued():
    """A scalar with nowhere to be spread names no space, and is refused.

    A caller that has to be told the shape of what it asked for is a caller
    that asked for the wrong thing, so this is a back-end error rather than
    an empty result: an empty nest around a scalar would generate C++ that
    compiles and assigns nothing.
    """
    lowering, statements = _lowering("  a(1) = b(1) + 1.0_r_def\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs)

    assert "is not an array-valued expression" in str(error.value)


def test_kae_slices_no_array_the_region_did_not_describe():
    """An array with no View to slice is not a subview, whatever its shape.

    ``Kokkos::subview`` takes a View, and an array the region did not
    describe is not one. The shape asked about here is contiguous in every
    other respect, so the description is the only thing deciding it, and the
    judgement is asserted directly: a body that read an undescribed array
    would be refused before it reached the lowering, which leaves nothing
    but this to keep a later caller from generating a subview of a name the
    generated C++ never declares.
    """
    lowering, statements = _lowering("  v(:,:) = 1.0_r_def\n")
    # pylint: disable=protected-access
    assert lowering._is_contiguous(statements[0].lhs) is False


def test_kae_copies_a_partial_section():
    """A slice taking no dimension whole is copied rather than sliced.

    ``Kokkos::subview`` is generated with ``Kokkos::ALL`` and nothing
    narrower, so a section with bounds of its own has no subview to be, even
    where the elements it names happen to be adjacent.
    """
    lowering, statements = _lowering(
        "  a(2:nlayers - 1) = b(2:nlayers - 1)\n")

    text = lowering.lower(statements[0].rhs)

    assert "auto _kae_sub" not in text
    assert lowering.result == "_kae_tmp0"
    assert lowering.ranks(statements[0].rhs) \
        == ("(((nlayers - 1)) - (2) + 1)",)


def test_kae_copies_a_section_whose_leading_dimension_is_whole():
    """A whole leading dimension does not save a narrowed trailing one.

    ``m(:,1:2)`` is contiguous in ``LayoutLeft`` and could in principle be
    sliced, but only by a subview carrying bounds this back-end does not
    generate, so it is copied. The refusal is the narrowness of the second
    subscript and not the order of the two.
    """
    lowering, statements = _lowering("  m(:,1:2) = 0.0_r_def\n")

    text = lowering.lower(statements[0].lhs)

    assert "auto _kae_sub" not in text
    assert lowering.result == "_kae_tmp0"
    assert "_kae_tmp0((_kae_i0 - 1), (_kae_i1 - 1)) = " \
        "m((_kae_i0 - 1), (_kae_i1 - 1));" in text


def test_kae_refuses_a_temporary_whose_kind_the_region_did_not_describe():
    """A temporary of an undescribed kind is refused rather than guessed.

    The alternative is a View of some default type, which would compile, and
    would silently narrow or widen every value the array carried.
    """
    lowering, statements = _lowering("  a(:) = b(:) + 1.0_r_def\n")
    # pylint: disable=protected-access
    lowering._writer._kind_types = {}

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs)

    assert "the region described no C type for its kind" in str(error.value)


def test_kokkos_writer_refuses_an_array_it_was_given_no_description_of():
    """An array reference the region did not describe is an error.

    Every array the body reads is a View, a scratch array or a constant, and
    the region names all three. One that is none of them has no origin, so
    there is no subscript to generate rather than a wrong one to guess.
    """
    lowering, statements = _lowering("  v(:,:) = 1.0_r_def\n")
    # pylint: disable=protected-access
    writer = lowering._writer

    with pytest.raises(ValueError) as error:
        writer._visit(statements[0].lhs.copy())

    assert "Array 'v' has no Kokkos View description." in str(error.value)


def test_kokkos_writer_refuses_an_array_described_with_the_wrong_rank():
    """A description whose offsets do not count the subscripts is an error.

    Each subscript is shifted by the offset beside it, so a description that
    supplied a different number of them would shift some subscripts and
    leave the rest, which is an access to the wrong element rather than a
    failure to generate one.
    """
    lowering, statements = _lowering("  a(1) = b(1) + 1.0_r_def\n")
    # pylint: disable=protected-access
    writer = lowering._writer
    writer._views["a"] = KokkosView("a", "a_data", "double",
                                    ("3", "nlayers"), index_offsets=(1, 1))

    with pytest.raises(ValueError) as error:
        writer._visit(statements[0].lhs.copy())

    assert "Array 'a' has 1 kernel indices but 2 offsets were supplied." \
        in str(error.value)


# ---------------------------------------------------------------------------
# The array-valued intrinsic tier
# ---------------------------------------------------------------------------
def test_kokkos_matmul_rank2_by_rank1():
    """``MATMUL`` of a matrix and a vector becomes a nest and a reduction.

    The result is one loop over the matrix's leading dimension, and inside it
    a scalar accumulator summed over the contracted dimension. Nothing is
    allocated: an element of a contraction is a scalar, so there is no
    intermediate array for the temporary a naive lowering would need.
    """
    lowering, statements = _lowering("  p(:) = matmul(w, q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "for (int _kae_i0 = 1; _kae_i0 <= (1 + 3 - 1); _kae_i0++) {" in text
    assert "double _kae_r0 = Kokkos::reduction_identity<double>::sum();" \
        in text
    assert "for (int _kae_j0 = 1; _kae_j0 <= (1 + 3 - 1); _kae_j0++) {" in text
    assert "_kae_r0 += w((_kae_i0 - 1), (_kae_j0 - 1)) * q((_kae_j0 - 1));" \
        in text
    assert "p((_kae_i0 - 1)) = _kae_r0;" in text
    assert "_kae_tmp" not in text
    assert not lowering.temporaries


def test_kokkos_matmul_rank2_by_rank2():
    """``MATMUL`` of two matrices nests two result loops round one sum.

    The result's dimensions come from different operands -- the first from
    the left matrix and the second from the right -- so a shape taken from
    either operand alone would be wrong for a non-square product.
    """
    lowering, statements = _lowering("  v(:,:) = matmul(w, w)\n")

    text = lowering.lower(statements[0].rhs)

    assert text.count("for (int _kae_i") == 2
    assert "_kae_r0 += w((_kae_i0 - 1), (_kae_j0 - 1)) * " \
        "w((_kae_j0 - 1), (_kae_i1 - 1));" in text


def test_kokkos_matmul_rank1_by_rank2():
    """``MATMUL`` of a vector and a matrix takes its shape from the matrix."""
    lowering, statements = _lowering("  p(:) = matmul(q, w)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_i") == 1
    assert "_kae_r0 += q((_kae_j0 - 1)) * w((_kae_j0 - 1), (_kae_i0 - 1));" \
        in text


def test_kokkos_matmul_of_a_transpose_emits_no_temporary():
    """``MATMUL(TRANSPOSE(m), x)`` swaps subscripts, it does not transpose.

    Materialising the transpose would need a rank-2 temporary, which the
    launch would have to have asked the run time for before the body that
    discovered it. Swapping the two subscripts of the operand computes the
    same product out of the storage the region already has, so this shape --
    which the model writes wherever it changes basis -- costs nothing.
    """
    lowering, statements = _lowering("  p(:) = matmul(transpose(w), q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "_kae_r0 += w((_kae_j0 - 1), (_kae_i0 - 1)) * q((_kae_j0 - 1));" \
        in text
    assert "_kae_tmp" not in text
    assert not lowering.temporaries


def test_kokkos_transpose_on_its_own_swaps_the_shape():
    """A bare ``TRANSPOSE`` is a nest whose shape is its operand's, reversed.

    Nothing in the model writes one -- every ``TRANSPOSE`` it has feeds a
    ``MATMUL`` -- but a shape that is only ever reached through another
    intrinsic is a shape no test would otherwise state.
    """
    lowering, statements = _lowering("  v(:,:) = transpose(m)\n")

    assert lowering.ranks(statements[0].rhs) == ("nlayers", "3")


def test_kokkos_dot_product():
    """``DOT_PRODUCT`` is a scalar, so it is a reduction and no nest at all.

    The statement it belongs to is an ordinary scalar assignment, generated
    after the loop that accumulates the sum rather than inside one.
    """
    lowering, statements = _lowering("  x = dot_product(p, q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "for (int _kae_i" not in text
    assert "double _kae_r0 = Kokkos::reduction_identity<double>::sum();" \
        in text
    assert "for (int _kae_j0 = 1; _kae_j0 <= (1 + 3 - 1); _kae_j0++) {" in text
    assert "_kae_r0 += p((_kae_j0 - 1)) * q((_kae_j0 - 1));" in text
    assert "x = _kae_r0;" in text


def test_kokkos_dot_product_inside_a_larger_expression():
    """A reduction is hoisted out of the expression that reads its value.

    The generated accumulator is a name, so the rest of the statement is
    written exactly as it would have been: the reduction's loop cannot appear
    where the operand did, because a loop is not an expression in C++.
    """
    lowering, statements = _lowering(
        "  x = 0.5_r_def * dot_product(p, q) + b(1)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "x = ((0.5 * _kae_r0) + b((1 - 1)));" in text


def test_kokkos_minval_uses_the_reduction_identity():
    """A reduction starts from ``Kokkos::reduction_identity``, not a literal.

    A written ``-DBL_MAX`` would be wrong for every kind but one and would
    have to be spelt again for each; the trait is ``constexpr`` and device
    callable, and is right for whatever type the accumulator has.
    """
    lowering, statements = _lowering("  x = minval(b)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "double _kae_r0 = Kokkos::reduction_identity<double>::min();" \
        in text
    assert "_kae_r0 = Kokkos::min(_kae_r0, b((_kae_j0 - 1)));" in text
    assert "x = _kae_r0;" in text


def test_kokkos_maxval_uses_the_reduction_identity():
    """``MAXVAL`` takes the matching identity and the matching fold."""
    lowering, statements = _lowering("  x = maxval(b)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "double _kae_r0 = Kokkos::reduction_identity<double>::max();" \
        in text
    assert "_kae_r0 = Kokkos::max(_kae_r0, b((_kae_j0 - 1)));" in text


def test_kokkos_sum_reduces_every_dimension_of_a_rank_2_operand():
    """A whole-array ``SUM`` runs one loop per dimension of its operand."""
    lowering, statements = _lowering("  x = sum(m)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_j") == 2
    assert "_kae_r0 += m((_kae_j0 - 1), (_kae_j1 - 1));" in text


def test_kokkos_sum_with_a_dim_argument():
    """``SUM(x, dim=)`` reduces one axis and leaves the rest as a shape.

    The result is array-valued, so it is a nest over the dimensions that
    survive with the reduction inside it, rather than the single accumulator
    a whole-array ``SUM`` gives.
    """
    lowering, statements = _lowering("  p(:) = sum(m, dim=2)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert lowering.ranks(statements[0].rhs) == ("3",)
    assert "for (int _kae_i0 = 1; _kae_i0 <= (1 + 3 - 1); _kae_i0++) {" in text
    assert "for (int _kae_j0 = 1; _kae_j0 <= (1 + nlayers - 1); _kae_j0++)" \
        in text
    assert "_kae_r0 += m((_kae_i0 - 1), (_kae_j0 - 1));" in text
    assert "p((_kae_i0 - 1)) = _kae_r0;" in text


def test_kokkos_reduction_refuses_a_dim_that_is_not_a_literal():
    """A ``dim`` that is not a constant names no axis at generation time."""
    lowering, statements = _lowering("  p(:) = sum(m, dim=df)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'SUM' takes its 'dim' from 'df'" in str(error.value)
    assert "a literal" in str(error.value)


def test_kokkos_reduction_refuses_a_dim_out_of_range():
    """A ``dim`` larger than the operand's rank names no axis at all."""
    lowering, statements = _lowering("  p(:) = sum(m, dim=3)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'dim' of 3" in str(error.value)
    assert "rank 2" in str(error.value)


def test_kokkos_matmul_refuses_operands_that_are_not_matrices():
    """``MATMUL`` of two vectors is not Fortran, and is refused as such."""
    lowering, statements = _lowering("  x = matmul(q, q)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'MATMUL' of a rank-1 and a rank-1 operand" in str(error.value)


def test_kokkos_array_intrinsic_refuses_an_undescribed_operand():
    """An operand with no View description has no extents to loop over."""
    lowering, statements = _lowering("  x = sum(v)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'v'" in str(error.value)
    assert "the region described no array of that name" in str(error.value)


def test_kokkos_array_intrinsic_refuses_a_nested_reduction():
    """A reduction inside a reduction is refused rather than mis-nested.

    Its accumulator would have to be declared outside the loop that indexes
    it, which is a second hoisting this tier does not perform. No kernel in
    the model writes one.
    """
    lowering, statements = _lowering("  x = sum(matmul(w, q))\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'MATMUL' inside 'SUM'" in str(error.value)


def test_kokkos_array_intrinsic_refuses_an_undescribed_kind():
    """An accumulator of an undescribed kind is refused, not guessed."""
    lowering, statements = _lowering("  x = dot_product(p, q)\n")
    # pylint: disable=protected-access
    lowering._writer._kind_types = {}

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'DOT_PRODUCT'" in str(error.value)
    assert "the region described no C type for its kind" in str(error.value)


def test_kokkos_reshape_as_a_review_when_contiguous():
    """A contiguous ``RESHAPE`` is an index map over the source's storage.

    ``LayoutLeft`` and Fortran's column-major element order agree, so the
    element the reshaped array holds at one subscript is the element the
    source holds at the same linear position. Nothing is copied: the
    subscript arithmetic is generated instead.
    """
    lowering, statements = _lowering("  g(:,:) = reshape(f, [2,3])\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "g((_kae_i0 - 1), (_kae_i1 - 1)) = " \
        "f(((1 + ((_kae_i0 - 1) + 2 * (_kae_i1 - 1))) - 1));" in text
    assert "_kae_tmp" not in text


def test_kokkos_reshape_refused_when_reordering():
    """A ``RESHAPE`` with an ``order`` is not the identity map, and is refused.

    ``pad`` is refused with it: both change which source element a result
    subscript names, and the whole reason a reshape costs nothing here is
    that it changes neither.
    """
    lowering, statements = _lowering(
        "  g(:,:) = reshape(f, [2,3], order=[2,1])\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'RESHAPE' with a 'pad' or an 'order'" in str(error.value)


def test_kokkos_reshape_refused_when_the_source_is_not_rank_1():
    """A reshape of a rank-2 source is refused, with the rank named."""
    lowering, statements = _lowering("  p(:) = reshape(w, [9])\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'RESHAPE' of a rank-2 source" in str(error.value)


def test_kokkos_reshape_refused_when_the_shape_is_not_a_constructor():
    """A reshape whose shape is a named array has no extents to read."""
    lowering, statements = _lowering("  g(:,:) = reshape(f, p)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'RESHAPE' takes its shape from 'p'" in str(error.value)


def test_kokkos_epsilon():
    """``EPSILON`` generates inside a region, whatever a bare probe says.

    The survey's probe replaces every argument with a kindless literal, which
    leaves ``EPSILON`` with no width to instantiate its trait at and so
    reports it unsupported. Every kind a region uses is on its ABI, so the
    generated region has the width and the intrinsic is written.
    """
    code = KokkosWriter()(_array_region("  a(:) = b(:) + epsilon(b(1))\n"))

    assert "Kokkos::Experimental::epsilon_v<double>" in code


def test_kokkos_nint_with_a_kind_argument():
    """``NINT(x, kind)`` is the one-argument cast with its kind checked.

    The C writer's cast formatter takes exactly one child, so the
    two-argument form reached it and was refused. The kind is not generated
    -- the cast is already ``(int)`` -- but it is checked, because a kind the
    region maps to some other width would be silently discarded otherwise.
    """
    assert _written_expressions("  i = nint(a, i_def)\n") \
        == ["(int)Kokkos::round(a)"]

    with pytest.raises(VisitorError) as error:
        _written_expressions("  i = nint(a, i_def)\n",
                             (("i_def", "long"), ("r_def", "double")))
    assert "'NINT' is written as a cast to 'int'" in str(error.value)
    assert "'long'" in str(error.value)


def test_kokkos_writer_reports_an_intrinsic_it_cannot_spell():
    """The writer answers which of a body's intrinsics it cannot write.

    ``validate`` has to refuse a body the writer will refuse, and the only
    thing that knows what the writer can spell is the writer: a second list
    beside it would be a list to keep in step. Every argument is replaced by
    a reference of its own type, so that an intrinsic answered from its
    argument's kind is still answered while an array the region never
    described is not looked up.
    """
    schedule = _array_probe("""
  a(:) = b(:) + epsilon(b(1))
  x = dot_product(p, q)
  x = tiny(x)
  a(1) = sqrt(b(1))
""")

    refusals = KokkosWriter().unsupported_intrinsics(
        schedule, _ARRAY_KINDS)

    assert list(refusals) == ["TINY/1"]


def test_kokkos_writer_reports_nothing_for_a_body_it_can_write():
    """A body of intrinsics the writer knows reports none."""
    schedule = _array_probe("  a(:) = matmul(w, q(1:3)) * sqrt(b(:))\n")

    refusals = KokkosWriter().unsupported_intrinsics(schedule, _ARRAY_KINDS)

    assert isinstance(refusals, tuple)
    assert not refusals


def test_kokkos_writer_reports_an_array_intrinsic_out_of_its_tier():
    """A reduction the array tier does not write is reported, not skipped.

    The tier generates a loop nest over an assignment's destination, so it
    writes such an intrinsic on a right-hand side and nowhere else.
    ``edge_lump_w2_mass_matrix_code`` tests for a domain edge with
    ``if (minval(smap_sizes) == 1)``, which no tier writes: skipping it for
    its name alone let ``validate`` accept a kernel ``apply`` then refused,
    and the whole-model capture gate says so.
    """
    schedule = _array_probe("""
  if (minval(p) == 1) then
    x = 1.0
  end if
""")

    refusals = KokkosWriter().unsupported_intrinsics(schedule, _ARRAY_KINDS)

    assert list(refusals) == ["MINVAL/1"]

    # On a right-hand side the same call is the tier's, and is not reported.
    schedule = _array_probe("  x = minval(p)\n")

    assert not KokkosWriter().unsupported_intrinsics(schedule, _ARRAY_KINDS)


def test_kokkos_array_intrinsic_over_a_section():
    """An operand may be a section, which is the form the model writes.

    ``matmul(m3(ik,:,:), p_e)`` and ``sum(t(1:n))`` take their extents from
    the ranges the subscript states rather than from the whole View, and the
    subscripts the section fixes are carried through to every element the
    generated loop reads.
    """
    lowering, statements = _lowering("  x = sum(m(:, 1))\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_j") == 1
    assert "_kae_j0 <= (1 + 3 - 1)" in text
    assert "_kae_r0 += m((_kae_j0 - 1), (1 - 1));" in text


def test_kokkos_array_intrinsic_refuses_an_expression_operand():
    """An operand that is computed rather than stored has no elements.

    Reading one element of it would mean generating the expression once per
    index, which is a rewrite of the operand rather than a subscript of it,
    so this tier takes a whole array or a section of one and nothing else.
    """
    lowering, statements = _lowering("  x = sum(q + b(1))\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "must be a whole array or a section of one" in str(error.value)


def test_kokkos_array_intrinsic_refuses_an_element_of_an_expression():
    """The same rule holds where the shape came from somewhere else.

    ``element`` is reached with whatever the shape was taken from, so it
    restates the rule rather than trusting it: an operand this tier accepted
    the shape of and cannot subscript would otherwise generate a loop with
    no body.
    """
    lowering, statements = _lowering("  x = sum(q + b(1))\n")
    # pylint: disable=protected-access
    intrinsics = lowering._intrinsics

    with pytest.raises(VisitorError) as error:
        intrinsics.element(statements[0].rhs.arguments[0], ["_kae_j0"])

    assert "cannot read an element of" in str(error.value)


def test_kokkos_transpose_refuses_an_operand_that_is_not_a_matrix():
    """``TRANSPOSE`` is defined over a matrix and over nothing else."""
    lowering, statements = _lowering("  p(:) = transpose(q)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'TRANSPOSE' of a rank-1 operand" in str(error.value)


def test_kokkos_reduction_refuses_an_argument_beside_its_dim():
    """A ``mask`` folds part of a dimension, which this tier cannot write.

    The generated loop runs over whole extents and accumulates every element
    it reads. A mask would have to become a condition inside it, evaluated
    from an array the reduction never declared it read, so it is refused by
    name rather than dropped.
    """
    lowering, statements = _lowering("  x = sum(q, mask=q > 0.0_r_def)\n")

    with pytest.raises(VisitorError) as error:
        lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "'SUM' with a 'mask' argument" in str(error.value)


def test_kokkos_array_intrinsic_inside_an_intrinsic_of_its_own():
    """A reduction under an elemental intrinsic is still hoisted.

    The hoist looks for the outermost *handled* call, not the outermost call:
    ``abs(sum(q))`` has no enclosing reduction to nest inside, so the sum is
    hoisted and the ``abs`` is written over the scalar it leaves behind.
    """
    lowering, statements = _lowering("  x = abs(sum(q))\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "double _kae_r0 = Kokkos::reduction_identity<double>::sum();" \
        in text
    assert "x = Kokkos::abs(_kae_r0);" in text


def test_kae_steps_over_the_sections_a_reduction_consumed():
    """A section inside a scalar reduction is not a section of the statement.

    ``a(:) = b(:) * dot_product(m(:,1), q)`` is rank 1 and the reduction's
    operand is a section of a different array with a different extent. The
    shape has to come from the sections the statement itself has, so the
    operands of the reduction are stepped over while looking for one.
    """
    lowering, statements = _lowering(
        "  a(:) = b(:) * dot_product(m(:, 1), q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_i") == 1
    assert "_kae_i0 = 1; _kae_i0 <= (1 + nlayers - 1)" in text
    assert "a((_kae_i0 - 1)) = (b((_kae_i0 - 1)) * _kae_r0);" in text


# ---------------------------------------------------------------------------
# A whole-array destination, written without subscripts
# ---------------------------------------------------------------------------
def test_kae_arithmetic_over_contractions_indexes_the_destination():
    """``lhs = matmul(..) - matmul(..) + matmul(..)`` writes one element.

    This is the shape ``apply_elim_mixed_lp_operator_kernel_mod`` writes, and
    it names its destination without subscripting it, which is what makes it
    different from every section this tier had been given. The nest still runs
    over the destination's own dimension, so the write inside it is to one
    element of the destination and not to the whole of it: a View has no
    ``operator=`` taking a scalar, so the alternative does not even compile.
    """
    lowering, statements = _lowering(
        "  p = matmul(w, q) - matmul(transpose(w), q) + matmul(w, f(1:3))\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_i") == 1
    assert "p((_kae_i0 - 1)) = ((_kae_r0 - _kae_r1) + _kae_r2);" in text
    # The defect this states was a write to the View itself rather than to an
    # element of it, so the absence of the unsubscripted name is the assertion
    # that matters and the presence of the subscripted one is not enough.
    assert "p = " not in text


def test_kae_a_section_destination_takes_the_same_arithmetic():
    """The same statement written as a section lowers the same way.

    A destination is a section or a whole array according to how the kernel
    spelt it, and the two spellings mean the same thing in Fortran, so they
    generate the same nest.
    """
    lowering, statements = _lowering(
        "  p(:) = matmul(w, q) - matmul(transpose(w), q)"
        " + matmul(w, f(1:3))\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "p((_kae_i0 - 1)) = ((_kae_r0 - _kae_r1) + _kae_r2);" in text
    assert "p = " not in text


def test_kae_a_whole_array_destination_of_one_contraction_is_indexed():
    """One intrinsic on the right-hand side is subscripted no differently.

    The destination's spelling is what decides this and the right-hand side's
    shape is not, so the single-intrinsic form is stated rather than left to
    be inferred from the arithmetic above it.
    """
    lowering, statements = _lowering("  p = matmul(w, q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "p((_kae_i0 - 1)) = _kae_r0;" in text
    assert "p = " not in text


def test_kae_a_scalar_term_beside_a_contraction_is_still_indexed():
    """A whole array read beside a contraction is read one element at a time.

    ``matmul(w, q) * 2.0 + q`` mixes a contraction, a scalar and a whole array
    in one expression. The contraction leaves a scalar behind, the literal is
    a scalar already, and the whole array is the one operand that has to
    follow the nest's index -- an unsubscripted View beside a ``double`` is
    not an addition C++ has.
    """
    lowering, statements = _lowering("  p = matmul(w, q) * 2.0_r_def + q\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "p((_kae_i0 - 1)) = ((_kae_r0 * 2.0) + q((_kae_i0 - 1)));" in text
    assert "p = " not in text
    assert "+ q)" not in text


def test_kae_a_destination_read_on_the_right_is_read_per_element():
    """``lhs = lhs + matmul(..)`` reads and writes the same element.

    Fortran evaluates the whole right-hand side before assigning any of it,
    and a nest does not; the two agree exactly where each iteration reads the
    element it writes, which is what a whole-array read of the destination
    becomes. So this is lowered rather than refused, and the element read is
    the element written.
    """
    lowering, statements = _lowering("  p = p + matmul(w, q)\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "p((_kae_i0 - 1)) = (p((_kae_i0 - 1)) + _kae_r0);" in text
    assert "p = " not in text


def test_kae_a_rank_two_whole_array_destination_takes_both_indices():
    """A destination of rank two is subscripted in both its dimensions.

    The number of subscripts a whole-array name stands for is its rank, so
    the rule is stated at a rank other than one: a fix that supplied the
    innermost index alone would pass every test above it and would write the
    wrong element here.
    """
    lowering, statements = _lowering("  v = matmul(w, w) + matmul(w, w)\n")
    lowering._writer._views["v"] = KokkosView(
        "v", "v_data", "double", ("3", "3"), index_offsets=(1, 1))

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert text.count("for (int _kae_i") == 2
    assert "v((_kae_i0 - 1), (_kae_i1 - 1)) = (_kae_r0 + _kae_r1);" in text
    assert "v = " not in text


def test_kae_leaves_a_whole_array_name_of_another_rank_alone():
    """A name the nest has no index for is not given one anyway.

    Only non-conforming source puts a rank-2 name in a rank-1 expression, and
    the nest that expression produced has one index rather than the two such
    a name would need. Inventing the second would write a wrong element where
    generating the name unchanged writes none, so the rank is checked before
    the subscripts are supplied rather than assumed from the shape.
    """
    lowering, statements = _lowering("  p(:) = b(1) * w\n")

    text = lowering.lower(statements[0].rhs, statements[0].lhs)

    assert "p((_kae_i0 - 1)) = (b((1 - 1)) * w);" in text


def test_kokkos_writer_inherits_the_integer_power_tree():
    """The Kokkos writer builds ``**`` into the product tree the C writer does.

    The tree is the C writer's and is inherited whole, and that is what is
    being asserted: the rounding fix belongs there, and the Kokkos writer must
    not acquire a second answer of its own on the way past a kernel's
    ``edge_height ** 3``.

    A single-precision base keeps its width. Every operand of the tree is the
    base itself, and the reciprocal's numerator is the integer one, so C++'s
    arithmetic conversions never widen the expression to double and round it
    back.

    What the tree does not write falls to a call, and only its *name* is the
    Kokkos writer's; the test below says why.
    """
    assert _written_expressions(
        "  a = b ** 2\n"
        "  a = b ** 3\n"
        "  s = s ** 3\n"
        "  a = b ** (-2)\n"
        "  a = b ** i\n"
        "  i = j ** 2\n") == [
            "(b * b)",
            "((b * b) * b)",
            "((s * s) * s)",
            "(1 / (b * b))",
            "Kokkos::pow(b, i)",
            "(j * j)"]


def test_kokkos_writer_writes_a_power_at_its_operands_width():
    """A power the tree does not write is ``Kokkos::pow``, not C's ``pow``.

    C's ``pow`` binds ``double`` for every argument it is given, so a
    single-precision power is computed in double and rounded back to
    ``float``, where gfortran calls ``powf`` and rounds once. The two disagree
    in the last bit often enough to move a whole-model checksum, which is what
    ``sample_eos_operators_code`` -- ``exner_cell ** onemk_over_k``, both
    ``r_solver`` -- did.

    ``Kokkos::pow`` is overloaded, so the width follows the operands. Both
    widths are asserted, because the fix would be worth nothing if it changed
    the double case: that one is the same library call under either name, and
    every region captured before this one relies on it.
    """
    assert _written_expressions(
        "  s = s ** s\n"
        "  a = b ** c\n") == [
            "Kokkos::pow(s, s)",
            "Kokkos::pow(b, c)"]
