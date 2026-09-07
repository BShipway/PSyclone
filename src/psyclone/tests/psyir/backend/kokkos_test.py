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
from psyclone.psyir.backend.kokkos_launch import range_launch, team_launch
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
    (KokkosScratch("x_new", "double", ("nlayers / 2",), index_offsets=(1,)),
     "Kokkos scratch 'x_new' has extent 'nlayers / 2' which is not an integer "
     "expression over named sizes."),
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
                   index_offsets=("nlayers / 2",)),
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
    """Return a one-dimensional file-scope constant description."""
    integer = ScalarType(ScalarType.Intrinsic.INTEGER,
                         ScalarType.Precision.UNDEFINED)
    fields = {"name": "x_dofs", "c_type": "int",
              "values": (Literal("1", integer), Literal("3", integer)),
              "index_offsets": (1,)}
    fields.update(kwargs)
    return KokkosConstant(**fields)


def test_kokkos_writer_declares_a_constant_at_file_scope():
    """A constant array is declared once, above the region, and not taken.

    The values are generated rather than held as text, so they cross the same
    literal path the body's own literals do; and nothing about the constant
    reaches the ABI, which is the point of describing it here rather than as
    an argument.
    """
    region = replace(_region(), constants=(_constant(),))

    generated = KokkosWriter()(region)

    assert generated.startswith(
        "#include <Kokkos_Core.hpp>\n\n"
        "static const int x_dofs[2] = {1, 3};\n\n"
        'extern "C" void moist_dyn_gas_kokkos(')
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
    """Every way of describing a file-scope constant wrongly is refused."""
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
    ("", False), ("   ", False), ("nlayers / 2", False), ("4nlayers", False),
    ("(nlayers", False), ("nlayers)", False), (")nlayers(", False),
    ("nlayers % 2", False), ("nlayers.size", False), (4, False), (None, False),
])
def test_is_extent(extent, accepted):
    """The extent predicate accepts arithmetic and refuses everything else.

    Division is refused rather than unsupported: Fortran and C++ can disagree
    about the rounding of an integer division, and an extent is a place where
    that would be silent. ``)nlayers(`` is here because a depth count that
    only checked the total would accept it.
    """
    assert is_extent(extent) is accepted


@pytest.mark.parametrize("offset, accepted", [
    (1, True), (0, True), (-3, True), ("1", True), ("0", True),
    ("(-nlayers)", True), ("2 * nlayers", True),
    (1.5, False), ("nlayers / 2", False), ("", False), (None, False),
])
def test_is_offset(offset, accepted):
    """An offset is an integer, or an extent expression standing in for one.

    The grammar is the extent grammar because an origin and an extent are two
    readings of one declaration: a shape admitted into the extent and refused
    in the origin would size a View correctly and index it from the wrong
    place. A negative integer is accepted, since an array centred on zero has
    a negative origin.
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
