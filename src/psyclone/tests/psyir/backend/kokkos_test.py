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
    KokkosRegion, KokkosScalar, KokkosScratch, KokkosView, KokkosWriter,
    extent_names, is_extent)
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

    # The team size depends on how much scratch a rank asks for, so it is
    # asked for rather than chosen, from a policy already carrying the
    # request. A probe without it would answer for a different launch.
    probe = code.index("TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO)")
    assert ".set_scratch_size(0, Kokkos::PerThread(scratch_bytes));" in \
        code[probe:]
    assert "const int team_size = probe.team_size_max(body, " \
        "Kokkos::ParallelForTag());" in code
    assert "const int league_size = (ncells + team_size - 1) / team_size;" \
        in code
    assert code.index("const int team_size") < code.index(
        'Kokkos::parallel_for("tri_solve_kokkos"')

    # The runtime guard and the fence belong to both shapes.
    assert "if (!Kokkos::is_initialized()) {" in code
    assert code.rstrip().endswith("Kokkos::fence();\n}")


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
                   "scratch_bytes", "thread_scratch", "team_size_max"):
        assert absent not in code


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
    (KokkosScratch("x_new", "double", ("nlayers",), index_offsets=("1",)),
     "Kokkos scratch 'x_new' index offsets must be integers."),
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
