# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the staging header and the spellings that call into it.

What this module can assert is what the generated *text* says: which role
each View is placed under, that the type the launch body is compiled against
is untouched, and that only a written View is copied back. What the three
modes *do* at run time is a property of the header, and is asserted where a
compiler and a running program are available: ``tests/test_generated_cxx_build
.sh`` in the coordinating repository compiles this header and runs one region
under each mode, and ``tests/test_cuda_region_build.sh`` compiles it for a
device. Those gates read :py:func:`header_text` from here rather than
carrying a copy, so the text asserted is the text the regions are generated
against.

"""

from types import SimpleNamespace

import pytest

from psyclone.psyir.backend.kokkos import KokkosView
from psyclone.psyir.backend.kokkos_staging import (
    HEADER_NAME, ROLES, header_text, include_line, release_statement, role_of,
    stage_declaration, staging_epilogue, unstage_statement, view_declaration)


def _view(name="x", **kwargs):
    """A View description, read-only unless told otherwise."""
    arguments = {"c_type": "double", "extents": ("undf",), "read_only": True,
                 "random_access": True}
    arguments.update(kwargs)
    return KokkosView(name, f"{name}_data", **arguments)


def test_kokkos_staging_roles_are_the_four_kinds():
    """The roles are the four the LFRic transformations decide between."""
    assert ROLES == ("field", "readonly", "readwrite", "transient")


def test_kokkos_staging_header_is_named_for_lfric():
    """The header's name is what the generated include names."""
    assert HEADER_NAME == "lfric_kokkos_staging.hpp"
    assert include_line() == '#include "lfric_kokkos_staging.hpp"'


def test_kokkos_staging_header_is_a_self_contained_header():
    """The header guards itself, includes Kokkos and closes its namespace."""
    text = header_text()

    assert text.startswith("//")
    assert text.endswith("\n")
    assert "#ifndef LFRIC_KOKKOS_STAGING_HPP\n#define " \
        "LFRIC_KOKKOS_STAGING_HPP\n" in text
    assert text.rstrip().endswith("#endif  // LFRIC_KOKKOS_STAGING_HPP")
    assert "#include <Kokkos_Core.hpp>" in text
    assert "namespace lfric_kokkos {" in text
    assert "}  // namespace lfric_kokkos" in text


def test_kokkos_staging_header_declares_every_role_the_writer_spells():
    """The enumerators are the roles, so the two cannot drift apart.

    A role the writer spells and the header does not declare is a region
    that does not compile; a role the header declares and the writer never
    spells is a mode nothing reaches. Both are caught by reading the two
    against each other here rather than by waiting for a compiler.
    """
    text = header_text()

    for role in ROLES:
        assert f"\n  {role}," in text or f"\n  {role}   " in text
    assert "enum class Role {" in text


def test_kokkos_staging_header_offers_the_three_modes():
    """The header reads the mode from the environment, and names all three."""
    text = header_text()

    assert 'std::getenv("LFRIC_KOKKOS_STAGING")' in text
    assert 'std::getenv("LFRIC_KOKKOS_STAGING_CACHE")' in text
    assert "enum class Mode { none, all, non_field };" in text
    # Unset is 'none', which is what makes every build that has not asked
    # for staging generate and run exactly as it did before.
    assert "setting == nullptr" in text
    assert 'std::strcmp(setting, "all")' in text
    assert 'std::strcmp(setting, "non-field")' in text
    # 'all' says once, on stderr, that its timings are not measurements.
    assert "timings taken in it mean " in text
    # An unrecognised setting is a stop, not a silent fallback to none.
    assert "Kokkos::abort(" in text


def test_kokkos_staging_header_releases_before_kokkos_finalises():
    """The cache is emptied by a finalize hook, not by a static destructor.

    A Kokkos View destroyed after ``Kokkos::finalize`` is an error Kokkos
    reports and a run that ends badly; the state this header holds therefore
    outlives static destruction and is emptied from a hook instead.
    """
    text = header_text()

    assert "Kokkos::push_finalize_hook(" in text
    assert "drop_all(current, current.cache, false);" in text
    assert "empty_pools(current);" in text


def test_kokkos_staging_header_counts_every_role_separately():
    """The finalise report splits what staging did by role.

    Totals say how much copying a run did; only the split says which kind of
    argument it was for, and that is the measurement any decision about
    staging rests on. One line per role, printed whether or not the role was
    met, so that a reader and a parser find the same four rows in every run.
    """
    text = header_text()

    assert "struct Counters {" in text
    assert "Counters role[role_count];" in text
    assert "constexpr int role_count = 4;" in text
    assert "staging_role=%s staged=%zu" in text
    assert "bytes_in=%zu bytes_out=%zu allocs=%zu frees=%zu" in text
    for role in ROLES:
        assert f'return "{role}";' in text
    # The totals are summed from the roles rather than counted a second
    # time, so the two lines of the report cannot disagree.
    assert "Counters total() const {" in text
    assert "for (int index = 0; index < role_count; ++index) {" in text


def test_kokkos_staging_header_counts_an_allocation_where_it_is_made():
    """Every allocation and every release is counted against its role.

    A block carries the role that took it so that the release -- which
    happens in ``release()`` and at finalise, far from the ``stage()`` call
    that asked for it -- is charged to the same role as the allocation.
    Allocations minus frees is then what the header still holds.
    """
    text = header_text()

    assert "Role role = Role::readonly;" in text
    assert "current.of(role).allocations += 1;" in text
    assert "current.of(block.role).frees += 1;" in text


def test_kokkos_staging_header_pools_released_buffers():
    """A released buffer is kept for the next argument of its size.

    The measurement this answers is that a transient table is allocated and
    released on every region call, so nine allocations in ten the header
    makes are for storage it released moments earlier. The pool hands that
    storage on instead. It is keyed by exact byte count, bounded, and can be
    turned off from the environment so that a run can be timed either way.
    """
    text = header_text()

    assert 'std::getenv("LFRIC_KOKKOS_STAGING_POOL")' in text
    assert 'std::getenv("LFRIC_KOKKOS_STAGING_POOL_MB")' in text
    assert "using Spares = std::map<std::size_t, std::vector<Spare>>;" in text
    assert "current.of(role).reuses += 1;" in text
    assert "current.of(block.role).recycled += 1;" in text
    assert "staging_pool=%s reuses=%zu recycled=%zu" in text
    # The bound is what stops a run meeting many distinct sizes from keeping
    # a spare of every one of them for ever.
    assert "current.pool_bytes + block.bytes <= pool_limit()" in text


def test_kokkos_staging_header_empties_the_pool_before_kokkos_goes():
    """Pooled buffers are released from the finalize hook, like the cache.

    A pooled buffer is a live Kokkos allocation that no region is using: it
    is held by the pool's ``shared_ptr``, and if that outlives
    ``Kokkos::finalize`` the run ends with the error Kokkos reports for a
    View released too late. The hook empties the pool for that reason, and
    counts what it released so the report balances.
    """
    text = header_text()

    assert "inline void empty_pools(State &current) {" in text
    assert "current.pool_released += entry.second.size();" in text
    # Finalise releases for real rather than recycling: there is nothing
    # left for a recycled buffer to be handed to.
    assert "inline void drop(State &current, Block &block, " \
        "bool recycle = true) {" in text
    assert "drop_all(current, current.live, false);" in text


def test_kokkos_staging_header_prefetches_a_field_behind_a_knob():
    """A field's pages are asked for before the launch, if the knob says so.

    A ``field`` role is the one role left over the caller's storage, which on
    a CUDA build is managed memory faulted onto the card 64 KB at a time by
    the launch that reads it. The prefetch turns those faults into one bulk
    copy on the stream the launch will use, so the region's existing fence is
    what orders it and nothing new is waited for. Off unless asked for: a
    prefetch of a field that is already resident costs the call and moves
    nothing, and which way that goes is a measurement.
    """
    text = header_text()

    assert 'std::getenv("LFRIC_KOKKOS_STAGING_PREFETCH")' in text
    assert 'std::getenv("LFRIC_KOKKOS_STAGING_PREFETCH_STRIDE")' in text
    # Unset is off, so a build that has not asked for it is the build it was.
    assert "return value != nullptr && value[0] != '\\0' &&\n" \
        "           std::strcmp(value, \"0\") != 0;" in text
    assert "cudaMemPrefetchAsync(const_cast<void *>(pointer), bytes, " \
        "on_device, 0," in text
    assert "stream_of(Kokkos::DefaultExecutionSpace())" in text
    assert "inline cudaStream_t stream_of(const Kokkos::Cuda &space) {" in text
    # Only the role whose storage stays where the caller put it, and only
    # when asked: every other role in every other mode is untouched.
    assert "if (role == Role::field && prefetching()) {" in text


def test_kokkos_staging_header_prefetch_is_cuda_only_and_include_ordered():
    """The prefetch is compiled only where there is a card to prefetch to.

    ``KOKKOS_ENABLE_CUDA`` is defined by ``Kokkos_Core.hpp``, which this
    header includes itself rather than relying on a region to have included
    it first: a knob switched off by an include order is exactly the silent
    null lever the measurement protocol exists to catch. On a host build the
    calls are not compiled at all, and a run that asked for the knob there
    says so by counting every staging as refused.
    """
    text = header_text()

    kokkos = text.index("#include <Kokkos_Core.hpp>")
    guard = text.index("#if defined(KOKKOS_ENABLE_CUDA)")
    assert kokkos < guard
    assert "#include <cuda_runtime_api.h>" in text
    assert text.index("#include <cuda_runtime_api.h>") > guard
    assert "#endif  // KOKKOS_ENABLE_CUDA" in text
    # The host arm of the prefetch itself: nothing to do, and it says so.
    assert "  (void)bytes;\n  current.prefetch_refused += 1;\n#endif" in text


def test_kokkos_staging_header_tolerates_a_pointer_it_cannot_prefetch():
    """A pointer that is not managed memory is refused, not fatal.

    A host-only build has no managed memory, and on a device build a field
    whose SharedSpace claim was refused falls back to a Fortran ALLOCATE, so
    a region may be handed a pointer the driver will not take. The call is
    counted, the sticky error it leaves is cleared so the next CUDA call is
    not blamed for it, and the answer is remembered per pointer so that such
    a run does not pay a driver query on every staging.
    """
    text = header_text()

    assert "cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&" \
        in text
    assert "attributes.type == cudaMemoryTypeManaged" in text
    assert "std::map<const void *, bool> managed;" in text
    assert text.count("(void)cudaGetLastError();") >= 3
    assert "current.prefetch_refused += 1;" in text
    # The stride's state: which staging a pointer was last prefetched at.
    assert "std::map<const void *, std::size_t> prefetched_at;" in text
    assert "current.prefetch_skipped += 1;" in text


def test_kokkos_staging_header_announces_what_the_prefetch_did():
    """The knob announces itself on the totals line, on or off.

    A knob that does nothing passes every correctness gate and reports a null
    lever, so the line is printed whenever the knob is on even in a mode that
    stages nothing at all, and it carries the counters a reader needs to tell
    "prefetching did not pay" from "prefetching did not happen": the calls
    issued, the bytes they covered, the calls the driver would not take, the
    calls the stride held back and the stagings they are read against.
    """
    text = header_text()

    assert "prefetch=%s prefetches=%zu prefetch_bytes=%zu " in text
    assert "prefetch_refused=%zu prefetch_skipped=%zu " in text
    assert "prefetch_stride=%zu field_stages=%zu " in text
    assert 'prefetching() ? "on" : "off"' in text
    assert "if (sum.staged + sum.cached > 0 || prefetching()) {" in text
    # The dedupe's own fields, ending the line: what it resolved to, the
    # repeats there were to take within a call and across the call boundary,
    # and the calls it did not issue.
    assert "prefetch_dedupe=%s prefetch_repeat_call=%zu " in text
    assert "prefetch_repeat_prev=%zu prefetch_deduped=%zu\\n" in text
    assert 'dedupe_prefetch() ? "on" : "off"' in text


def test_kokkos_staging_header_dedupes_a_repeated_field_in_one_call():
    """A field staged twice by one region call is prefetched once.

    An invoke may pass one field under two of a region's arguments -- a field
    vector's components, a built-in whose input and output are one field --
    and the second prefetch asks the driver to move a range it has just been
    asked to move. The dedupe is keyed by (pointer, bytes), as every other
    key in this header is, so two arguments over one base pointer with
    different lengths are still two ranges.
    """
    text = header_text()

    assert 'std::getenv("LFRIC_KOKKOS_STAGING_PREFETCH_DEDUPE")' in text
    assert "const Key range(pointer, bytes);" in text
    assert "const bool repeat_in_call = holds(current.call_prefetched, " \
        "range);" in text
    assert "    if (dedupe_prefetch()) {\n" \
        "      current.prefetch_deduped += 1;\n      return;" in text
    # Counted whether or not the knob is on: the count is what the knob was
    # built from, and a run with it off is the run that reports it.
    assert "current.prefetch_repeat_call += 1;" in text
    assert "current.prefetch_repeat_prev += 1;" in text


def test_kokkos_staging_header_drops_the_dedupe_at_the_end_of_a_call():
    """Nothing is ever skipped across a region call boundary.

    Between two regions the host writes fields, which is what put the pages
    back on the host in the first place, and wave 1 measured that skipping a
    prefetch on that account costs time. So the ranges this call prefetched
    are dropped at ``release()`` -- ahead of the mode check there, because
    ``none`` mode returns early from everything else and still prefetches --
    and the call before's are kept only to be counted against.
    """
    text = header_text()

    assert "inline void end_prefetch_call() {" in text
    assert "current.previous_prefetched.swap(current.call_prefetched);\n" \
        "  current.call_prefetched.clear();" in text
    release = text.index("inline void release() {")
    assert text.index("end_prefetch_call();", release) < \
        text.index("if (mode() == Mode::none) {", release)


def test_kokkos_staging_role_of_reads_a_stated_role():
    """A description that names a role is taken at its word."""
    assert role_of(_view(role="field")) == "field"
    assert role_of(_view(role="readonly")) == "readonly"
    assert role_of(_view(read_only=False, role="readwrite")) == "readwrite"


def test_kokkos_staging_role_of_falls_back_to_constness():
    """An unstated role is read from what the region does with the View.

    Never ``field``: that role says the caller allocated in a space the
    device shares, which is a claim about the caller and not about the
    description, and assuming it wrongly is a region that reads host memory
    from a kernel.
    """
    assert role_of(_view()) == "readonly"
    assert role_of(_view(read_only=False, random_access=False)) == "readwrite"


def test_kokkos_staging_role_of_refuses_a_role_it_does_not_know():
    """A role outside the four is a mistake, and is named as one."""
    with pytest.raises(ValueError) as error:
        role_of(_view(name="theta", role="halo"))

    assert "View 'theta' carries staging role 'halo'" in str(error.value)
    assert "['field', 'readonly', 'readwrite', 'transient']" in str(
        error.value)


def test_kokkos_staging_declaration_names_the_type_and_the_role():
    """A staged declaration is the View type, the pointer and the role."""
    assert stage_declaration(
        "Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, Unmanaged>",
        "theta", "theta_data", "field", ("undf_wtheta",)) == (
            "  auto theta = lfric_kokkos::stage<\n"
            "      Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged>>(\n"
            "      theta_data, lfric_kokkos::Role::field, undf_wtheta);")


def test_kokkos_staging_declaration_takes_every_extent():
    """Every dimension reaches the call, in order, after the role."""
    assert stage_declaration(
        "View", "map", "map_data", "readonly", ("ndf", "ncells")) == (
            "  auto map = lfric_kokkos::stage<\n"
            "      View>(\n"
            "      map_data, lfric_kokkos::Role::readonly, ndf, ncells);")


def test_kokkos_staging_declaration_takes_its_indentation():
    """The indentation is given, because a region body may be nested."""
    declaration = stage_declaration(
        "View", "x", "x_data", "readwrite", ("n",), indent="    ")

    assert declaration.startswith("    auto x = ")
    assert "\n        View>(\n" in declaration


def test_kokkos_staging_view_declaration_keeps_the_type_it_had():
    """Staging changes the memory a View covers, never the View's type.

    The launch body is compiled against this type: its element type, its
    layout, its rank and its memory traits. Every one of them is what it was
    before staging existed, which is what makes a body generated for one
    mode the body generated for the others.
    """
    read_only = view_declaration(
        _view(name="mr_v", extents=("undf_wtheta",), role="field"))
    assert ("Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly>>(") in read_only
    assert "mr_v_data, lfric_kokkos::Role::field, undf_wtheta);" in read_only

    written = view_declaration(_view(
        name="theta", read_only=False, random_access=False,
        extents=("undf_wtheta",), role="field"))
    assert ("Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged>>(") in written

    dofmap = view_declaration(_view(
        name="map_wtheta", c_type="int", extents=("ndf", "ncells"),
        role="readonly"))
    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly>>(") in dofmap

    operator_stencil = view_declaration(_view(
        name="matrix", read_only=False, random_access=False,
        extents=("ncell_3d", "ndf1", "ndf2"), role="readwrite"))
    assert ("matrix_data, lfric_kokkos::Role::readwrite, ncell_3d, ndf1, "
            "ndf2);") in operator_stencil


def test_kokkos_staging_epilogue_statements():
    """The write-back names the View and the storage it belongs to."""
    assert unstage_statement("theta", "theta_data") == \
        "  lfric_kokkos::unstage(theta, theta_data);\n"
    assert release_statement() == "  lfric_kokkos::release();\n"
    assert unstage_statement("x", "x_data", indent="    ") == \
        "    lfric_kokkos::unstage(x, x_data);\n"
    assert release_statement(indent="    ") == \
        "    lfric_kokkos::release();\n"


def test_kokkos_staging_epilogue_is_per_region():
    """A region writes back only what it wrote, and releases once.

    The release is not conditional on anything the region wrote: a read-only
    View staged into a device mirror is held for the call too, and in the
    cached modes the release is what decides whether it is kept or dropped.
    A region with no View at all emits neither statement, so that a region
    that never had an array to stage reads exactly as it did before.
    """
    written = _view(name="theta", read_only=False, random_access=False)
    read = _view(name="mr_v")
    scalar = SimpleNamespace(name="ncells")

    assert staging_epilogue(SimpleNamespace(arguments=(scalar,))) == ""
    assert staging_epilogue(SimpleNamespace(arguments=(read,))) == \
        release_statement()
    assert staging_epilogue(
        SimpleNamespace(arguments=(read, written, scalar))) == \
        unstage_statement("theta", "theta_data") + release_statement()
