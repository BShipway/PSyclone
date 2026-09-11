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

import pytest

from types import SimpleNamespace

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
    assert "current.cache.clear();" in text


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
