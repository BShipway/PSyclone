# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the module holding the staging header's text.

The text itself is asserted by ``kokkos_staging_test.py``, which reads it
through :py:func:`psyclone.psyir.backend.kokkos_staging.header_text` as every
caller does. What is asserted here is the split: that the accessor returns
what this module holds, and that it still does when the accessor's module is
loaded from source by path with no PSyclone package around it -- which is how
four shell gates of the coordinating repository take it, and the only reason
the two modules are reached the way they are.

"""

import importlib.util
from pathlib import Path

from psyclone.psyir.backend import kokkos_staging
from psyclone.psyir.backend.kokkos_staging_header import HEADER_TEXT


def _by_path():
    """The staging module loaded from source, the way the shell gates do.

    :returns: the module, with no package around it.
    :rtype: module

    """
    source = Path(kokkos_staging.__file__)
    spec = importlib.util.spec_from_file_location("kokkos_staging", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kokkos_staging_header_text_is_the_header():
    """The text is the C++ header, and the accessor strips its lead."""
    assert HEADER_TEXT.startswith("\n//")
    assert "#ifndef LFRIC_KOKKOS_STAGING_HPP" in HEADER_TEXT
    assert HEADER_TEXT.rstrip().endswith("#endif  // LFRIC_KOKKOS_STAGING_HPP")
    assert kokkos_staging.header_text() == HEADER_TEXT.lstrip("\n")


def test_kokkos_staging_header_holds_only_the_text():
    """Nothing else is published from here: the sibling holds the rest."""
    from psyclone.psyir.backend import kokkos_staging_header

    assert kokkos_staging_header.__all__ == ["HEADER_TEXT"]
    assert not [name for name in dir(kokkos_staging_header)
                if not name.startswith("__") and name != "HEADER_TEXT"]


def test_kokkos_staging_loaded_by_path_gives_the_same_header():
    """A by-path load finds the sibling beside it, and the same bytes.

    The four shell callers name the file and have no package: the fallback
    that reads the sibling by path is the whole of what keeps them working.

    """
    loaded = _by_path()

    assert not loaded.__package__
    assert loaded.header_text() == HEADER_TEXT.lstrip("\n")
    assert loaded.HEADER_NAME == kokkos_staging.HEADER_NAME
    assert loaded.ROLES == kokkos_staging.ROLES


def test_kokkos_staging_header_module_is_loaded_once():
    """The sibling is found once and remembered, on either path."""
    loaded = _by_path()

    # pylint: disable=protected-access
    assert loaded._HEADER_MODULE is None
    loaded.header_text()
    first = loaded._HEADER_MODULE
    assert first is not None
    loaded.header_text()
    assert loaded._HEADER_MODULE is first
    assert kokkos_staging._header_module() is not first
