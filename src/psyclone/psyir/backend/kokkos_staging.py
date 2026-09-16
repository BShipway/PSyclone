# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

"""Staging: how a region's array arguments reach the device it runs on.

A generated region is handed raw pointers by the Fortran that calls it, and
until now it wrapped each of them in an unmanaged ``Kokkos::View`` and ran.
That is correct exactly when the storage behind the pointer is reachable from
the execution space -- true on a CPU build, true on a GPU only for storage
allocated in a space the device can read. LFRic field data and an LMA
operator's local stencil are allocated in ``SharedSpace``; a dofmap, a basis
table, a quadrature weight, a stencil or colour map and a columnwise
operator's banded matrix are not.

This module holds the text of the header the generated regions use to close
that gap, ``lfric_kokkos_staging.hpp``, and the spellings the writer emits to
call into it. The header reads ``LFRIC_KOKKOS_STAGING`` once per process and
offers three modes:

``none``
    An unmanaged View over the caller's pointer -- what a region did before
    this existed, and the default when the variable is unset, so no build and
    no gate moves unless it is asked to.

``all``
    Every array is allocated in the execution space's memory space and copied
    in on entry; a written one is copied back after the fence. Correct on any
    device whatever space the caller allocated in, and slow enough that
    timings taken in it say nothing about the prototype.

``non-field``
    A ``field`` view is left unmanaged over the caller's pointer, because
    LFRic claims field data and an LMA operator's local stencil from
    ``SharedSpace``. A ``readonly`` array is copied
    once and cached by ``(pointer, bytes)``, because a dofmap or a map is
    allocated once and immutable for the run. A ``readwrite`` array, and a
    ``transient`` one whose storage the PSy layer allocates and frees around
    the invoke, is staged per call: neither may be keyed by an address.
    ``LFRIC_KOKKOS_STAGING_CACHE=0`` turns the cache off and copies on every
    call.

What the header did is reported to stderr at ``Kokkos::finalize``: a line of
totals, and then a line per role carrying that role's arrays, copies, bytes
copied each way and allocations made and released. The split is the
measurement any decision about staging rests on -- a total cannot say whether
a run's copying is a dofmap copied once or a basis table copied on every
call -- and ``psy-ir-aidev``'s ``bin/measure-timestep`` reads both lines into
one row per run.

The *role* -- which of those kinds an argument is -- is decided by the
LFRic transformation that knows what the argument means, never here and never
by the writer: a C++ writer sees a ``double *`` and cannot tell a field from a
basis table.

Nothing in this module imports PSyclone at import time. That is deliberate:
the shell gates that compile the header load this file directly and print
:py:func:`header_text`, so no gate carries a pasted copy that can drift from
what the regions are generated against. The header text itself lives in the
sibling module :py:mod:`kokkos_staging_header` -- seven hundred lines of C++
in a string, which this module would otherwise be mostly made of -- and
:py:func:`header_text` reaches it by path when there is no package to import
it from, so a by-path caller is unaffected by the split.

"""

import importlib.util
import os

#: The roles a View may carry, in the order the header declares them.
ROLES = ("field", "readonly", "readwrite", "transient")

#: The name the header is written under, beside the regions that include it.
HEADER_NAME = "lfric_kokkos_staging.hpp"

#: The sibling module holding the header text, once it has been loaded.
_HEADER_MODULE = None


def _header_module():
    '''The sibling module :py:mod:`kokkos_staging_header`, loaded once.

    Reached two ways, because this module is reached two ways. Imported as
    part of PSyclone it is an ordinary sibling import. Loaded from source by
    path -- which is how the shell gates that compile the header take it, with
    no PSyclone package around it and sometimes no PSyclone installed at all --
    an import by name would fail, or worse, succeed against a *different*
    checkout's package; so the file beside this one is loaded by path instead.
    ``__package__`` is empty exactly in the second case and names the package
    in the first, which is what decides.

    :returns: the module holding ``HEADER_TEXT``.
    :rtype: module

    '''
    global _HEADER_MODULE  # pylint: disable=global-statement
    if _HEADER_MODULE is None:
        if __package__:
            # pylint: disable=import-outside-toplevel
            from psyclone.psyir.backend import kokkos_staging_header
            _HEADER_MODULE = kokkos_staging_header
        else:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "kokkos_staging_header.py")
            spec = importlib.util.spec_from_file_location(
                "kokkos_staging_header", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _HEADER_MODULE = module
    return _HEADER_MODULE


def header_text():
    '''The text of ``lfric_kokkos_staging.hpp``.

    The single source of it: the transformation that captures a model's
    regions writes this beside them, and the gates that compile it load this
    function rather than carrying a copy. The text itself is in the sibling
    module :py:mod:`kokkos_staging_header`; this is the supported way to read
    it, and stays here because four shell callers load this file by path and
    call this function.

    :returns: the header, ending in a newline.
    :rtype: str

    '''
    return _header_module().HEADER_TEXT.lstrip("\n")


def include_line():
    '''The ``#include`` a generated region carries for the header.

    :returns: the line, without its newline.
    :rtype: str

    '''
    return f'#include "{HEADER_NAME}"'


def role_of(view):
    '''The staging role of a View, falling back to what its constness says.

    A description built by the LFRic transformations names the role, because
    that is where an argument's meaning is known. A description built by hand
    -- a test, or a writer used directly -- may not, and then the role is
    read from ``read_only``: a read-only array is ``readonly`` and a written
    one is ``readwrite``. Neither of those claims the argument is in a space
    the device shares, which is the claim ``field`` makes and the only one
    that would be unsafe to assume.

    :param view: the View to place.
    :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

    :returns: one of :py:data:`ROLES`.
    :rtype: str

    :raises ValueError: if the View names a role that is not one of
        :py:data:`ROLES`.

    '''
    role = getattr(view, "role", None)
    if role is None:
        return "readonly" if view.read_only else "readwrite"
    if role not in ROLES:
        raise ValueError(
            f"View '{view.name}' carries staging role '{role}', which is not "
            f"one of {list(ROLES)}.")
    return role


# Six arguments, and each is one field of the declaration being written: the
# View's type, the name the body subscripts, the pointer the storage
# arrives in, the role, the extents and the indentation. Gathering them
# into an object would be a second description of a View beside
# KokkosView, which is the thing the caller already holds.
def stage_declaration(view_type, name, data_name, role, extents, indent="  "):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    '''The declaration of one staged View.

    :param str view_type: the C++ View type, which is what ``stage`` returns
        in every mode and therefore what the launch body is compiled against.
    :param str name: the name the region's body subscripts.
    :param str data_name: the pointer argument the storage arrives in.
    :param str role: one of :py:data:`ROLES`.
    :param extents: one C++ integer expression per dimension.
    :type extents: Iterable[str]
    :param str indent: the indentation of the first line.

    :returns: the declaration, indented, without a trailing newline.
    :rtype: str

    '''
    arguments = ", ".join(
        (data_name, f"lfric_kokkos::Role::{role}", *extents))
    return (f"{indent}auto {name} = lfric_kokkos::stage<\n"
            f"{indent}    {view_type}>(\n"
            f"{indent}    {arguments});")


def unstage_statement(name, data_name, indent="  "):
    '''The write-back of one staged View, for after the region's fence.

    :param str name: the View.
    :param str data_name: the pointer its contents belong to.
    :param str indent: the indentation.

    :returns: the statement, ending in a newline.
    :rtype: str

    '''
    return f"{indent}lfric_kokkos::unstage({name}, {data_name});\n"


def release_statement(indent="  "):
    '''The end-of-region release of whatever is still staged.

    :param str indent: the indentation.

    :returns: the statement, ending in a newline.
    :rtype: str

    '''
    return f"{indent}lfric_kokkos::release();\n"


def view_declaration(view):
    '''The declaration of one of a region's array arguments.

    The C++ type is the type it was before staging existed -- element type,
    layout, rank and memory traits -- because that is what the launch body
    is compiled against and none of it may move. What staging decides is the
    memory the View covers, and that is decided at run time inside
    :py:func:`stage`.

    :param view: the View to declare over storage the caller owns.
    :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

    :returns: the declaration, indented for the region body, without a
        trailing newline.
    :rtype: str

    '''
    const = "const " if view.read_only else ""
    rank = "*" * len(view.extents)
    traits = "ReadOnly" if view.random_access else "Unmanaged"
    return stage_declaration(
        f"Kokkos::View<{const}{view.c_type}{rank}, "
        f"Kokkos::LayoutLeft, MemorySpace, {traits}>",
        view.name, view.data_name, role_of(view), view.extents)


def staging_epilogue(region) -> str:
    """Return the statements a region runs after its fence.

    After the fence, so that every write the launch made has landed: a
    written View is copied back to the caller's storage, and then whatever
    staging still holds for this call is dropped. Both are no-ops in the
    default mode, and a region with no Views at all emits neither rather
    than a call about nothing.

    :param region: the captured region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos_region.KokkosRegion`

    :returns: the epilogue statements, each already a full line, or the
        empty string for a region that describes no View.
    """
    # Imported here rather than at the top of the module: the shell gates
    # that compile the header load this file by path, with no PSyclone
    # package around it, and only the spelling helpers are wanted there.
    # pylint: disable=import-outside-toplevel
    from psyclone.psyir.backend.kokkos_region import KokkosView
    view_arguments = [argument for argument in region.arguments
                      if isinstance(argument, KokkosView)]
    if not view_arguments:
        return ""
    return "".join(
        unstage_statement(argument.name, argument.data_name)
        for argument in view_arguments
        if not argument.read_only) + release_statement()


__all__ = ["HEADER_NAME", "ROLES", "header_text", "include_line",
           "release_statement", "role_of", "stage_declaration",
           "staging_epilogue", "unstage_statement", "view_declaration"]
