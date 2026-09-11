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

"""The description a Kokkos region is generated from.

A region is described before it is written: the driving transformation builds
the frozen records below out of PSyIR, and
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` turns one of them into
a translation unit. They are separate from the writer because they are what
its callers construct and its tests assert over, while nothing here knows how
any of it is generated -- and because the grammar of an extent, which says
what may go into a description at all, belongs with the description rather
than with the code that emits it.

Nothing in this module visits PSyIR or produces C++.
"""

from dataclasses import dataclass
import re
from typing import Optional, Tuple, Union

from psyclone.psyir.backend.c_intrinsics_mixin import (
    INTEGER_INTRINSIC_ALTERNATIVES)
from psyclone.psyir.backend.kokkos_array_expression import KokkosScratch
from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.nodes import KernelSchedule, Loop


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: The calls an extent may carry, over and above arithmetic: the C++
#: spellings of an integer ``MAX`` and ``MIN``. They are read from the C
#: writer's own table rather than restated, since what may appear in an
#: extent is exactly what that writer emits for a declared bound. Nothing
#: else is admitted, and an unqualified ``max`` is not one of them: it would
#: name whatever the generated translation unit happened to have declared.
_EXTENT_CALLS = tuple(sorted(INTEGER_INTRINSIC_ALTERNATIVES.values()))


def is_identifier(value):
    """Return whether ``value`` is a C++ identifier.

    The narrowest of the three grammars here: a name the generated source
    declares or passes, as against the expressions :py:func:`is_extent` and
    :py:func:`is_offset` admit.

    :param value: the candidate name, which need not be a string.

    :returns: whether it can be written into generated C++ as a name.
    :rtype: bool
    """
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _without_extent_calls(value):
    """Return an extent with the names of :py:data:`_EXTENT_CALLS` removed.

    ``std::max`` is punctuation as far as either grammar is concerned: it is
    not a size the launch has to be able to evaluate, and it is not a token
    an identifier check should meet. Removing it leaves the parenthesised
    argument list behind, which both grammars already know what to do with.

    :param str value: the extent to strip.

    :returns: the same text with each call name replaced by a space.
    :rtype: str
    """
    for name in _EXTENT_CALLS:
        value = value.replace(name, " ")
    return value


def extent_names(value):
    """Return the identifiers an extent expression is sized from.

    An integer literal contributes nothing, so ``"(nlayers + 1)"`` gives
    ``{"nlayers"}`` and ``"4"`` gives the empty set. Callers use this to ask
    whether an extent can be evaluated where it is written, without having to
    parse the expression themselves.

    The name of a call the extent is allowed to carry is not one of them:
    ``std::max`` is written by the back-end, not evaluated by the launch, and
    a caller checking every name against the kernel's arguments would refuse
    the shape for the spelling of its own maximum.

    :param value: the candidate extent, which need not be a string.

    :returns: every C++ identifier appearing in it.
    :rtype: set[str]
    """
    if not isinstance(value, str):
        return set()
    return set(_IDENTIFIER.findall(_without_extent_calls(value)))


def _extent_tokens(value):
    """Split an extent into its operand tokens, or ``None`` if it is not one.

    The grammar :py:func:`is_extent` documents, applied once: arithmetic over
    names and integer literals, with balanced parentheses, and with the calls
    of :py:data:`_EXTENT_CALLS`. A comma is admitted inside one of those
    calls' own parentheses and nowhere else, so ``std::max(n, 1)`` is an
    extent and ``(n, 1)`` is not.

    :param value: the candidate extent, which need not be a string.

    :returns: its operand tokens, or ``None`` where it is not an extent at
        all. An empty tuple is not returned: an extent with no operands is
        not one.
    :rtype: Optional[Tuple[str, ...]]
    """
    if not isinstance(value, str) or not value.strip():
        return None
    # Checked with the call names taken out, so that the two colons of the
    # one qualified name an extent may carry do not have to be admitted
    # everywhere else.
    if not re.fullmatch(r"[A-Za-z0-9_ (),+\-*/]+",
                        _without_extent_calls(value)):
        return None
    #: One entry per open parenthesis, saying whether it opened a call, so
    #: that a comma can be admitted by what encloses it rather than by
    #: whether the text contains a call anywhere.
    opened = []
    plain = []
    index = 0
    while index < len(value):
        for name in _EXTENT_CALLS:
            if value.startswith(name + "(", index):
                opened.append(True)
                plain.append(" (")
                index += len(name) + 1
                break
        else:
            character = value[index]
            index += 1
            if character == "(":
                opened.append(False)
            elif character == ")":
                if not opened:
                    return None
                opened.pop()
            elif character == ",":
                # A separator between a call's arguments, which is where an
                # operator would be in any other extent.
                if not opened or not opened[-1]:
                    return None
                character = " "
            plain.append(character)
    if opened:
        return None
    # Split on the operators rather than searching for names, so that a
    # malformed token such as ``4nlayers`` is seen whole and refused instead
    # of reading as a literal beside an identifier.
    return tuple(token for token in re.split(r"[ ()+\-*/]+", "".join(plain))
                 if token)


def is_offset(value):
    """Return whether ``value`` may be written as an index offset.

    An offset is the declared origin of one dimension, subtracted from every
    Fortran subscript of that dimension. It is an integer where the origin is
    a constant -- ``1`` for the Fortran default -- and an integer expression
    over named sizes where it is not, as for an array declared
    ``dimension(-stencil:stencil)``. The expression grammar is
    :py:func:`is_extent`'s, and deliberately the same one: an origin and an
    extent are two readings of one declaration, and a rule that admitted a
    shape into the extent and refused it in the origin would size a View
    correctly and index it from the wrong place. Division is admitted here
    for that reason rather than for symmetry alone: an origin that rounded
    the other way would shift every subscript of the array by one.

    :param value: the candidate offset, which need not be a string.

    :returns: whether it can be written into generated C++ as an offset.
    :rtype: bool
    """
    return isinstance(value, int) or is_extent(value)


def is_extent(value):
    """Return whether ``value`` may be written as a Kokkos extent.

    An extent is an integer expression over named sizes, so a bare name is
    accepted and so are ``max_length``, ``4``, ``(nlayers + 1)`` and
    ``((stencil_size + 1) / 2)``.

    Division was refused here until a GungHo kernel declared a local with
    one, on the ground that Fortran and C++ might round an integer quotient
    differently and that a wrongly sized allocation would not announce
    itself. They do not differ: both truncate toward zero, Fortran by
    ``13.7.2`` of its standard and C++ by ``[expr.mul]`` since C++11. What
    remains true is that a reader should not have to know that, which is why
    a launch sizing scratch from a divided extent says so in the generated
    source and stops on an extent that has come out negative; see
    :py:func:`~psyclone.psyir.backend.kokkos_launch.scratch_guard`.

    The two calls of :py:data:`_EXTENT_CALLS` are accepted, so
    ``std::max((monotone_above - 1), 1)`` is an extent: a GungHo kernel
    declares a local with an integer ``MAX`` in its shape, and the C writer
    spells that with ``std::max``. They are accepted by name and by position
    -- a comma is admitted inside one of their argument lists and nowhere
    else -- rather than by admitting the comma generally, because the text is
    a ``shmem_size`` argument that nothing rewrites and every other call
    reaching one would be a name the generated unit never declared.

    What is still refused is anything that is not that: a call such as
    ``pow(nlayers, nlayers)``, or an unqualified ``max(nlayers, 1)``. A power
    reaches an extent as a call only when its exponent is not an integer
    literal: ``nlayers ** 2`` is written as ``(nlayers * nlayers)`` and is
    arithmetic this accepts. See
    :py:mod:`psyclone.psyir.backend.c_integer_power`.

    :param value: the candidate extent, which need not be a string.

    :returns: whether it can be written into generated C++ as an extent.
    :rtype: bool
    """
    tokens = _extent_tokens(value)
    if not tokens:
        return False
    return all(token.isdigit() or _IDENTIFIER.fullmatch(token)
               for token in tokens)


@dataclass(frozen=True)
class KokkosScalar:
    """A scalar on the generated C ABI."""

    name: str
    c_type: str


@dataclass(frozen=True)
class KokkosView:
    """An unmanaged Kokkos View over storage owned by the caller."""
    # pylint: disable=too-many-instance-attributes

    name: str
    data_name: str
    c_type: str
    #: One integer expression per dimension, over the region's scalar
    #: arguments and integer literals: ``nlayers``, ``4``, ``(nlayers + 1)``.
    #: They are emitted into the generated C++ verbatim.
    extents: Tuple[str, ...]
    #: The declared origin of each dimension: the value subtracted from a
    #: Fortran subscript to reach the zero-based View element it names. An
    #: integer, or an integer expression over the region's scalar arguments
    #: for an array whose origin is not a constant. ``1`` for the Fortran
    #: default, ``0`` for an array declared ``dimension(0:nlayers)``.
    index_offsets: Tuple[Union[int, str], ...] = ()
    extra_indices: Tuple[str, ...] = ()
    read_only: bool = False
    random_access: bool = False
    managed: bool = False
    #: Whether an element of this View may be updated by more than one cell
    #: of one launch, so that every read-modify-write of it is generated as a
    #: ``Kokkos::atomic_*`` call rather than as an assignment. It describes
    #: the *sharing*, not the arithmetic: which updates exist is read from the
    #: kernel body, and a plain read of an element stays a plain read. False
    #: is the answer for every argument no other cell reaches, which is every
    #: argument of every region captured before this field existed, so those
    #: regions generate the source they generated then, byte for byte.
    atomic: bool = False
    #: Whether the cells sharing an element *replace* it rather than
    #: contribute to it, so that a statement assigning to one is generated as
    #: a ``Kokkos::atomic_store`` instead of being refused. It is set only
    #: beside :py:attr:`atomic`, and the two say different things: that one
    #: says an element is shared, and this one says what the cells sharing it
    #: do to it.
    #:
    #: What the store buys is exactly one thing, and it is worth being exact
    #: about: the element is written whole, so no reader sees a value neither
    #: cell stored. It settles nothing about *which* cell wrote last. A
    #: description setting this for a field cells accumulate into would
    #: generate a store that kept one contribution and lost the rest.
    atomic_store: bool = False
    #: What kind of thing this array is, for the staging header the region
    #: obtains its Views through: ``"field"`` for field data, which LFRic
    #: allocates in a space a device shares; ``"readonly"`` for a dofmap, a
    #: stencil, colour, intergrid or reference-element array, or a module
    #: array, none of which moves or changes for the run; ``"readwrite"`` for
    #: an operator's local stencil or scratch the caller supplies; and
    #: ``"transient"`` for a basis or differential-basis table or a rule's
    #: quadrature weights, which are read-only for the call but allocated and
    #: freed around the invoke, so their address outlives nothing. It is the
    #: one question about
    #: an argument the C++ writer cannot answer -- a ``double *`` says
    #: nothing about where it was allocated -- so it is answered by the
    #: transformation that knows what the argument means and carried here.
    #:
    #: ``None`` means unstated, which a description built by hand may be.
    #: The writer then reads a conservative role from :py:attr:`read_only`;
    #: what it will not do is assume ``field``, because that is the one
    #: answer asserting something about the caller's allocation.
    role: Optional[str] = None


@dataclass(frozen=True)
class KokkosAlias:
    """A View handle standing in for a Fortran pointer that aliases an array.

    A ``real(kind=r_tran), pointer :: p(:)`` a routine aims at one whole
    array or another is not storage: it is a second name for storage
    something else owns, and every read through it is a read of the array it
    was last aimed at. A Kokkos ``View`` is the same thing -- a handle whose
    copy shares the elements of the original -- so the Fortran is generated
    by declaring one more handle and assigning the target's to it, and the
    subscripts the body writes through the pointer need no rewriting at all.

    The handle is declared in ``Kokkos::AnonymousSpace``, a memory space
    Kokkos declares assignable from and to every other, so that one handle
    holds a ``View`` whatever space its target is in. A target may be an
    argument View, in ``MemorySpace`` with the ``Unmanaged`` or ``ReadOnly``
    traits, or a scratch array, in ``ScratchSpace``, and a pointer aimed at
    one of each in the two branches of an ``if`` -- which is the shape
    LFRic's vertical-support helpers have -- is one alias rather than a
    refusal. Rank and the memory traits are taken from the first target,
    which is where the ``decltype`` this replaces took them from; the space
    is the only part of that View's type replaced, and the layout is
    ``LayoutLeft`` for every View this back-end writes.

    The element type is ``const`` where **any** target is ``const``, and not
    merely where the first one is. A ``View`` of ``T`` cannot be assigned
    from a ``View`` of ``const T``, so a pointer aimed at a kernel-local
    column first and at a read-only argument second would otherwise declare
    a handle the second branch could not assign to. Reading through such a
    handle is then all the body may do.

    Every target is described elsewhere in the region -- as an argument or as
    scratch -- and is named here only by the name that description carries.
    """

    #: The name the body knows the pointer by, which is the name the alias
    #: handle is declared under.
    name: str
    #: The arrays the body aims the pointer at, in the order the body's
    #: pointer assignments name them. The first is the one the declaration
    #: takes its rank and traits from, and any of them being read-only makes
    #: its element type ``const``; all of them have to agree in element type
    #: and rank for the generated unit to compile -- their memory spaces need
    #: not -- which is what
    #: :py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter._validate_alias`
    #: checks before it is written.
    targets: Tuple[str, ...]


@dataclass(frozen=True)
class KokkosColourMap:
    """Where a coloured launch finds the mesh cell each of its indices names.

    A launch given this runs the cells of one colour, and its own index runs
    over those rather than over the mesh: the map is what turns one into the
    other. It is the second of the two answers to a write two cells share --
    the first being
    :py:attr:`KokkosView.atomic` -- and the two are alternatives, not a
    sequence: cells of one colour meet at no dof, so the update they make is
    a plain read-modify-write and no atomic is generated for it.

    The launch is one colour's, not the whole loop's. What runs the colours
    one after another is the caller, which enters the region once per colour;
    that ordering is where the safety comes from and it is deliberately not
    inside the region, so that the generated unit stays one ``parallel_for``
    as every other capture is.
    """

    #: The rank-2 View of the map, indexed by colour and then by the launch's
    #: own index. Its first extent is the number of colours, which is the
    #: stride under ``LayoutLeft`` and so the extent that has to be exact.
    name: str
    #: The scalar naming which colour this launch runs, as the caller's
    #: one-based Fortran colour index.
    colour: str
    #: The name the launch gives its own index, which counts the cells of
    #: this colour. It is not
    #: :py:attr:`KokkosRegion.cell_index`: that one names the mesh cell, and
    #: is what the body's dofmaps are indexed by.
    index: str


@dataclass(frozen=True)
class KokkosRegion:
    """All information required to generate one Kokkos translation unit."""
    # A description carries as many fields as the thing it describes has
    # parts, and splitting them into sub-objects would only move the count.
    # pylint: disable=too-many-instance-attributes

    name: str
    schedule: KernelSchedule
    cell_count: str
    # Spelt out rather than given a module-level alias: ``autoapi`` renders
    # every module variable as a literal block and Sphinx then appends its
    # own "alias of" line unindented, which fails the ``-W`` doc build.
    arguments: Tuple[Union[KokkosScalar, KokkosView], ...]
    #: The name the launch gives its own cell index: the ``RangePolicy``
    #: lambda's parameter, or the value the team launch computes from the
    #: league rank. It defaults to ``cell``, so a region built before this
    #: field existed generates exactly the source it generated then. A caller
    #: renames it when the kernel already declares ``cell`` itself, because a
    #: lambda parameter and a body declaration share one C++ scope, so the
    #: collision is rejected by the compiler rather than silently miscompiled.
    #: The per-cell Views' ``extra_indices`` have to name it too; the writer
    #: does not rewrite them.
    cell_index: str = "cell"
    #: One ``(Fortran kind name, C type)`` pair per kind the region's body
    #: mentions, such as ``("r_solver", "float")``. The region's arguments
    #: carry their own C types, but its locals and its literals cross no
    #: interface and would otherwise be generated at the C writer's default
    #: width -- silently promoting a single-precision kernel to double.
    #: Empty means "generate as the C writer would", which is what every
    #: region built before this field existed did.
    kind_types: Tuple[Tuple[str, str], ...] = ()
    #: One :py:class:`KokkosScratch` per kernel-local automatic array. This
    #: field selects the launch shape: empty gives the ``RangePolicy`` region
    #: generated for every capture before scratch existed, byte for byte,
    #: while a non-empty tuple gives a ``TeamPolicy`` region carrying
    #: per-thread scratch. A kernel with no local arrays has no scratch to
    #: place, so it keeps the simpler launch.
    scratch: Tuple[KokkosScratch, ...] = ()
    #: The loops of the region's own schedule that are to be spread over the
    #: team. A non-empty tuple selects the hierarchical launch -- one team per
    #: cell, with each of these loops rendered as a ``TeamVectorRange``
    #: ``parallel_for`` -- and overrides the selection by :py:attr:`scratch`;
    #: an empty tuple leaves that selection in force, so a region built before
    #: this field existed generates exactly the source it generated then.
    #: Which loops may be spread is a dependence judgement made by the driving
    #: transformation, not here; the writer checks only that what it is given
    #: is a set of unnested unit-stride loops it can find in the schedule.
    parallel_loops: Tuple[Loop, ...] = ()
    #: The name of the kernel formal carrying LFRic's cell index, or ``None``
    #: for a kernel that has none. LFRic passes that index to every kernel
    #: taking an operator, which uses it arithmetically to find its own slice
    #: of the operator's local stencil. It is the one formal the region
    #: declares rather than takes: the launch already knows which cell it is
    #: on, so the value is generated from :py:attr:`cell_index` inside the
    #: functor. Taking it across the ABI instead would compile and run, and
    #: give every cell whatever the caller passed once.
    cell_position: Optional[str] = None
    #: The map from this launch's index to a mesh cell, for a region captured
    #: from an already-coloured loop; ``None`` for every other region, which
    #: launches over the mesh cells themselves and generates exactly the
    #: source it generated before colouring was accepted.
    colour_map: Optional[KokkosColourMap] = None
    #: The team size the hierarchical launch asks for. ``None`` renders
    #: ``Kokkos::AUTO`` and lets the backend choose; a positive integer
    #: renders itself, which is how a host build reaches the team-level
    #: concurrency that ``AUTO`` sizes to one member. The flat shapes ignore
    #: it: the range launch has no team, and the flat team launch takes the
    #: size the backend recommends for its own functor.
    team_size: Optional[int] = None
    #: One :py:class:`KokkosConstant` per ``parameter`` array the body reads.
    #: Declared among the body's locals and taking no place on the ABI, so a
    #: region built before this field existed generates what it did then.
    constants: Tuple[KokkosConstant, ...] = ()
    #: The name of the scalar formal holding the first cell the launch runs,
    #: or ``None`` for a launch beginning at the first cell of the mesh.
    #: An LFRic loop over the halo cells alone begins where the owned cells
    #: end, and that is the one shape a count on its own cannot express: a
    #: launch from zero would run the owned cells the loop was told to skip.
    #: It is a formal rather than anything the generated source computes,
    #: because the PSy layer already holds the value. ``None`` writes the
    #: text every shape wrote before this field existed; see
    #: :py:func:`~psyclone.psyir.backend.kokkos_launch.launch_offsets`.
    #: It is not combined with :py:attr:`colour_map`: that map's index counts
    #: the cells of one colour and this counts the mesh's, so a region naming
    #: both is refused by
    #: :py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter._validate`.
    #: Nothing generates the pair either --
    #: :py:class:`~psyclone.transformations.LFRicColourTrans` gives the
    #: coloured loop it makes a lower bound of ``start`` whatever the loop it
    #: replaced had, the halo moving into that loop's upper bound instead.
    cell_start: Optional[str] = None
    #: Whether the region's iteration space is dofs rather than cell columns.
    #: It selects the launch shape ahead of every other field, because a dof
    #: region has no cells to give a team; see
    #: :py:func:`~psyclone.psyir.backend.kokkos_launch_dof.dof_launch`. Where
    #: it is true, :py:attr:`cell_count` holds the dof count and
    #: :py:attr:`cell_index` the name of the dof index: a region has one
    #: iteration space and one count of it, whichever space that is. A dof
    #: region names no :py:attr:`colour_map` and takes no atomic update: one
    #: iteration writes one dof and no two iterations write the same one, so
    #: there is no shared write for either answer to make safe.
    dof: bool = False
    #: The pointer locals of the body that alias a whole array, each with
    #: the arrays it is aimed at. Empty for every region captured before
    #: aliasing was described, which is why it is last and defaulted: such a
    #: region declares no alias handle and generates the source it generated
    #: then, byte for byte.
    aliases: Tuple[KokkosAlias, ...] = ()


__all__ = ["KokkosAlias", "KokkosColourMap", "KokkosRegion", "KokkosScalar",
           "KokkosView", "extent_names", "is_extent", "is_identifier",
           "is_offset"]
