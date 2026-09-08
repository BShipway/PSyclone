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

"""Say what the region takes for each argument, and what the PSy layer passes.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

One question is asked here, once per argument
:py:meth:`~psyclone.domain.lfric.ArgOrdering.generate` emits: what does the
generated region declare in its place, and what does the PSy layer pass. The
two answers are built together because they are index-aligned, and a mistake
in that alignment is a region that compiles, links, runs and reads the wrong
data. The kinds of argument, and what each becomes:

* A **field** is a View over the PSy layer's whole data array, and the dofmap
  beside it is a per-cell one: the actual is sliced ``map_w3(:,cell)``, so the
  region takes the array whole and indexes it by the launch's own cell. Its
  element type is the kernel's own declaration of that formal, read through
  ``cls._c_type`` as every other argument's is, so a ``gh_integer`` field
  becomes a ``View<int*>`` and a ``gh_real`` one a ``View<double*>`` without
  either being written down here.
* A **scalar** is passed by value, an integer or a real by width and a logical
  by conversion; see :py:meth:`LFRicKokkosInterfaceMixin._as_c_bool`.
* An **LMA operator** is a rank-3 View whose every extent is a formal of its
  own. What it needs is the cell, which the region declares from its launch
  index rather than taking; the cell-position formal is dropped from both
  lists together.
* A **stencil** of the accepted shapes hands over a sliced dofmap, and beside
  it either a sliced size array or -- for the 1-D and region shapes -- a
  *scalar* size fed from ``x_stencil_size(cell)``. Both become per-cell Views
  like any other dofmap: see
  :py:meth:`LFRicKokkosArgumentMixin._per_cell_scalars`. The dofmap those two
  shapes hand over is declared over that per-cell size, which no View can be
  strided by, so the region also carries the dofmap's storage extent as a
  scalar of its own; see
  :py:meth:`LFRicKokkosArgumentMixin._storage_extents`.
* **Quadrature** hands over its point counts as scalars, its weights as rank-1
  Views and one basis array per function space that asks for one, shaped
  ``(dim, ndf, np_xy, np_z)``. None of them is per-cell: the actual names the
  whole array, so the region takes it whole.
* An **evaluator** hands over a basis array shaped ``(dim, ndf, ndf_target)``
  and neither weights nor point counts, the points being the nodal points of
  the target space rather than a rule's.
* An **inter-grid** kernel needs nothing built here at all. Its cell map is an
  actual the PSy layer slices by cell, ``cell_map_c(:,:,cell)``, so the
  per-cell rule below turns it into a rank-3 View like any sliced dofmap; the
  three counts beside it are scalars; and the fine mesh's dofmap arrives whole,
  ``map_f(:,:)``, which the same rule leaves whole because it is not sliced.
  The launch is over the coarse mesh because that is the mesh LFRic's own loop
  bound names, so the second mesh reaches the region as data and never as an
  iteration space.

Beside the kernel's own formals the region carries the cell count and the
module state the body reads. The ``bind(C)`` interface the PSy layer calls
through comes after them and is written elsewhere, by
``LFRicKokkosInterfaceMixin``: this module answers one question per *argument*
and that one answers one question per *region*, once every argument has been
described.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin``, ``LFRicKokkosBoundsMixin``,
``LFRicKokkosCallMixin``, ``LFRicKokkosConstantsMixin``,
``LFRicKokkosContractMixin``, ``LFRicKokkosInterfaceMixin``,
``LFRicKokkosIterationMixin`` and ``LFRicKokkosScheduleMixin`` does so through
``cls``, resolved on
``LFRicKokkosTrans``. Calling a method here directly on this mixin is
therefore not supported, and most of them do reach across:
:py:meth:`LFRicKokkosArgumentMixin._region_arguments` asks ``cls._c_type``,
``cls._extents`` and ``cls._origins``, :py:meth:`\
LFRicKokkosArgumentMixin._region` calls ``cls._cell_position``,
``cls._constants``, ``cls._constant_arrays``, ``cls._is_dof``,
``cls._kind_types``, ``cls._parallel_loops``, ``cls._count_name``,
``cls._start_name`` and ``cls._implicit_extent_actuals``, :py:meth:`\
LFRicKokkosArgumentMixin._call_region` calls ``cls._launch_symbol`` and
``cls._as_c_bool``, and :py:meth:`\
LFRicKokkosArgumentMixin._scratch_arrays` calls ``cls._local_arrays``.
"""

import re
from dataclasses import replace

from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList, KernStubArgList
from psyclone.lfric import LFRicHaloExchange
from psyclone.psyGen import InvokeSchedule
from psyclone.psyir.backend.kokkos import (
    KokkosColourMap, KokkosRegion, KokkosScalar, KokkosView)
from psyclone.psyir.nodes import (
    ArrayReference, BinaryOperation, Call, IntrinsicCall, Literal, Reference,
    Routine)
from psyclone.psyir.symbols import (
    ArgumentInterface, ScalarType, SymbolTable)
from psyclone.psyir.transformations import TransformationError


class _SharedArgumentPositions(KernStubArgList):
    """A stub argument list recording which formals cells share.

    The question it answers is which *formal* of the kernel carries an
    argument whose metadata says two cells may update one element of it --
    ``gh_inc`` and ``gh_readinc``. Nothing already in PSyclone answers it
    without side effects.
    :py:meth:`~psyclone.domain.lfric.ArgOrdering.\
metadata_index_from_actual_index`
    would, but only
    :py:class:`~psyclone.domain.lfric.KernCallArgList` records the positions
    it reads, and that builder creates PSy-layer symbols as it walks: running
    it inside :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTrans.validate`
    would leave those symbols behind in invokes whose loops validation then
    refuses.

    :py:class:`~psyclone.domain.lfric.KernStubArgList` walks the same order,
    so recording the positions here gives the same answer. It is not free of
    the same side effect, though: every
    :py:class:`~psyclone.domain.lfric.ArgOrdering` writes its scalars into
    :py:attr:`~psyclone.domain.lfric.ArgOrdering._symtab`, which is the
    Invoke's own table whenever the kernel it is given is in a Schedule
    rather than a stub. Walking a real kernel that way created ``ndf_<space>``
    ahead of :py:class:`~psyclone.lfric.LFRicFunctionSpaces`, whose
    declarations are made with ``new_symbol`` and so became ``ndf_<space>_1``,
    declared and never used -- which the LFRic build rejects under
    ``-Werror=unused-variable``. This class therefore forces a private table,
    as :py:class:`~psyclone.domain.lfric.KernelInterface` does, and the walk
    leaves the tree untouched.

    The positions are recorded by bracketing each field with the public
    :py:attr:`~psyclone.domain.lfric.ArgOrdering.num_args`, which covers a
    field vector's several entries as it covers a field's one.

    :param kernel: the kernel whose argument list is to be walked.
    :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
    """
    # Inherited verbatim from KernStubArgList, which carries the same
    # disable for the same reason: ArgOrdering declares handlers a subclass
    # is free to leave to the base, and this one adds no coverage of its own.
    # pylint: disable=abstract-method

    #: The accesses under which one cell's contribution to an element has to
    #: be combined with another's rather than replacing it.
    _SHARED = (AccessType.INC, AccessType.READINC)

    def __init__(self, kernel):
        super().__init__(kernel)
        # Everything this walk declares goes here and is dropped with it.
        self._forced_symtab = SymbolTable()
        #: The positions in the argument list that carry shared data.
        self.shared_positions = set()

    def _record(self, argument, first):
        """Record the entries one argument added, if cells share it.

        :param argument: the argument just appended.
        :type argument: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param int first: the argument count before it was appended.
        """
        if argument.access in self._SHARED:
            self.shared_positions.update(range(first, self.num_args))

    def field(self, arg, var_accesses=None):
        """Append a field and record whether cells share it.

        :param arg: the field to be added.
        :type arg: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional map in which to store the accesses.
        :type var_accesses:
            Optional[:py:class:`psyclone.core.VariablesAccessMap`]
        """
        first = self.num_args
        super().field(arg, var_accesses)
        self._record(arg, first)

    def field_vector(self, argvect, var_accesses=None):
        """Append a field vector and record whether cells share it.

        :param argvect: the field vector to be added.
        :type argvect: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional map in which to store the accesses.
        :type var_accesses:
            Optional[:py:class:`psyclone.core.VariablesAccessMap`]
        """
        first = self.num_args
        super().field_vector(argvect, var_accesses)
        self._record(argvect, first)


class LFRicKokkosArgumentMixin:
    """Build the region's argument list and the actuals the PSy layer passes.

    Every question here is about one argument: what the generated signature
    declares for it and what expression the PSy layer supplies in its place.
    What the ``bind(C)`` interface then says each one is in Fortran is
    ``LFRicKokkosInterfaceMixin``; what a symbol is in C terms is
    ``LFRicKokkosTypesMixin``; the scratch the region reserves, the counter
    the PSy layer drops and the imports it gains are
    ``LFRicKokkosCallMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    @staticmethod
    def _region_name(schedule):
        """Name the generated region after the implementation it captures.

        After the *implementation*, not after the kernel: a kind-polymorphic
        kernel has one name and several implementations, so naming the region
        ``kernel.name`` would give both members of one interface the same
        region name with different C types. Two invokes at different precisions
        in one build would then collide. For a kernel that is not polymorphic
        the schedule carries the kernel's own name, so nothing else moves.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the region's name, the implementation's with its ``_code``
            component dropped and ``_kokkos`` appended.
        :rtype: str
        """
        name = schedule.name.lower()
        if name.endswith("_code"):
            name = name[:-len("_code")]
        elif "_code_" in name:
            # LFRic names an interface's members '<kernel>_code_<kind>'.
            name = name.replace("_code_", "_", 1)
        return f"{name}_kokkos"

    @staticmethod
    def _per_cell_scalars(formals, per_cell):
        """Return the per-cell formals the kernel declares as scalars.

        A stencil of the 1-D or region shapes is the only thing that produces
        one. LFRic hands its size to the kernel as an ``integer`` dummy, fed
        from ``x_stencil_size(cell)``, because the Fortran PSy layer is inside
        the cell loop when it evaluates that; a region is inside no such loop
        and runs every cell at once, so one cell's value is the wrong size for
        the others wherever the mesh is not uniform. The formal therefore
        becomes a rank-1 View over the whole array, subscripted by the
        launch's own cell -- the change of shape a sliced dofmap already has,
        applied to a formal whose *declaration* is a scalar.

        :param formals: the kernel formals the generated signature carries,
            in call order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param set[str] per_cell: formals the PSy layer slices by cell, as
            :py:meth:`_per_cell` returns them.

        :returns: the names of those of them the kernel declares as scalars,
            in call order.
        :rtype: tuple[str, ...]
        """
        return tuple(symbol.name for symbol in formals
                     if symbol.name in per_cell and not symbol.is_array)

    @classmethod
    def _storage_extents(cls, formals, actuals, sizes, table):
        """Measure the array each per-cell size is the used length of.

        A per-cell size is the length of *this cell's* stencil, and the kernel
        declares the dofmap beside it as ``dimension(ndf, stencil_size)``.
        Fortran allows that: the dummy is shaped by the actual's value at the
        cell, and the actual ``x_stencil_dofmap(:,:,cell)`` is at least that
        long. A View cannot be declared the same way. Its extents fix its
        strides, so a stride that varied cell to cell would read another
        cell's dofmap, and the extent it needs is instead the one the PSy
        layer allocated -- ``stencil_dofmap_type`` sizes its dofmap for the
        stencil whole and reports the used length per cell separately.

        That length is not a kernel argument, so the region carries it as a
        scalar of its own, measured with ``SIZE`` on the very array the PSy
        layer passes. The dimension measured is the one whose declared extent
        *is* the size, matched exactly: an extent that is an expression over
        the size states a length derived from it rather than the length
        itself, and measuring the array would answer a different question.

        :param formals: the kernel formals the generated signature carries,
            in call order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param actuals: the actuals the PSy layer passes for them, in the same
            order and already unsliced by :py:meth:`_per_cell`.
        :type actuals: list[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param sizes: the per-cell scalar formals, as
            :py:meth:`_per_cell_scalars` returns them.
        :type sizes: tuple[str, ...]
        :param table: the kernel's own table, asked for a name no formal and
            no local has taken.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :returns: per per-cell size that some array formal is declared over,
            the name of the scalar carrying that array's storage extent and
            the expression the PSy layer passes for it, in call order.
        :rtype: dict[str, tuple[str,
            :py:class:`psyclone.psyir.nodes.IntrinsicCall`]]
        """
        measured = {}
        for name in sizes:
            for formal, actual in zip(formals, actuals):
                if not formal.is_array or not isinstance(actual, Reference):
                    continue
                declared = cls._extents(formal)
                if name not in declared:
                    continue
                measured[name] = (
                    table.next_available_name(f"{name}_max"),
                    IntrinsicCall.create(
                        IntrinsicCall.Intrinsic.SIZE,
                        [Reference(actual.symbol),
                         ("dim", Literal(str(declared.index(name) + 1),
                                         ScalarType.integer_type()))]))
                break
        return measured

    @staticmethod
    def _rename_extents(bounds, renames):
        """Rewrite one array's declared bounds around the per-cell sizes.

        Each bound is generated C, so the rewrite is over its text: a name
        appearing in it is replaced whole, leaving ``ndf_w3`` and
        ``stencil_size_w3`` alone where ``stencil_size`` is renamed. An origin
        that is an integer rather than an expression is carried through
        untouched.

        :param bounds: the origins or the extents of one array, as
            :py:meth:`LFRicKokkosBoundsMixin._origins` and
            :py:meth:`LFRicKokkosBoundsMixin._extents` give them.
        :type bounds: tuple[Union[int, str], ...]
        :param dict[str, str] renames: the new name of each per-cell size.

        :returns: the same bounds with every per-cell size renamed.
        :rtype: tuple[Union[int, str], ...]
        """
        if not renames:
            return tuple(bounds)
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(name) for name in renames) + r")\b")
        return tuple(
            pattern.sub(lambda match: renames[match.group()], bound)
            if isinstance(bound, str) else bound
            for bound in bounds)

    @classmethod
    def _shared_formals(cls, kernel, schedule):
        """Return the names of the formals two cells may both update.

        The names, and not the positions, because the caller has by then
        dropped the cell-position formal from the front of its list and
        appended the extents it measured to the back. A name survives both.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param schedule: the kernel schedule, whose argument list is in the
            order :py:meth:`~psyclone.domain.lfric.ArgOrdering.generate`
            fixes -- which is what makes the positions comparable.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the names of the kernel's own formals that carry data an
            element of which more than one cell contributes to.
        :rtype: set[str]

        :raises TransformationError: if the kernel declares a different
            number of formals than its metadata describes, so that no
            position can be trusted.
        """
        builder = _SharedArgumentPositions(kernel)
        builder.generate()
        if not builder.shared_positions:
            return set()
        implicit = cls._implicit_extents(schedule.symbol_table)
        formals = [symbol.name
                   for symbol in schedule.symbol_table.argument_list
                   if symbol.name not in implicit]
        if builder.num_args != len(formals):
            raise TransformationError(
                f"LFRicKokkosTrans cannot say which formals of "
                f"'{kernel.name}' are shared between cells: the kernel "
                f"declares {len(formals)} of them and its metadata describes "
                f"{builder.num_args}.")
        return {formals[position] for position in builder.shared_positions}

    @staticmethod
    def _colour_symbols(kernel):
        """Return the PSy-layer symbols a coloured launch is described from.

        LFRic creates each of the three where it first needs one, and the
        number of colours is the one a coloured schedule can be missing:
        nothing in the PSy layer names it until a bound or a declaration
        asks. It is not optional here, because the generated colour map is a
        View and that number is its ``LayoutLeft`` stride.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :returns: the colour map, the colour being launched, and the number
            of colours.
        :rtype: tuple[:py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]

        :raises TransformationError: if the PSy layer has no name for the
            number of colours.
        """
        table = kernel.ancestor(InvokeSchedule).symbol_table
        ncolours_name = kernel.ncolours_var
        if not ncolours_name:
            raise TransformationError(
                "LFRicKokkosTrans cannot capture a coloured loop whose "
                "invoke has no number of colours: the generated colour map "
                "is a View and that number is the extent it is addressed "
                "by.")
        return (kernel.colourmap, table.lookup_with_tag("colours_loop_idx"),
                table.lookup(ncolours_name))

    @classmethod
    def _colouring(cls, kernel, node, schedule, cell_index):
        """Describe how a coloured launch finds the mesh cell of each index.

        A colouring leaves two loops, and the one captured is the inner: the
        outer loop over colours stays as Fortran and enters the region once
        per colour, which is what makes a write two cells share safe without
        an atomic. The launch therefore counts the cells of one colour and
        the region's own cell index is no longer what it iterates over, so
        three things cross the ABI that an uncoloured region does not carry:
        LFRic's colour map, the colour this launch is on, and the number of
        colours.

        The last of those looks redundant beside the map itself and is not.
        The map is passed as bare storage and rebuilt as a View inside the
        region, and its first extent is the ``LayoutLeft`` stride: get that
        wrong and every lookup reads the wrong cell. The second extent has
        no such duty -- under ``LayoutLeft`` it takes no part in the address
        -- so the cell count is reused for it rather than a fourth argument
        added.

        Nothing is described for an uncoloured loop, so such a region
        generates exactly the source it generated before this existed.

        :param kernel: the kernel being captured, which owns LFRic's colour
            map and colour count.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param node: the loop being captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param schedule: the kernel schedule, whose names the launch index is
            generated clear of.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param str cell_index: the name the region gives the mesh cell, which
            the launch index must not also be.

        :returns: the region's colour map or ``None``, the descriptions of
            the three generated arguments, and the actuals the PSy layer
            passes for them.
        :rtype: tuple[
            Optional[
                :py:class:`psyclone.psyir.backend.kokkos.KokkosColourMap`],
            tuple[Union[
                :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
                :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...],
            list[:py:class:`psyclone.psyir.nodes.Reference`]]

        :raises TransformationError: if the PSy layer has no name for the
            number of colours, as :py:meth:`_colour_symbols` raises it.
        """
        if node.loop_type != cls._COLOURED_LOOP_TYPE:
            return None, (), []
        map_symbol, colour_symbol, ncolours_symbol = cls._colour_symbols(
            kernel)
        # Generated clear of the kernel's own names for the reason the cell
        # index is: the launch index and the map are declared in the scope
        # the kernel's locals are declared in.
        taken = {cell_index}
        map_name, colour_name, ncolours, index = (
            cls._clear_name(schedule.symbol_table, candidate, taken)
            for candidate in (map_symbol.name, colour_symbol.name,
                              ncolours_symbol.name, "cell_in_colour"))
        colours = KokkosColourMap(
            name=map_name, colour=colour_name, index=index)
        arguments = (
            # Both origins are Fortran's: the colour is one-based, and the
            # launch's zero-based index names the cell one past it. The
            # writer subtracts them where it writes the lookup, because this
            # is the one View no PSyIR reference reaches, but they are stated
            # here rather than left at nothing so the description is true.
            KokkosView(map_name, f"{map_name}_data", "int",
                       (ncolours, cls._CELL_COUNT), index_offsets=(1, 1),
                       read_only=True, random_access=True),
            KokkosScalar(colour_name, "int"),
            KokkosScalar(ncolours, "int"),
        )
        return colours, arguments, [
            Reference(map_symbol), Reference(colour_symbol),
            Reference(ncolours_symbol)]

    @staticmethod
    def _clear_name(table, candidate, taken):
        """Return a name clear of the kernel's own and of the names beside it.

        ``next_available_name`` answers for the symbol table alone, and the
        generated region declares two names the table does not hold: its cell
        index, and each name generated just before this one. A launch that
        declared its index under the cell's name would initialise the cell
        from itself, and one that gave two of the three colour arguments the
        same name would not compile.

        :param table: the kernel's symbol table.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param str candidate: the name to start from, which is LFRic's own
            where there is one.
        :param set[str] taken: the names already generated, added to here.

        :returns: a name no symbol and no earlier generated name carries.
        :rtype: str
        """
        name = table.next_available_name(candidate)
        while name in taken:
            name = table.next_available_name(f"{name}_")
        taken.add(name)
        return name

    @classmethod
    def _region_arguments(cls, formals, per_cell, cell_index, renames,
                          count, start, shared=frozenset(), colour=()):
        # Seven descriptions of one argument list, which is what describing
        # an argument list takes; grouping them into an object would only
        # move the count into its constructor. The locals are one per
        # argument for the same reason: the list is assembled in the order
        # the signature declares, and a shorter routine would be one that
        # assembled part of it somewhere else.
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        # pylint: disable=too-many-locals
        """Describe the generated signature down to the cell count.

        The formals are passed in rather than read from the schedule because
        one of them may already have been dropped: a kernel taking an LMA
        operator has a leading cell argument the region declares instead of
        taking, and :py:meth:`apply` removes it from the formals and the
        actuals together, so that the two stay index-aligned.

        The module state the region carries follows what this returns, and is
        described by :py:meth:`_constant_arguments`.

        :param formals: the formals the generated signature carries, in call
            order: the kernel's own, and then each extent
            :py:meth:`LFRicKokkosBoundsMixin._implicit_extent_actuals`
            measured, which is a scalar argument of the region as any other
            integer formal is.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param set[str] per_cell: formals the PSy layer slices by cell.
        :param str cell_index: the name the launch gives its own cell index,
            which every sliced View is indexed by. It is the region's
            :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`
            and is passed rather than assumed because the kernel may declare
            ``cell`` itself.
        :param dict[str, str] renames: the name of the scalar carrying the
            storage extent behind each per-cell size, as
            :py:meth:`_storage_extents` names them. Each becomes a scalar
            argument of its own, appended after the kernel's formals and
            before the cell count so that the order here and the order
            :py:meth:`_region` extends the actuals in are one order.
        :param str count: the formal the launch is bounded above by, which is
            also the last extent of every sliced View. It is
            ``LFRicKokkosIterationMixin._count_name``'s answer for the loop
            being captured, and is passed rather than read from
            ``LFRicKokkosIterationMixin._CELL_COUNT`` because a
            loop over dofs counts dofs, and a coloured loop counts the cells
            of one colour.
        :param start: the formal the launch begins at, or ``None`` for a
            launch beginning at zero. It is
            ``LFRicKokkosIterationMixin._start_name``'s answer,
            and is appended after the count so that the order here and the
            order :py:meth:`_call_region` completes the actuals in are one
            order. It is ``None`` for every coloured loop, which
            :py:class:`~psyclone.transformations.LFRicColourTrans` always
            begins at the first cell of its colour.
        :type start: Optional[str]
        :param shared: the names of the formals more than one cell of the
            launch may update, as :py:meth:`_shared_formals` gives them. Each
            becomes an atomic View, so that the contributions of two cells to
            one element are combined rather than one of them lost. Empty is
            the answer for every kernel whose writes are its own cell's, and
            is what makes such a region generate the source it generated
            before this argument existed.
        :type shared: Container[str]
        :param colour: the descriptions of the colour map, the colour and the
            number of colours, as :py:meth:`_colouring` builds them, or the
            empty tuple for a loop that is not coloured. They go after the
            renamed extents and before the count for the reason those go
            where they do: :py:meth:`_region` extends the actuals in this
            order and the two lists are read as one.
        :type colour: tuple[Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...]

        :returns: one description per generated C argument, in call order, up
            to and including the count and, where there is one, the first
            cell after it.
        :rtype: tuple[Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...]
        """
        # The two bounds are parameters of their own rather than one pair,
        # because each is written into the signature in a place of its own
        # and only one of them is optional.
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        arguments = []
        for symbol in formals:
            c_type = cls._c_type(symbol)
            sliced = symbol.name in per_cell
            read_only = (
                symbol.interface.access == ArgumentInterface.Access.READ)
            # A per-cell formal the kernel declares as a scalar has no shape
            # of its own, so the cell count is its whole shape.
            extents = cls._rename_extents(cls._extents(symbol), renames)
            if not extents and not sliced:
                arguments.append(KokkosScalar(symbol.name, c_type))
                continue
            arguments.append(KokkosView(
                symbol.name, f"{symbol.name}_data", c_type,
                extents + ((count,) if sliced else ()),
                index_offsets=cls._rename_extents(
                    cls._origins(symbol), renames),
                extra_indices=(cell_index,) if sliced else (),
                read_only=read_only, random_access=read_only,
                atomic=symbol.name in shared))
        for renamed in renames.values():
            arguments.append(KokkosScalar(renamed, "int"))
        arguments.extend(colour)
        arguments.append(KokkosScalar(count, "int"))
        if start is not None:
            arguments.append(KokkosScalar(start, "int"))
        return tuple(arguments)

    @classmethod
    def _constant_arguments(cls, constants):
        """Describe the module state the region carries.

        A scalar becomes an argument passed by value, and an array of literal
        extents a read-only View: a module array is state the region reads and
        never writes, so it crosses as a View exactly as a read-only formal
        does. These follow the arguments :py:meth:`_region_arguments`
        describes, the cell count last among those.

        :param constants: the module state the region carries, as
            :py:meth:`_constants` returns it.
        :type constants: list[tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]]

        :returns: one description per generated C argument, in call order.
        :rtype: tuple[Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...]
        """
        arguments = []
        for name, _, _, c_type, symbol in constants:
            extents = cls._extents(symbol)
            if not extents:
                arguments.append(KokkosScalar(name, c_type))
                continue
            arguments.append(KokkosView(
                name, f"{name}_data", c_type, extents,
                index_offsets=cls._origins(symbol),
                read_only=True, random_access=True))
        return tuple(arguments)

    @classmethod
    def _argument_lists(cls, kernel, node, schedule):
        """Return the region's formals, its actuals and its cell position.

        The three are built together because the first two are index-aligned.
        The formals are the kernel's, in the order
        :py:meth:`~psyclone.domain.lfric.ArgOrdering.generate` fixes, and the
        actuals are what
        :py:class:`~psyclone.domain.lfric.KernCallArgList` emits walking that
        same order. One entry of each may be dropped -- the cell position --
        and it has to be dropped from both, since nothing downstream carries
        the correspondence except the two lists' shared indices.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param node: the loop the kernel sits in.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param schedule: the kernel schedule, carrying every rewrite
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTrans.apply` makes.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the kernel's own formals, the actuals the PSy layer passes
            for them, and the name of the cell-position formal the region
            declares instead of taking, or ``None``. Any measured extent is
            left out of both, for
            :py:meth:`LFRicKokkosBoundsMixin._implicit_extent_actuals` to add
            back to the two together.
        :rtype: tuple[list[:py:class:`psyclone.psyir.symbols.DataSymbol`],
            list[:py:class:`psyclone.psyir.nodes.DataNode`], Optional[str]]

        :raises TransformationError: if the PSy layer supplies a different
            number of actual arguments than the kernel has formals.
        """
        # KernCallArgList creates references to PSy-layer symbols. Ensure the
        # LFRic invoke has first specialised those symbols as DataSymbols.
        node.ancestor(InvokeSchedule).invoke.setup_psy_layer_symbols()
        argument_builder = KernCallArgList(kernel)
        argument_builder.generate()
        # An extent :py:meth:`LFRicKokkosBoundsMixin._resolve_assumed_shapes`
        # measured is a formal the region declares and the kernel never wrote,
        # so the PSy layer has no actual for it and
        # ``cls._implicit_extent_actuals`` writes the one it would have
        # passed. Held back here so that the count compared is the kernel's
        # own, and so that the two lists below stay index-aligned.
        implicit = cls._implicit_extents(schedule.symbol_table)
        formals = [symbol for symbol in schedule.symbol_table.argument_list
                   if symbol.name not in implicit]
        actuals = [argument.copy()
                   for argument in argument_builder.psyir_arglist]
        if len(actuals) != len(formals):
            raise TransformationError(
                f"LFRicKokkosTrans expected {len(formals)} actual arguments "
                f"for '{kernel.name}' but the PSy layer supplies "
                f"{len(actuals)}.")

        cell_position = cls._cell_position(kernel, node, formals, actuals)
        if cell_position is not None:
            # Both lists, together. They are walked in step below -- and
            # 'formals' is what describes the generated signature -- so
            # dropping the actual alone would attribute every later actual to
            # the formal before it, and the dofmaps, being the ones detected
            # by the shape of their actual, would silently lose the cell
            # dimension that makes them per-cell.
            formals = formals[1:]
            del actuals[0]
        return formals, actuals, cell_position

    @staticmethod
    def _per_cell(formals, actuals):
        """Say which formals the PSy layer slices by cell, and unslice them.

        :param formals: the formals the generated signature carries, in call
            order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param actuals: the actuals the PSy layer passes for them, in the same
            order. A sliced one is replaced in place by a reference to the
            whole array it slices.
        :type actuals: list[:py:class:`psyclone.psyir.nodes.DataNode`]

        :returns: the names of the formals the region indexes by its own cell.
        :rtype: set[str]
        """
        # An actual that indexes into PSy-layer storage -- a dofmap sliced as
        # map(:,cell) -- is passed whole instead, and the region takes the
        # cell index itself. Everything else is already a plain reference.
        per_cell = set()
        for index, (formal, actual) in enumerate(zip(formals, actuals)):
            if not isinstance(actual, ArrayReference):
                continue
            per_cell.add(formal.name)
            actuals[index] = Reference(actual.symbol)
        return per_cell

    @classmethod
    def _region(cls, kernel, node, schedule, options):
        """Describe the region the backend generates, and its actual arguments.

        The loops to spread over the team are chosen here rather than passed
        in, because they are a property of the rewritten schedule and this is
        the only caller that needs them.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param node: the loop the kernel sits in.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param schedule: the kernel schedule, carrying every rewrite
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTrans.apply` makes.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param options: the transformation options, read here for
            ``"team_size"``.
        :type options: Optional[Dict[str, Any]]

        :returns: the region, the actuals the PSy layer passes for its
            formals, and the module state it carries. The actuals returned
            already carry every measured assumed shape, then the storage
            extent of every per-cell size, and then the colour map, colour
            and colour count of a coloured loop, appended after the kernel's
            own because :py:meth:`_region_arguments` puts those arguments in
            the same place and in the same order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`,
            list[:py:class:`psyclone.psyir.nodes.DataNode`],
            list[tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]]]

        :raises TransformationError: if the PSy layer supplies a different
            number of actual arguments than the kernel has formals.
        """
        # Each local here names one of the region's fields or one step of
        # deriving it, and the description has more parts than pylint's
        # default allows a routine to hold. Splitting it would move the
        # description of a region into more than one place, and would divide
        # the argument list from the actuals that must match it position for
        # position, which is the one property this routine exists to keep.
        # pylint: disable=too-many-locals
        formals, actuals, cell_position = cls._argument_lists(
            kernel, node, schedule)
        per_cell = cls._per_cell(formals, actuals)
        storage = cls._storage_extents(
            formals, actuals, cls._per_cell_scalars(formals, per_cell),
            schedule.symbol_table)
        renames = {name: renamed for name, (renamed, _) in storage.items()}
        # Appended to both lists at once, and before the region is described,
        # because _region_arguments reads them from the formals: a measured
        # extent is a scalar of the generated signature like any other.
        formals, measurements = cls._implicit_extent_actuals(
            formals, actuals, schedule.symbol_table)
        actuals.extend(measurements)
        constants = cls._constants(schedule)
        # The launch index shares a C++ scope with the kernel's own
        # declarations, so a kernel declaring 'cell' would collide with it.
        # Spelt from the dataclass default so the two cannot drift: a kernel
        # that has not taken the name still generates 'cell'.
        # A dof launch indexes dofs, so its index is named for one: the
        # generated source is read, and 'cell' over a dof loop would be as
        # misleading there as it would be here.
        dof = cls._is_dof(node)
        cell_index = schedule.symbol_table.next_available_name(
            cls._DOF_INDEX if dof else KokkosRegion.cell_index)
        count = cls._count_name(node)
        start = cls._start_name(node)
        # After the index, which the colour map's own lookup declares from.
        # A dof loop is never coloured -- LFRicColourTrans refuses one -- and
        # a coloured loop never names a first cell, so neither of the two
        # fields below composes with the two above.
        colours, colour_arguments, colour_actuals = cls._colouring(
            kernel, node, schedule, cell_index)
        region = KokkosRegion(
            name=cls._region_name(schedule),
            schedule=schedule,
            cell_count=count,
            cell_start=start,
            dof=dof,
            cell_index=cell_index,
            cell_position=cell_position,
            colour_map=colours,
            arguments=(cls._region_arguments(
                formals, per_cell, cell_index, renames, count, start,
                cls._shared_formals(kernel, schedule)
                if cls._uses_atomics(node, options) else frozenset(),
                colour_arguments)
                + cls._constant_arguments(constants)),
            constants=cls._constant_arrays(schedule),
            kind_types=cls._kind_types(schedule),
            scratch=cls._scratch_arrays(schedule, renames),
            parallel_loops=cls._parallel_loops(schedule),
            team_size=(options or {}).get(cls._TEAM_SIZE_OPTION))
        actuals.extend(actual.copy() for _, actual in storage.values())
        actuals.extend(colour_actuals)
        return region, actuals, constants

    @classmethod
    def _scratch_arrays(cls, schedule, renames):
        """Describe the kernel's automatic arrays, per-cell sizes renamed.

        A local sized from a stencil's size -- ``dimension(stencil_size)`` --
        is reserved on the host before the launch enters the region, so its
        extent has to be a value the launch holds. The per-cell size is not
        one: it is a View by the time the region is entered. The scalar
        carrying the array's storage extent is, and reserving that much is
        never short, so the rename that sizes the dofmap's View sizes this
        too.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param dict[str, str] renames: the new name of each per-cell size.

        :returns: one description per automatic array, in declaration order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`, ...]
        """
        return tuple(
            replace(item,
                    extents=cls._rename_extents(item.extents, renames),
                    index_offsets=cls._rename_extents(
                        item.index_offsets, renames))
            for item in cls._local_arrays(schedule))

    @classmethod
    def _call_region(cls, node, region, actuals, constants):
        """Replace ``node`` with the typed call into the generated region.

        The actuals are completed here rather than by :py:meth:`_region`,
        because the last of them are the PSy layer's own rather than the
        kernel's: the count is the bound of the loop being replaced, the
        first cell -- where the region takes one -- is that loop's lower
        bound less one, and each module constant is an import this routine
        gains.

        :param node: the loop to replace with the launch call.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param region: the region the backend has generated.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
        :param actuals: the actuals for the kernel's formals, as
            :py:meth:`_region` returns them. Extended in place.
        :type actuals: list[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param constants: the module state the region carries, as
            :py:meth:`_constants` returns it.
        :type constants: list[tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]]
        """
        cls._lower_halo_exchanges(node)
        lowered_loop = node.lower_to_language_level()
        cell_count = lowered_loop.stop_expr.copy()
        routine = lowered_loop.ancestor(Routine)
        symbol_table = routine.symbol_table
        launch = cls._launch_symbol(symbol_table, region)
        actuals.append(cell_count)
        if region.cell_start is not None:
            # The loop's Fortran lower bound counts from one and the launch
            # index from zero, so the conversion is a subtraction. It is made
            # here, in the PSy layer, rather than in the generated C++: the
            # region takes a value, as it does for the count, and nothing in
            # the generated source has to know which convention the caller
            # counts in.
            actuals.append(BinaryOperation.create(
                BinaryOperation.Operator.SUB,
                lowered_loop.start_expr.copy(),
                Literal("1", ScalarType.integer_type())))
        actuals.extend(
            Reference(cls._import_constant(
                symbol_table, name, container, orig_name))
            for name, container, orig_name, _, _ in constants)
        # region.arguments is the formals, then the storage extent of each
        # per-cell size, then a coloured loop's three colour arguments, then
        # the count, then a halo loop's first cell, then the constants --
        # which is exactly the order 'actuals' is in once _region's own
        # extensions and the appends above have run. The last two are
        # optional and independent: no loop carries both, a colouring always
        # beginning at the first cell of its colour.
        # The two are index-aligned, and one
        # loop covers a logical formal and an imported logical constant alike.
        # That alignment is what makes this correct and it is not visible from
        # the loop itself.
        for index, argument in enumerate(region.arguments):
            if argument.c_type == cls._C_LOGICAL_TYPE:
                actuals[index] = cls._as_c_bool(actuals[index], symbol_table)
        counter = lowered_loop.variable
        lowered_loop.replace_with(Call.create(launch, actuals))
        cls._drop_unused_counter(routine, counter)

    @staticmethod
    def _lower_halo_exchanges(node):
        """Lower every halo exchange in ``node``'s invoke, before ``node`` is.

        A halo exchange does not know its own depth: it computes one by
        walking forward for the accesses that read the field it exchanges, and
        those accesses are LFRic kernel arguments carrying LFRic metadata.
        :py:meth:`apply` is about to replace the loop holding them with a
        plain :py:class:`~psyclone.psyir.nodes.Call`, which carries none, so
        an exchange lowered afterwards finds no reader at all and PSyclone
        raises :py:class:`~psyclone.errors.InternalError` rather than
        generating a wrong depth.

        Lowering the exchanges first is the order whole-container lowering
        would have used anyway -- an exchange precedes the loop it feeds, and
        lowering runs in schedule order -- so this restores that order rather
        than choosing a new one. It does nothing for a schedule that has no
        exchange, which is every region captured before stencils.

        :param node: the loop about to be captured, used only to reach the
            invoke schedule containing it. A node with no
            :py:class:`~psyclone.psyGen.InvokeSchedule` ancestor, as a unit
            test's bare schedule has, is left alone.
        :type node: :py:class:`~psyclone.domain.lfric.LFRicLoop`

        """
        schedule = node.ancestor(InvokeSchedule)
        if schedule is None:
            return
        for exchange in schedule.walk(LFRicHaloExchange):
            exchange.lower_to_language_level()


__all__ = ["LFRicKokkosArgumentMixin"]
