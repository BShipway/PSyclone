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

"""Say what a captured loop may write, and how a shared write is made safe.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method but one is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object. The exception is
:py:meth:`LFRicKokkosWriteMixin._validate_atomics_option`, which is called
from ``validate`` on the instance and takes ``self`` as the option checks
beside it do.

It is a module of its own because the capture contract asks two unrelated
questions of a loop, and this is the second of them: what those iterations may
*write*. Where the iterations are is ``LFRicKokkosIterationMixin``; the rest of
the contract, and the walk over the kernel body, is
``LFRicKokkosContractMixin``.

One fact decides everything here. LFRic's ``gh_inc`` and ``gh_readinc`` say
that two cells of one launch contribute to the same element of a field, and
every rule below follows from that. A plain write to a continuous space is
refused outright -- :py:meth:`LFRicKokkosWriteMixin._validate_written_space`
-- because neither answer helps a cell that *decides* an element rather than
adding to it. A shared contribution is admitted, and answered in one of two
ways: an atomic update, or a colouring that puts the neighbouring cells in
different launches. :py:meth:`LFRicKokkosWriteMixin._uses_atomics` says which
is in force, :py:meth:`LFRicKokkosWriteMixin._validate_atomics_option` refuses
a caller asking for both or for neither, and
:py:meth:`LFRicKokkosWriteMixin._validate_shared_updates` checks that the
atomic arm's statements are ones a single atomic carries out. The colouring
arm's three extra arguments -- the map, the colour and the colour count -- are
built by :py:meth:`LFRicKokkosWriteMixin._colouring`, which is here rather
than in ``LFRicKokkosArgumentMixin`` because a coloured region exists only as
the alternative answer to a shared write.

Which formals carry shared data is asked twice, of two different things. The
metadata answers it per *argument*, which is
:py:meth:`LFRicKokkosWriteMixin._shared_arguments` and is askable before any
rewrite; the stub argument walk answers it per *formal*, which is
:py:meth:`LFRicKokkosWriteMixin._shared_formals` and needs the schedule.
``_SharedArgumentPositions`` is the walk that makes the second askable without
side effects. The two name the same pair of accesses and still write that pair
out separately, once here and once on the walk; joining them is a change of
its own.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``: :py:meth:`LFRicKokkosWriteMixin._shared_formals` asks
``cls._implicit_extents``, and
:py:meth:`LFRicKokkosWriteMixin._uses_atomics`,
:py:meth:`LFRicKokkosWriteMixin._validate_atomics_option` and
:py:meth:`LFRicKokkosWriteMixin._colouring` ask ``cls._COLOURED_LOOP_TYPE``
and ``cls._CELL_COUNT``, which are ``LFRicKokkosIterationMixin``'s. Calling a
method here directly on this mixin is therefore not supported.
"""

from psyclone.core import AccessType
from psyclone.domain.lfric import KernStubArgList, LFRicConstants
from psyclone.psyGen import InvokeSchedule
from psyclone.psyir.backend.kokkos import (
    KokkosColourMap, KokkosScalar, KokkosView)
from psyclone.psyir.backend.kokkos_array_expression import (
    KokkosArrayExpression)
from psyclone.psyir.backend.kokkos_array_expression_mixin import (
    ATOMIC_UPDATES, atomic_update_operands)
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Assignment, Range, Reference)
from psyclone.psyir.symbols import SymbolTable
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


class LFRicKokkosWriteMixin:
    """What a captured loop may write, and how a shared write is made safe.

    Every question here is about a field the launch updates: whether its
    function space admits the write at all, whether more than one cell of the
    launch reaches the same element, and, where one does, which of the two
    answers -- an atomic update or a colouring -- is in force and what each
    needs built for it. The rest of the capture contract is
    ``LFRicKokkosContractMixin``; where the loop iterates is
    ``LFRicKokkosIterationMixin``; how every other argument is described is
    ``LFRicKokkosArgumentMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The accesses under which two cells contribute to one element, so that
    #: the update has to be made indivisible or serialised by colour.
    _SHARED_ACCESSES = (AccessType.INC, AccessType.READINC)

    #: The option choosing between the two answers to a write two cells of
    #: one launch share. ``True`` generates a ``Kokkos::atomic_*`` update for
    #: every read-modify-write of a shared field; ``False`` generates none
    #: and requires the loop to have been coloured first, so that the cells
    #: meeting at a dof are in different launches. Absent, the choice follows
    #: the loop: a coloured loop takes the coloured arm and every other loop
    #: takes atomics, which is what makes atomics the default and makes every
    #: capture predating this option generate the source it generated then.
    #:
    #: The two are alternatives rather than a ranking. Both are correct, and
    #: which is faster is a measurement neither this class nor the branch
    #: that added it has made.
    _ATOMICS_OPTION = "atomics"

    @classmethod
    def _uses_atomics(cls, node, options):
        """Say which of the two answers to a shared write is in force.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: the transformation options.
        :type options: Optional[Dict[str, Any]]

        :returns: whether a shared update is to be generated as an atomic.
        :rtype: bool
        """
        requested = (options or {}).get(cls._ATOMICS_OPTION)
        if requested is None:
            return node.loop_type != cls._COLOURED_LOOP_TYPE
        return bool(requested)

    def _validate_atomics_option(self, node, options):
        """Check the ``"atomics"`` option against the loop it is given with.

        Each arm answers a shared write on its own, and the two together
        answer it twice: an atomic on data colouring has already made private
        to one launch costs an instruction and buys nothing. Asking for both
        is therefore a contradiction in what the caller stated rather than a
        preference to be resolved quietly, and so is asking for neither on a
        loop that has a shared write and has not been coloured -- which would
        generate a race.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: the transformation options.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if the option is neither absent nor a
            bool; if it is ``True`` on a coloured loop; or if it is ``False``
            on an uncoloured loop whose kernel has a shared write.
        """
        requested = (options or {}).get(self._ATOMICS_OPTION)
        if requested is not None and not isinstance(requested, bool):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option must be "
                f"absent or a bool, but found '{requested}'.")
        coloured = node.loop_type == self._COLOURED_LOOP_TYPE
        if requested and coloured:
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option is True "
                "on a coloured loop. Colouring has already made every write "
                "the launch's own, so an atomic would guard data no other "
                "cell of the launch reaches; ask for one answer to a shared "
                "write or the other.")
        if (requested is False and not coloured
                and self._shared_arguments(node.kernels()[0])):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option is False "
                "on a loop that is not coloured, whose kernel writes a field "
                "two cells share. Colour the loop first, or leave the option "
                "out and take the atomic update.")

    @classmethod
    def _shared_arguments(cls, kernel):
        """Return the kernel arguments more than one cell of a launch updates.

        Read from the kernel's metadata rather than from its body, and so
        askable before any rewrite: what makes an argument shared is the
        access LFRic declares for it, ``gh_inc`` or ``gh_readinc``, and not
        the statement that carries out the update.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :returns: the shared arguments, in metadata order.
        :rtype: list[:py:class:`psyclone.lfric.LFRicKernelArgument`]
        """
        return [argument for argument in kernel.arguments.args
                if argument.access in cls._SHARED_ACCESSES]

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
    def _validate_written_space(argument, discontinuous):
        """Check one argument's function space, if it is a written field.

        A read field is passed over, as is anything that is not a field:
        only a written space decides whether cells may run in parallel.

        :param argument: the kernel argument to check.
        :type argument: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param discontinuous: the names of the discontinuous function spaces,
            as :py:class:`psyclone.domain.lfric.LFRicConstants` gives them.
        :type discontinuous: List[str]

        A field accumulated into is passed over too. ``gh_inc`` is illegal
        on a discontinuous space -- the metadata parser refuses it -- so
        every shared write there is on a continuous one by construction, and
        a rule refusing those would refuse the whole pattern. What makes such
        a write safe is not the space but the update: an atomic combines the
        two cells' contributions, and a coloured launch keeps them apart in
        time. Which of the two is in force is decided by
        :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTrans._uses_atomics`,
        and the shape of the update itself is checked by
        :py:meth:`_validate_shared_updates`. What remains refused here is a
        plain ``gh_write`` or ``gh_readwrite`` to a continuous space, which
        neither answer helps: two cells there do not contribute to a value,
        they each decide it.

        :raises TransformationError: if the argument is a field written on a
            continuous space by an access that replaces the element rather
            than contributing to it, where one cell's write could overwrite
            another's.
        """
        if argument.argument_type != "gh_field":
            return
        if argument.access == AccessType.READ:
            return
        if argument.access in LFRicKokkosWriteMixin._SHARED_ACCESSES:
            return
        space = argument.function_space.orig_name.lower()
        if space not in discontinuous:
            raise TransformationError(
                f"LFRicKokkosTrans requires a discontinuous space for "
                f"the written field '{argument.name}', but found "
                f"'{space}': one cell's contribution could overwrite "
                "another's.")

    @classmethod
    def _validate_continuous_write(cls, kernel):
        """Check every field the kernel writes for a discontinuous space.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if any written field argument is on a
            continuous space, for the reason
            :py:meth:`_validate_written_space` gives.
        """
        discontinuous = LFRicConstants().VALID_DISCONTINUOUS_NAMES
        for argument in kernel.arguments.args:
            cls._validate_written_space(argument, discontinuous)

    @classmethod
    def _validate_shared_updates(cls, kernel, schedule):
        """Check that every write to a shared field is one an atomic answers.

        Asked only when the atomic arm is in force: a coloured launch runs
        the cells that meet at a dof in different launches, so any statement
        at all is safe there and no shape is required of it.

        The rules are the writer's own, asked here so that a loop the backend
        could not express is refused rather than captured and then failed
        part-way through. Two shapes are refused. One is an update that is
        not a read-modify-write of the element by one of the operators in
        :py:data:`~psyclone.psyir.backend.\
kokkos_array_expression_mixin.ATOMIC_UPDATES`
        -- there is no indivisible instruction for an arbitrary computation.
        The other is a statement the backend lowers to a nest of its own,
        such as one holding a section or an array-valued intrinsic: what the
        atomic has to cover is then a whole loop rather than a statement.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param schedule: the kernel schedule, carrying every rewrite
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTrans.apply`
            makes, because those rewrites decide which statements survive as
            statements.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a shared field is written by a
            statement no single atomic carries out.
        """
        shared = cls._shared_formals(kernel, schedule)
        if not shared:
            return
        shapes = ", ".join(sorted(name for name, _ in ATOMIC_UPDATES.values()))
        for assignment in schedule.walk(Assignment):
            target = assignment.lhs
            if (not isinstance(target, ArrayReference)
                    or target.name not in shared):
                continue
            if cls._is_lowered(assignment):
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture '{kernel.name}': it "
                    f"updates the shared field '{target.name}' with a "
                    "whole-array expression, which no single atomic carries "
                    "out. Colour the loop instead.")
            if atomic_update_operands(assignment) is None:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture '{kernel.name}': it "
                    f"writes the shared field '{target.name}' with a "
                    "statement that is not one of the read-modify-write "
                    f"shapes an atomic answers ({shapes}). Colour the loop "
                    "instead.")

    @staticmethod
    def _is_lowered(assignment):
        """Answer whether the backend lowers this statement to a nest.

        The same question
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.assignment_node`
        asks, and spelt the same way so that the two cannot drift: a section
        anywhere in the statement, or an array-valued intrinsic on the right,
        and in neither case a constructor, which the C writer spreads over
        its destination itself.

        :param assignment: the statement to classify.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: whether the statement becomes a nest rather than a
            statement.
        :rtype: bool
        """
        return (bool(assignment.walk(Range))
                or KokkosArrayExpression.holds(assignment.rhs)) and not \
            isinstance(assignment.rhs, ArrayConstructor)

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


__all__ = ["LFRicKokkosWriteMixin"]
