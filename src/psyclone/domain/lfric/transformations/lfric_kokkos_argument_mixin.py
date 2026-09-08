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
  by conversion; see :py:meth:`LFRicKokkosArgumentMixin._as_c_bool`.
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
module state the body reads, and after them the ``bind(C)`` interface the PSy
layer calls through, whose kind assertions are the one place the generated
Fortran and the generated C++ meet.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin``, ``LFRicKokkosBoundsMixin``,
``LFRicKokkosCallMixin``, ``LFRicKokkosConstantsMixin`` and
``LFRicKokkosContractMixin`` does so through ``cls``, resolved on
``LFRicKokkosTrans``. Calling a method here directly on this mixin is
therefore not supported, and most of them do reach across:
:py:meth:`LFRicKokkosArgumentMixin._region_arguments` asks ``cls._c_type``,
``cls._extents`` and ``cls._origins``, :py:meth:`\
LFRicKokkosArgumentMixin._kind_assertions` reads ``cls._C_TYPES``,
``cls._KIND_PROBES`` and ``cls._DEFAULT_KINDS``, and :py:meth:`\
LFRicKokkosArgumentMixin._region` calls ``cls._cell_position``,
``cls._constants``, ``cls._constant_arrays`` and
``cls._implicit_extent_actuals``, and :py:meth:`\
LFRicKokkosArgumentMixin._scratch_arrays` calls ``cls._local_arrays``.
"""

import re
import textwrap
from dataclasses import replace

from psyclone.domain.lfric import KernCallArgList
from psyclone.lfric import LFRicHaloExchange
from psyclone.psyGen import InvokeSchedule
from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosView)
from psyclone.psyir.nodes import (
    ArrayReference, Call, IntrinsicCall, Literal, Reference, Routine)
from psyclone.psyir.symbols import (
    ArgumentInterface, RoutineSymbol, ScalarType, UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosArgumentMixin:
    """Build the region's argument list and the actuals the PSy layer passes.

    Every question here is about one argument: what the generated signature
    declares for it, what the ``bind(C)`` interface says it is in Fortran, and
    what expression the PSy layer supplies in its place. What a symbol is in C
    terms is ``LFRicKokkosTypesMixin``; the scratch the region reserves, the
    counter the PSy layer drops and the imports it gains are
    ``LFRicKokkosCallMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The region's iteration count, and the second extent of every per-cell
    #: array. Named by the PSy layer, not by the kernel.
    _CELL_COUNT = "ncells"

    #: Per C type, the Fortran declaration the ``bind(C)`` interface uses and
    #: the ``iso_c_binding`` kind that declaration needs imported. One table
    #: rather than two, so the interface's ``use`` line and its declarations
    #: cannot disagree.
    #: ``bool`` is first so that the ``use iso_c_binding`` line an interface
    #: writes stays in this table's order whichever types it carries.
    _FORTRAN_TYPES = {
        "bool": ("logical(c_bool)", "c_bool"),
        "int": ("integer(c_int)", "c_int"),
        "float": ("real(c_float)", "c_float"),
        "double": ("real(c_double)", "c_double"),
    }

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
    def _region_arguments(cls, formals, per_cell, cell_index, renames):
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

        :returns: one description per generated C argument, in call order, up
            to and including the cell count.
        :rtype: tuple[Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...]
        """
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
                extents + ((cls._CELL_COUNT,) if sliced else ()),
                index_offsets=cls._rename_extents(
                    cls._origins(symbol), renames),
                extra_indices=(cell_index,) if sliced else (),
                read_only=read_only, random_access=read_only))
        for renamed in renames.values():
            arguments.append(KokkosScalar(renamed, "int"))
        arguments.append(KokkosScalar(cls._CELL_COUNT, "int"))
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
            already carry every measured assumed shape and then the storage
            extent of every per-cell size, appended after the kernel's own,
            because :py:meth:`_region_arguments` puts those scalars in the
            same place and in the same order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`,
            list[:py:class:`psyclone.psyir.nodes.DataNode`],
            list[tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]]]

        :raises TransformationError: if the PSy layer supplies a different
            number of actual arguments than the kernel has formals.
        """
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
        cell_index = schedule.symbol_table.next_available_name(
            KokkosRegion.cell_index)
        region = KokkosRegion(
            name=cls._region_name(schedule),
            schedule=schedule,
            cell_count=cls._CELL_COUNT,
            cell_index=cell_index,
            cell_position=cell_position,
            arguments=(cls._region_arguments(
                formals, per_cell, cell_index, renames)
                + cls._constant_arguments(constants)),
            constants=cls._constant_arrays(schedule),
            kind_types=cls._kind_types(schedule),
            scratch=cls._scratch_arrays(schedule, renames),
            parallel_loops=cls._parallel_loops(schedule),
            team_size=(options or {}).get(cls._TEAM_SIZE_OPTION))
        actuals.extend(actual.copy() for _, actual in storage.values())
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
        because the last two of them are the PSy layer's own rather than the
        kernel's: the cell count is the bound of the loop being replaced, and
        each module constant is an import this routine gains.

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
        actuals.extend(
            Reference(cls._import_constant(
                symbol_table, name, container, orig_name))
            for name, container, orig_name, _, _ in constants)
        # region.arguments is the formals, then the storage extent of each
        # per-cell size, then the cell count, then the constants -- which is
        # exactly the order 'actuals' is in once _region's own extension and
        # both appends above have run. The two are index-aligned, and one
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

    @classmethod
    def _as_c_bool(cls, actual, symbol_table):
        """Wrap one actual argument in a conversion to ``logical(c_bool)``.

        This is what puts a Fortran ``logical`` on the C ABI without either
        side knowing the other's width. The dummy is ``logical(c_bool),
        value``; the actual is whatever kind LFRic declared, typically
        ``l_def``; and ``LOGICAL(x, c_bool)`` is a standard conversion the
        compiler performs, not a reinterpretation of storage. Neither side
        consults the precision map, which is why PSyclone issue #1941 --
        recording ``l_def`` as 1 byte where it is 4 -- cannot affect the
        result.

        :param actual: the argument expression to convert, already detached
            from the tree or freshly built.
        :type actual: :py:class:`psyclone.psyir.nodes.DataNode`
        :param symbol_table: the PSy-layer routine's table, which gains the
            ``c_bool`` import if it does not already carry one.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :returns: the conversion, ready to stand in the actual's place.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        c_bool = cls._import_constant(symbol_table, "c_bool", "iso_c_binding")
        return IntrinsicCall.create(
            IntrinsicCall.Intrinsic.LOGICAL,
            [actual, ("kind", Reference(c_bool))])

    @classmethod
    def _kind_types(cls, schedule):
        """Return the C type of every kind the captured body names.

        The region's arguments carry their own C types, but its locals and
        its literals cross no interface: nothing outside the generated file
        constrains them, so a kind the backend cannot resolve is silently
        generated at the C writer's default width. This is what stops that.

        A kind :py:meth:`_map_kind` cannot resolve is left out rather than
        refused, because the argument checks have already refused every kind
        that reaches the ABI; what is left is a local or a literal whose width
        the C writer's own default is free to choose.

        A kind named only as a cast target -- the ``r_def`` of
        ``real(x, r_def)``, which no declaration in the body repeats -- is
        collected too. The backend resolves a cast's width through this table,
        so leaving it out would silently write ``(float)`` for a cast the
        Fortran asked to be ``double``: the one case where an unresolved kind
        changes a value rather than only a local's width. A cast naming a kind
        the precision map does not carry is still left out, and still written
        at that default, because there is no width to write instead.

        A declaration naming no kind is collected under the name
        :py:attr:`_DEFAULT_KINDS` gives it, for the same reason and with more
        force: the width it fixes is not written down anywhere at all, so
        leaving it out would be the one case where nothing -- neither this
        table nor the compiler's own argument check -- had looked at it.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: one ``(kind name, C type)`` pair per resolvable kind,
            name-ordered.
        :rtype: tuple[tuple[str, str], ...]
        """
        table = schedule.symbol_table
        kinds = {}
        for symbol in list(table.argument_list) + list(
                table.automatic_datasymbols):
            c_type = cls._c_type(symbol)
            kind = cls._kind_name(symbol)
            if kind is None and c_type is not None:
                # On the ABI with no kind named at all, since _c_type refuses
                # a width stated in place of a name. The region carries it
                # under the name _DEFAULT_KINDS gives it so that the width it
                # fixes can be asserted like any other.
                kind = cls._default_kind_name(symbol)
            if kind is not None:
                kinds[kind] = c_type
        for literal in schedule.walk(Literal):
            datatype = literal.datatype
            precision = getattr(datatype, "precision", None)
            if not isinstance(precision, Reference):
                continue
            kind = precision.symbol.name
            kinds[kind] = cls._map_kind(datatype.intrinsic, kind)
        for reference in schedule.walk(Reference):
            if not cls._kind_argument(reference):
                continue
            # The call's own datatype says which intrinsic the kind qualifies,
            # which the kind name alone does not: i_def and r_def are both
            # just names until the cast around them says integer or real.
            intrinsic = getattr(reference.parent.datatype, "intrinsic", None)
            kind = reference.symbol.name
            c_type = (cls._map_kind(intrinsic, kind)
                      if intrinsic is not None else None)
            if c_type is None:
                # Left out rather than written in as None. A declaration above
                # may already have resolved this kind, and the filter below
                # drops whatever is left None, so writing it in would lose the
                # width that declaration found.
                continue
            kinds[kind] = c_type
        return tuple(
            (kind, kinds[kind]) for kind in sorted(kinds)
            if kinds[kind] is not None)

    @classmethod
    def _kind_assertions(cls, kind_types):
        """Write the compile-time width checks for one region's kinds.

        The compiler already checks the arguments, because the interface names
        an ``iso_c_binding`` kind where the PSy layer names an LFRic one. It
        cannot check what the generated body assumed about a local or a
        literal, and it cannot check anything at all if the two kinds happen to
        agree today and stop agreeing when LFRic is rebuilt at another
        precision. These assertions close both gaps in the one place the
        generated Fortran and the generated C++ meet.

        Each is the standard Fortran static assert: ``merge`` selects the kind
        ``4`` when the widths match and ``-1`` when they do not, and ``-1`` is
        not a supported integer kind, so a mismatch is a hard compile error
        naming the parameter and therefore the kind.

        A kind ``cls._DEFAULT_KINDS`` named because the declaration did not is
        written without the two things a name would otherwise buy: it is left
        out of the ``use`` line, since ``constants_mod`` declares no such kind,
        and its probe is the bare literal rather than a suffixed one, since
        ``1_default_integer`` would name a kind parameter that does not exist
        where ``1`` is the very kind in question. A region whose only asserted
        kind is that one therefore imports nothing from ``constants_mod``
        rather than importing nothing by name.

        :param kind_types: one ``(kind name, C type)`` pair per kind, as
            :py:meth:`_kind_types` returns them.
        :type kind_types: tuple[tuple[str, str], ...]

        :returns: the ``use`` line and one assertion per pair, each line
            already indented for an interface body, or the empty string when
            there are no kinds to assert.
        :rtype: str
        """
        intrinsics = {c_type: intrinsic
                      for (intrinsic, _), c_type in cls._C_TYPES.items()}
        # A logical kind has no width to assert -- it crosses the ABI by
        # conversion, as LFRicKokkosTypesMixin._C_LOGICAL_TYPE explains -- so
        # it is dropped before anything is written, the `use` line included. A
        # region whose only body kind is logical therefore emits no assertion
        # block at all rather than an empty one.
        asserted = [(kind, c_type) for kind, c_type in kind_types
                    if c_type in intrinsics]
        if not asserted:
            return ""
        defaults = {name for _, name in cls._DEFAULT_KINDS.values()}
        names = ", ".join(kind for kind, _ in asserted if kind not in defaults)
        lines = [f"    use constants_mod, only : {names}"] if names else []
        for kind, c_type in asserted:
            probe = cls._KIND_PROBES[intrinsics[c_type]]
            c_kind = cls._FORTRAN_TYPES[c_type][1]
            suffix = "" if kind in defaults else f"_{kind}"
            lines.append(
                f"    integer(kind=merge(4, -1, "
                f"storage_size({probe}{suffix}) == &\n"
                f"        storage_size({probe}_{c_kind}))), parameter :: "
                f"assert_kind_{kind} = 0")
        return "\n".join(lines) + "\n"

    @classmethod
    def _interface(cls, region):
        """Write the ``bind(C)`` interface the PSy layer calls through.

        :param region: the captured region the interface declares.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :returns: a complete ``interface`` block, for the PSy layer to carry
            as an :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`.
        :rtype: str
        """
        names = [argument.name for argument in region.arguments]
        signature = textwrap.wrap(
            ", ".join(names), width=58, break_long_words=False)
        header = f"  subroutine {region.name}({signature[0]}"
        for line in signature[1:]:
            header += " &\n      " + line
        declarations = []
        # The assertions name an iso_c_binding kind too, and a body-only kind
        # can have a C type no argument carries, so both sources are counted.
        used = {argument.c_type for argument in region.arguments}
        used |= {c_type for _, c_type in region.kind_types}
        for argument in region.arguments:
            fortran = cls._FORTRAN_TYPES[argument.c_type][0]
            if isinstance(argument, KokkosScalar):
                declarations.append(f"    {fortran}, value :: {argument.name}")
            else:
                intent = "in" if argument.read_only else "inout"
                declarations.append(
                    f"    {fortran}, dimension(*), intent({intent}) :: "
                    f"{argument.name}")
        # Only the kinds this region's arguments declare, in table order, so
        # that a region using none of a kind does not import it unused.
        kinds = ", ".join(kind for c_type, (_, kind)
                          in cls._FORTRAN_TYPES.items() if c_type in used)
        body = "\n".join(declarations)
        # A `use` must precede every other specification statement, so the
        # assertions follow both of them; and they sit inside the interface
        # body so that the generated interface stays self-contained and needs
        # nothing added to the PSy layer around it.
        assertions = cls._kind_assertions(region.kind_types)
        return (
            "interface\n"
            f"{header}) bind(C)\n"
            f"    use iso_c_binding, only : {kinds}\n"
            f"{assertions}"
            f"{body}\n"
            f"  end subroutine {region.name}\n"
            "end interface")

    @classmethod
    def _launch_symbol(cls, symbol_table, region):
        """Create or return the explicit interoperable launch interface.

        :param symbol_table: the PSy-layer table the interface is added to.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param region: the captured region the interface declares.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :returns: the symbol the generated call is made through, carrying the
            ``interface`` block as an
            :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        existing = symbol_table.lookup(region.name, otherwise=None)
        if existing:
            return existing
        symbol = RoutineSymbol(
            region.name, UnsupportedFortranType(cls._interface(region)))
        symbol_table.add(symbol)
        return symbol


__all__ = ["LFRicKokkosArgumentMixin"]
