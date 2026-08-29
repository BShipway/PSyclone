# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture a supported LFRic loop as a Kokkos launch."""

import re
import textwrap

from psyclone.configuration import Config
from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList, LFRicConstants, LFRicLoop
from psyclone.psyGen import InvokeSchedule, Transformation
from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosView, KokkosWriter)
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Call, CodeBlock, IntrinsicCall, Loop, Range,
    Reference, Routine)
from psyclone.psyir.symbols import (
    ArgumentInterface, ArrayType, ContainerSymbol, DataSymbol,
    ImportInterface, RoutineSymbol, ScalarType, UnsupportedFortranType,
    UnresolvedType)
from psyclone.psyir.transformations import (
    ArrayAssignment2LoopsTrans, TransformationError)


class LFRicKokkosTrans(Transformation):
    """Replace one supported LFRic cell-column loop with a C ABI call.

    The transformation recognises a kernel shape rather than a named kernel:
    an uncoloured owned-cell loop over a single kernel whose arguments are
    fields and scalars, whose written fields are on discontinuous spaces, and
    whose formals and referenced module constants all map onto the
    ``int``/``float``/``double`` ABI the Kokkos backend emits. Every part of
    the
    generated region -- its name, its C signature, its Views and the
    ``bind(C)`` interface the PSy layer calls through -- is derived from that
    kernel, so a second kernel needs no change here.

    A whole-column array section such as ``a(i:j)``, which the finite-volume
    kernels use to assign a column as a unit, is accepted and lowered to an
    explicit loop by
    :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
    before the region is described. The generated region has no way to say
    ``a(i:j)``, so a section that transformation refuses -- one carrying a
    loop-carried dependency, for instance -- is refused here too, with its
    reason quoted. A section outside an assignment altogether is beyond what
    lowering can reach and is refused before the backend sees it.

    It captures all information needed by the Kokkos backend before lowering
    the LFRic loop. The LFRic loop is then lowered so that its bound setup and
    halo-dirty calls are retained, and only the resulting generic loop is
    replaced.
    """

    #: The region's iteration count, and the second extent of every per-cell
    #: array. Named by the PSy layer, not by the kernel.
    _CELL_COUNT = "ncells"

    #: The C types the Kokkos backend emits, by what LFRic says a kind
    #: actually is. Widths come from PSyclone's own precision map rather than
    #: from a list of kind names, so ``r_tran`` and ``r_bl`` are accepted or
    #: refused on the same evidence as ``r_def``, and a single-precision
    #: ``r_solver`` build is honoured rather than silently promoted.
    #:
    #: There is deliberately no ``BOOLEAN`` entry. The precision map records
    #: ``l_def: 1``, but LFRic defines ``l_def = kind(.false.)``, which
    #: measures 4 bytes; mapping it would emit ``logical(c_bool)`` against a
    #: ``logical(4)`` actual and the model build would fail. A kind this table
    #: does not name -- ``l_def``, a 16-byte ``r_quad``, an undeclared
    #: precision -- fails closed rather than being guessed at.
    _C_TYPES = {
        (ScalarType.Intrinsic.INTEGER, 4): "int",
        (ScalarType.Intrinsic.REAL, 4): "float",
        (ScalarType.Intrinsic.REAL, 8): "double",
    }

    #: Per C type, the Fortran declaration the ``bind(C)`` interface uses and
    #: the ``iso_c_binding`` kind that declaration needs imported. One table
    #: rather than two, so the interface's ``use`` line and its declarations
    #: cannot disagree.
    _FORTRAN_TYPES = {
        "int": ("integer(c_int)", "c_int"),
        "float": ("real(c_float)", "c_float"),
        "double": ("real(c_double)", "c_double"),
    }

    #: Accesses a plain ``parallel_for`` over cells can honour. ``INC``,
    #: ``READINC`` and ``REDUCTION`` all need colouring or atomics.
    _SAFE_ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.READWRITE)

    def __str__(self):
        return "Capture a supported LFRic loop as a Kokkos launch"

    def validate(self, node, options=None, **kwargs):
        """Check that ``node`` matches the capture contract.

        The check is side-effect free: it predicts what :py:meth:`apply`
        would do, including whether each array section could be lowered,
        without altering the kernel schedule.

        :param node: the loop that is to be captured as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if ``node`` is not an LFRicLoop, or if
            its bounds, its kernel's metadata, its body, its array sections,
            its formal arguments or the module constants it reads fall
            outside the contract stated in this class's description.
        """
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

        self._validate_loop(node)
        kernel = node.kernels()[0]
        self._validate_kernel_metadata(kernel)
        schedule = self._schedule(kernel)
        self._validate_body(schedule)
        self._validate_sections(schedule)
        self._validate_formals(schedule)
        self._constants(schedule)

    @staticmethod
    def _validate_loop(node):
        """Check the loop's own iteration contract."""
        if node.loop_type or node.iteration_space != "cell_column":
            raise TransformationError(
                "LFRicKokkosTrans supports only an uncoloured cell-column "
                "loop.")
        if node.upper_bound_halo_depth is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support a halo depth.")
        # LFRicLoop does not currently expose its lower-bound name.
        # pylint: disable=protected-access
        if (node._lower_bound_name != "start" or
                node.upper_bound_name != "ncells"):
            raise TransformationError(
                "LFRicKokkosTrans supports only owned-cell bounds.")
        if len(node.kernels()) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel in the loop.")

    @classmethod
    def _validate_kernel_metadata(cls, kernel):
        """Check the LFRic metadata of the kernel to be captured."""
        if kernel.qr_required or kernel.eval_shapes:
            raise TransformationError(
                "LFRicKokkosTrans does not support quadrature or evaluator "
                "data.")
        if kernel.cma_operation is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support CMA operators.")
        if kernel.is_intergrid:
            raise TransformationError(
                "LFRicKokkosTrans does not support inter-grid kernels.")

        discontinuous = LFRicConstants().VALID_DISCONTINUOUS_NAMES
        for argument in kernel.arguments.args:
            if argument.argument_type not in ("gh_field", "gh_scalar"):
                raise TransformationError(
                    "LFRicKokkosTrans supports only gh_field and gh_scalar "
                    f"arguments, found '{argument.argument_type}'.")
            if argument.access not in cls._SAFE_ACCESSES:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture the '{argument.access}' "
                    f"access of '{argument.name}': a cell-parallel launch "
                    "would need colouring or atomics.")
            if argument.argument_type != "gh_field":
                continue
            if argument.intrinsic_type != "real":
                raise TransformationError(
                    f"LFRicKokkosTrans supports only real fields, but "
                    f"'{argument.name}' is {argument.intrinsic_type}.")
            if argument.stencil:
                raise TransformationError(
                    "LFRicKokkosTrans does not support stencil accesses.")
            if argument.access == AccessType.READ:
                continue
            space = argument.function_space.orig_name.lower()
            if space not in discontinuous:
                raise TransformationError(
                    f"LFRicKokkosTrans requires a discontinuous space for "
                    f"the written field '{argument.name}', but found "
                    f"'{space}': one cell's contribution could overwrite "
                    "another's.")

    @staticmethod
    def _schedule(kernel):
        """Return the single PSyIR schedule of the kernel to be captured."""
        schedules = kernel.get_callees()
        if len(schedules) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel schedule.")
        return schedules[0]

    @staticmethod
    def _validate_body(schedule):
        """Check that nothing in the body escapes the generated region."""
        if schedule.walk(CodeBlock):
            raise TransformationError(
                "LFRicKokkosTrans cannot capture a CodeBlock.")
        for call in schedule.walk(Call):
            if isinstance(call, IntrinsicCall):
                continue
            routine = call.routine
            name = routine.symbol.name if routine else "an unnamed routine"
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture the call to '{name}': the "
                "generated region has no Fortran to call into.")

    @classmethod
    def _validate_sections(cls, schedule):
        """Check that every whole-column section can be lowered to a loop.

        The generated region has no way to say ``a(i:j)``, so a section is
        rewritten as an explicit loop before the backend sees it. This
        predicts that rewrite rather than performing it, because
        :py:meth:`validate` must leave the schedule as it found it.

        The prediction is made by lowering a **copy** of the schedule rather
        than by asking ``ArrayAssignment2LoopsTrans.validate``, whose answer is
        weaker than its own ``apply``: it skips the references a ``Call``
        encloses, and ``apply`` expands them like any other, so an unresolved
        import used as an intrinsic's argument is accepted and then refused.
        Restating that rule here would leave two copies of it to keep in step,
        so the lowering itself is the predicate and the copy is what keeps it
        side-effect free.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a section is not part of an
            assignment, and so is beyond what lowering can reach.
        :raises TransformationError: if a section's own assignment cannot be
            lowered, in which case the reason is the one PSyclone gives.
        """
        for section in schedule.walk(Range):
            if section.ancestor(Assignment) is None:
                raise TransformationError(
                    "LFRicKokkosTrans cannot capture an array section outside "
                    "an assignment: only a whole-column assignment can be "
                    "lowered to a loop the generated region can express.")
        if not schedule.walk(Range):
            return
        try:
            cls._lower_sections(schedule.copy())
        except TransformationError as err:
            raise TransformationError(
                "LFRicKokkosTrans cannot lower an array section to a "
                f"loop: {err}") from err

    @staticmethod
    def _lower_sections(schedule):
        """Replace every whole-column section with an explicit loop.

        Applied to the schedule :py:meth:`_schedule` returns, which
        :py:meth:`~psyclone.domain.lfric.LFRicKern.get_callees` caches so that
        transformations applied to a kernel persist. That is the intended
        idiom, so the lowering is done once here rather than repeated for
        every consumer of the schedule.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if an assignment holding a section
            cannot be lowered. :py:meth:`_validate_sections` predicts this by
            running this method over a copy, so reaching it from
            :py:meth:`apply` would mean that prediction had been skipped.
        """
        lowering = ArrayAssignment2LoopsTrans()
        for assignment in schedule.walk(Assignment):
            if assignment.walk(Range):
                lowering.apply(assignment)

    @classmethod
    def _validate_formals(cls, schedule):
        """Check that every kernel formal has a place on the C ABI."""
        table = schedule.symbol_table
        formals = table.argument_list
        names = {symbol.name for symbol in formals}
        if cls._CELL_COUNT in names:
            raise TransformationError(
                f"LFRicKokkosTrans adds '{cls._CELL_COUNT}' to the generated "
                "signature, but the kernel already declares it.")
        for symbol in formals:
            if cls._c_type(symbol) is None:
                raise TransformationError(
                    f"LFRicKokkosTrans supports {cls._supported_kinds()} "
                    f"argument kinds only, but '{symbol.name}' has "
                    f"'{cls._kind_name(symbol)}'.")
            for extent in cls._extents(symbol):
                if extent not in names:
                    raise TransformationError(
                        f"LFRicKokkosTrans needs the extent '{extent}' of "
                        f"'{symbol.name}' to be a kernel argument, so that "
                        "the generated View can be sized.")

    @classmethod
    def _supported_kinds(cls):
        """Name the widths on the ABI, in the order :py:attr:`_C_TYPES` has.

        Derived from the table rather than spelt out, so the two refusal
        messages that quote it cannot drift from what is actually accepted.

        :returns: a phrase such as ``4-byte integer, 4-byte real and 8-byte
            real``.
        :rtype: str
        """
        widths = [f"{width}-byte {intrinsic.name.lower()}"
                  for intrinsic, width in cls._C_TYPES]
        return " and ".join([", ".join(widths[:-1]), widths[-1]])

    @staticmethod
    def _kind_name(symbol):
        """Return the name of a scalar or array element's kind symbol."""
        datatype = symbol.datatype
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        if not isinstance(datatype, ScalarType):
            return None
        precision = datatype.precision
        if not isinstance(precision, Reference):
            return None
        return precision.symbol.name

    @classmethod
    def _c_type(cls, symbol):
        """Return the C type of a symbol, or ``None`` if it has no mapping."""
        datatype = symbol.datatype
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        if not isinstance(datatype, ScalarType):
            return None
        return cls._map_kind(datatype.intrinsic, cls._kind_name(symbol))

    @classmethod
    def _map_kind(cls, intrinsic, kind):
        """Return the C type for one LFRic kind name, or ``None``.

        :param intrinsic: the Fortran intrinsic type the kind qualifies.
        :param str kind: the LFRic kind parameter, such as ``r_tran``.
        """
        if kind is None:
            return None
        precision = Config.get().api_conf("lfric").precision_map
        return cls._C_TYPES.get((intrinsic, precision.get(kind)))

    @staticmethod
    def _extents(symbol):
        """Return the declared extents of an array formal, in order.

        :returns: one name per dimension, empty for a scalar.
        :rtype: tuple[str]
        """
        datatype = symbol.datatype
        if not isinstance(datatype, ArrayType):
            return ()
        extents = []
        for dimension in datatype.shape:
            upper = getattr(dimension, "upper", None)
            if not isinstance(upper, Reference) or upper.children:
                raise TransformationError(
                    f"LFRicKokkosTrans requires '{symbol.name}' to be "
                    "declared with simple named extents.")
            extents.append(upper.symbol.name)
        return tuple(extents)

    @classmethod
    def _constants(cls, schedule):
        """Return the module constants the kernel body reads.

        These are not kernel arguments, so the PSy layer has to import each
        one and pass it by value into the region.

        :returns: ``(name, container, c_type)`` per constant, name-ordered.
        :rtype: list[tuple[str, str, str]]
        """
        table = schedule.symbol_table
        local = {symbol.name for symbol in table.argument_list}
        local |= {symbol.name for symbol in table.automatic_datasymbols}
        constants = {}
        for reference in schedule.walk(Reference):
            symbol = reference.symbol
            if symbol.name in local or isinstance(symbol, RoutineSymbol):
                continue
            if symbol.name in constants:
                continue
            constants[symbol.name] = cls._describe_constant(symbol)
        return [constants[name] for name in sorted(constants)]

    @classmethod
    def _describe_constant(cls, symbol):
        """Resolve one non-local symbol onto the generated C ABI."""
        if not isinstance(symbol.interface, ImportInterface):
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}': it is "
                "neither a kernel argument nor imported from a module.")
        container = symbol.interface.container_symbol.name
        try:
            symbol.resolve_type()
        except Exception as err:                 # pylint: disable=W0703
            # The kind is only stated in the module, so guessing here would
            # put a silently wrong type on the C ABI. Say what is missing
            # instead: the caller decides what PSyclone may read.
            raise TransformationError(
                f"LFRicKokkosTrans cannot type '{symbol.name}' without the "
                f"source of '{container}'. Add its directory to PSyclone's "
                f"module search path. ({err})") from err
        c_type = cls._c_type(symbol)
        if c_type is None:
            c_type = cls._declared_c_type(symbol)
        if c_type is None:
            raise TransformationError(
                f"LFRicKokkosTrans cannot pass '{symbol.name}' from "
                f"'{container}' by value: only {cls._supported_kinds()} "
                "scalars have a place on the generated C ABI.")
        return (symbol.name, container, c_type)

    @classmethod
    def _declared_c_type(cls, symbol):
        """Recover a C type from a declaration PSyIR could not model.

        LFRic module constants routinely carry attributes -- ``PROTECTED``,
        an initialiser -- that leave the frontend with an
        :py:class:`UnsupportedFortranType` holding the original text. The kind
        is still stated there, so read it rather than give up; anything with a
        shape is refused, because only scalars are passed by value.
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, UnsupportedFortranType):
            return None
        attributes = datatype.declaration.split("::")[0]
        if "DIMENSION" in attributes.upper():
            return None
        match = re.match(
            r"\s*(REAL|INTEGER)\s*\(\s*KIND\s*=\s*(\w+)\s*\)",
            attributes, re.IGNORECASE)
        if not match:
            return None
        intrinsic = {
            "real": ScalarType.Intrinsic.REAL,
            "integer": ScalarType.Intrinsic.INTEGER,
        }[match.group(1).lower()]
        return cls._map_kind(intrinsic, match.group(2).lower())

    def apply(self, node, options=None, **kwargs):
        """Generate C++ and replace ``node`` with the typed launch call.

        Unlike :py:meth:`validate`, this alters the kernel schedule: any
        array section it holds is lowered to an explicit loop before the
        region is described.

        :param node: the loop to capture as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
        :type options: Optional[Dict[str, Any]]

        :returns: the generated Kokkos C++ translation unit.
        :rtype: str

        :raises TransformationError: if ``node`` fails :py:meth:`validate`,
            if the PSy layer supplies a different number of actual arguments
            than the kernel has formals, or if the Kokkos backend cannot
            generate the region that validation predicted it could.
        """
        self.validate(node, options=options, **kwargs)
        kernel = node.kernels()[0]
        schedule = self._schedule(kernel)
        self._lower_sections(schedule)

        # KernCallArgList creates references to PSy-layer symbols. Ensure the
        # LFRic invoke has first specialised those symbols as DataSymbols.
        node.ancestor(InvokeSchedule).invoke.setup_psy_layer_symbols()
        argument_builder = KernCallArgList(kernel)
        argument_builder.generate()
        formals = schedule.symbol_table.argument_list
        actuals = [argument.copy()
                   for argument in argument_builder.psyir_arglist]
        if len(actuals) != len(formals):
            raise TransformationError(
                f"LFRicKokkosTrans expected {len(formals)} actual arguments "
                f"for '{kernel.name}' but the PSy layer supplies "
                f"{len(actuals)}.")

        # An actual that indexes into PSy-layer storage -- a dofmap sliced as
        # map(:,cell) -- is passed whole instead, and the region takes the
        # cell index itself. Everything else is already a plain reference.
        per_cell = set()
        for index, (formal, actual) in enumerate(zip(formals, actuals)):
            if not isinstance(actual, ArrayReference):
                continue
            per_cell.add(formal.name)
            actuals[index] = Reference(actual.symbol)

        constants = self._constants(schedule)
        region = KokkosRegion(
            name=self._region_name(kernel),
            schedule=schedule,
            cell_count=self._CELL_COUNT,
            arguments=self._region_arguments(schedule, per_cell, constants))
        try:
            cpp = KokkosWriter()(region)
        except (VisitorError, ValueError, TypeError) as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot express '{kernel.name}' in the "
                f"Kokkos backend: {err}") from err

        lowered_loop = node.lower_to_language_level()
        cell_count = lowered_loop.stop_expr.copy()
        routine = lowered_loop.ancestor(Routine)
        symbol_table = routine.symbol_table
        launch = self._launch_symbol(symbol_table, region)
        actuals.append(cell_count)
        actuals.extend(
            Reference(self._import_constant(symbol_table, name, container))
            for name, container, _ in constants)
        counter = lowered_loop.variable
        lowered_loop.replace_with(Call.create(launch, actuals))
        self._drop_unused_counter(routine, counter)
        return cpp

    @staticmethod
    def _drop_unused_counter(routine, symbol):
        """Undeclare the replaced loop's counter if nothing else counts by it.

        The PSy layer declares one counter per iteration space, so an invoke
        whose only cell loop is captured is left declaring a variable it
        never mentions again -- which LFRic compiles with
        ``-Werror=unused-variable``. An invoke with a second cell loop keeps
        it, which is why this asks rather than assumes.

        :param routine: the PSy-layer routine holding the replaced loop.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the counter the replaced loop was counting with.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        table = routine.symbol_table
        if table.lookup(symbol.name, otherwise=None) is not symbol:
            return
        if symbol in table.argument_list:
            return
        if any(reference.symbol is symbol
               for reference in routine.walk(Reference)):
            return
        # A Loop holds its control variable as an attribute rather than as a
        # Reference, so the walk above would not see a sibling loop still
        # counting with it.
        if any(loop.variable is symbol for loop in routine.walk(Loop)):
            return
        table.remove(symbol)

    @staticmethod
    def _region_name(kernel):
        """Name the generated region after the kernel it captures."""
        name = kernel.name.lower()
        if name.endswith("_code"):
            name = name[:-len("_code")]
        return f"{name}_kokkos"

    @classmethod
    def _region_arguments(cls, schedule, per_cell, constants):
        """Describe the generated signature for the backend.

        :param schedule: the kernel schedule being captured.
        :param set[str] per_cell: formals the PSy layer slices by cell.
        :param constants: the module constants passed by value.

        :returns: one description per generated C argument, in call order.
        :rtype: tuple
        """
        arguments = []
        for symbol in schedule.symbol_table.argument_list:
            c_type = cls._c_type(symbol)
            extents = cls._extents(symbol)
            if not extents:
                arguments.append(KokkosScalar(symbol.name, c_type))
                continue
            sliced = symbol.name in per_cell
            read_only = (
                symbol.interface.access == ArgumentInterface.Access.READ)
            arguments.append(KokkosView(
                symbol.name, f"{symbol.name}_data", c_type,
                extents + ((cls._CELL_COUNT,) if sliced else ()),
                index_offsets=(1,) * len(extents),
                extra_indices=("cell",) if sliced else (),
                read_only=read_only, random_access=read_only))
        arguments.append(KokkosScalar(cls._CELL_COUNT, "int"))
        arguments.extend(
            KokkosScalar(name, c_type) for name, _, c_type in constants)
        return tuple(arguments)

    @staticmethod
    def _import_constant(symbol_table, name, container):
        """Return the PSy-layer import for one kernel module constant."""
        existing = symbol_table.lookup(name, otherwise=None)
        if existing:
            return existing
        module = symbol_table.find_or_create(
            container, symbol_type=ContainerSymbol)
        return symbol_table.find_or_create(
            name, symbol_type=DataSymbol, datatype=UnresolvedType(),
            interface=ImportInterface(module))

    @classmethod
    def _launch_symbol(cls, symbol_table, region):
        """Create or return the explicit interoperable launch interface."""
        existing = symbol_table.lookup(region.name, otherwise=None)
        if existing:
            return existing
        symbol = RoutineSymbol(
            region.name, UnsupportedFortranType(cls._interface(region)))
        symbol_table.add(symbol)
        return symbol

    @classmethod
    def _interface(cls, region):
        """Write the ``bind(C)`` interface the PSy layer calls through."""
        names = [argument.name for argument in region.arguments]
        signature = textwrap.wrap(
            ", ".join(names), width=58, break_long_words=False)
        header = f"  subroutine {region.name}({signature[0]}"
        for line in signature[1:]:
            header += " &\n      " + line
        declarations = []
        used = {argument.c_type for argument in region.arguments}
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
        return (
            "interface\n"
            f"{header}) bind(C)\n"
            f"    use iso_c_binding, only : {kinds}\n"
            f"{body}\n"
            f"  end subroutine {region.name}\n"
            "end interface")


__all__ = ["LFRicKokkosTrans"]
