# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture the first supported LFRic loop as a Kokkos launch."""

from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList, LFRicLoop
from psyclone.psyGen import InvokeSchedule, Transformation
from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosView, KokkosWriter)
from psyclone.psyir.nodes import (
    ArrayReference, Call, CodeBlock, Reference, Routine)
from psyclone.psyir.symbols import (
    ArrayType, ContainerSymbol, DataSymbol, ImportInterface, RoutineSymbol,
    ScalarType, UnsupportedFortranType, UnresolvedType)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosTrans(Transformation):
    """Replace one supported ``moist_dyn_gas`` loop with a C ABI call.

    This initial transformation intentionally recognises one exact kernel
    shape. It captures all information needed by the Kokkos backend before
    lowering the LFRic loop. The LFRic loop is then lowered so that its bound
    setup and halo-dirty calls are retained, and only the resulting generic
    loop is replaced.
    """

    _REGION_NAME = "moist_dyn_gas_kokkos"
    _FORMAL_NAMES = (
        "nlayers", "moist_dyn_gas", "mr_v", "ndf_wtheta",
        "undf_wtheta", "map_wtheta")

    def __str__(self):
        return "Capture a supported LFRic loop as a Kokkos launch"

    def validate(self, node, options=None, **kwargs):
        """Check that ``node`` has the exact first-prototype contract."""
        # pylint: disable=too-many-branches,too-many-locals
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

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

        kernels = node.kernels()
        if len(kernels) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel in the loop.")
        kernel = kernels[0]
        if kernel.name.lower() != "moist_dyn_gas_code":
            raise TransformationError(
                "LFRicKokkosTrans currently supports only "
                "moist_dyn_gas_code.")
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

        arguments = kernel.arguments.args
        if len(arguments) != 2 or any(
                argument.argument_type != "gh_field"
                for argument in arguments):
            raise TransformationError(
                "LFRicKokkosTrans requires exactly two field arguments.")
        expected = ((AccessType.WRITE, "wtheta"),
                    (AccessType.READ, "wtheta"))
        for argument, (access, space) in zip(arguments, expected):
            if argument.intrinsic_type != "real" or argument.access != access:
                raise TransformationError(
                    "LFRicKokkosTrans field argument metadata is unsupported.")
            if argument.function_space.orig_name.lower() != space:
                raise TransformationError(
                    "LFRicKokkosTrans requires a discontinuous Wtheta "
                    "write and read.")
            if argument.stencil:
                raise TransformationError(
                    "LFRicKokkosTrans does not support stencil accesses.")

        schedules = kernel.get_callees()
        if len(schedules) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel schedule.")
        schedule = schedules[0]
        if schedule.walk(CodeBlock):
            raise TransformationError(
                "LFRicKokkosTrans cannot capture a CodeBlock.")
        formal_names = tuple(
            symbol.name for symbol in schedule.symbol_table.argument_list)
        if formal_names != self._FORMAL_NAMES:
            raise TransformationError(
                "LFRicKokkosTrans kernel formal arguments have changed: "
                f"{formal_names}.")
        expected_kinds = (
            "i_def", "r_def", "r_def", "i_def", "i_def", "i_def")
        actual_kinds = tuple(
            self._kind_name(symbol)
            for symbol in schedule.symbol_table.argument_list)
        if actual_kinds != expected_kinds:
            raise TransformationError(
                "LFRicKokkosTrans kernel argument kinds have changed: "
                f"{actual_kinds}.")

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

    def apply(self, node, options=None, **kwargs):
        """Generate C++ and replace ``node`` with the typed launch call.

        :returns: the generated Kokkos C++ translation unit.
        :rtype: str
        """
        # pylint: disable=too-many-locals
        self.validate(node, options=options, **kwargs)
        kernel = node.kernels()[0]
        schedule = kernel.get_callees()[0]

        # KernCallArgList creates references to PSy-layer symbols. Ensure the
        # LFRic invoke has first specialised those symbols as DataSymbols.
        node.ancestor(InvokeSchedule).invoke.setup_psy_layer_symbols()
        argument_builder = KernCallArgList(kernel)
        argument_builder.generate()
        if len(argument_builder.psyir_arglist) != 6:
            raise TransformationError(
                "LFRicKokkosTrans requires the six-argument Wtheta call "
                "contract.")
        actuals = [argument.copy()
                   for argument in argument_builder.psyir_arglist]
        map_argument = actuals[5]
        if not isinstance(map_argument, ArrayReference):
            raise TransformationError(
                "LFRicKokkosTrans expected a sliced dofmap argument.")
        actuals[5] = Reference(map_argument.symbol)

        region = self._region(schedule)
        cpp = KokkosWriter()(region)

        lowered_loop = node.lower_to_language_level()
        cell_count = lowered_loop.stop_expr.copy()
        routine = lowered_loop.ancestor(Routine)
        symbol_table = routine.symbol_table
        recip_epsilon = self._import_recip_epsilon(symbol_table)
        launch = self._launch_symbol(symbol_table)
        actuals.extend([cell_count, Reference(recip_epsilon)])
        lowered_loop.replace_with(Call.create(launch, actuals))
        return cpp

    @classmethod
    def _region(cls, schedule):
        """Create the backend description for the captured schedule."""
        return KokkosRegion(
            name=cls._REGION_NAME,
            schedule=schedule,
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
                KokkosScalar("recip_epsilon", "double")))

    @staticmethod
    def _import_recip_epsilon(symbol_table):
        """Return the PSy-layer import for the kernel module constant."""
        existing = symbol_table.lookup("recip_epsilon", otherwise=None)
        if existing:
            return existing
        module = symbol_table.find_or_create(
            "planet_config_mod", symbol_type=ContainerSymbol)
        return symbol_table.find_or_create(
            "recip_epsilon", symbol_type=DataSymbol,
            datatype=UnresolvedType(), interface=ImportInterface(module))

    @classmethod
    def _launch_symbol(cls, symbol_table):
        """Create or return the explicit interoperable launch interface."""
        existing = symbol_table.lookup(cls._REGION_NAME, otherwise=None)
        if existing:
            return existing
        declaration = f"""interface
  subroutine {cls._REGION_NAME}(nlayers, moist_dyn_gas, mr_v, &
      ndf_wtheta, undf_wtheta, map_wtheta, ncells, &
      recip_epsilon) bind(C)
    use iso_c_binding, only : c_int, c_double
    integer(c_int), value :: nlayers, ndf_wtheta, undf_wtheta, ncells
    real(c_double), dimension(*), intent(inout) :: moist_dyn_gas
    real(c_double), dimension(*), intent(in) :: mr_v
    integer(c_int), dimension(*), intent(in) :: map_wtheta
    real(c_double), value :: recip_epsilon
  end subroutine {cls._REGION_NAME}
end interface"""
        symbol = RoutineSymbol(
            cls._REGION_NAME, UnsupportedFortranType(declaration))
        symbol_table.add(symbol)
        return symbol


__all__ = ["LFRicKokkosTrans"]
