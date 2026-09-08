# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture a supported LFRic loop as a Kokkos launch."""

from psyclone.domain.lfric import LFRicLoop
from psyclone.domain.lfric.transformations.lfric_kokkos_argument_mixin \
    import LFRicKokkosArgumentMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_bounds_mixin import (
    LFRicKokkosBoundsMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_call_mixin import (
    LFRicKokkosCallMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_constants_mixin \
    import LFRicKokkosConstantsMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_contract_mixin import (
    LFRicKokkosContractMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_interface_mixin \
    import LFRicKokkosInterfaceMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_intrinsic_mixin \
    import LFRicKokkosIntrinsicMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_iteration_mixin \
    import LFRicKokkosIterationMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin import (
    LFRicKokkosInlineMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_schedule_mixin \
    import LFRicKokkosScheduleMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_types_mixin import (
    LFRicKokkosTypesMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_write_mixin import (
    LFRicKokkosWriteMixin)
from psyclone.psyGen import Transformation
from psyclone.psyir.backend.kokkos import KokkosWriter
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.transformations import TransformationError


# Twelve mixins and Transformation, which is one contract split by subject
# rather than thirteen layers of behaviour: every base but the last holds only
# private helpers, and none of them overrides anything.
# pylint: disable-next=too-many-ancestors
class LFRicKokkosTrans(LFRicKokkosContractMixin, LFRicKokkosTypesMixin,
                       LFRicKokkosArgumentMixin, LFRicKokkosBoundsMixin,
                       LFRicKokkosCallMixin,
                       LFRicKokkosConstantsMixin, LFRicKokkosInlineMixin,
                       LFRicKokkosInterfaceMixin,
                       LFRicKokkosIntrinsicMixin, LFRicKokkosIterationMixin,
                       LFRicKokkosScheduleMixin, LFRicKokkosWriteMixin,
                       Transformation):
    """Replace one supported LFRic loop with a C ABI call.

    The transformation recognises a kernel shape rather than a named kernel:
    a loop running a single kernel -- over cell columns, coloured or not, or
    over dofs -- whose arguments are fields, scalars and LMA operators, whose
    written fields are either free of a shared write or made safe by one of
    the two answers to one below, and whose formals and referenced module
    constants all map onto the ``int``/``float``/``double`` ABI the Kokkos
    backend emits. Every part of the generated region -- its name, its C
    signature, its Views and the ``bind(C)`` interface the PSy layer calls
    through -- is derived from that kernel, so a second kernel needs no change
    here. A field written from a dof loop, or from a cell-column loop on a
    discontinuous space, is free of one to begin with.

    The contract this recognises -- what it accepts, what it refuses and why
    -- is stated in full in the LFRic user guide, beside this entry: see
    :ref:`the LFRicKokkosTrans capture contract
    <lfric-kokkos-capture-contract>`. It is published text and is kept there
    rather than here so that a reader of the guide meets it where the guide
    describes this transformation.
    """

    #: The option naming the team size the hierarchical launch asks for.
    #: Absent, the launch writes ``Kokkos::AUTO`` and lets the backend size
    #: the team; on the OpenMP backend that is one member, so a host build
    #: reaches the team-level concurrency only by setting this.
    _TEAM_SIZE_OPTION = "team_size"

    def __str__(self):
        return "Capture a supported LFRic loop as a Kokkos launch"

    def validate(self, node, options=None, **kwargs):
        """Check that ``node`` matches the capture contract.

        The check is side-effect free: it predicts what :py:meth:`apply`
        would do, including whether each array section could be lowered,
        without altering the kernel schedule.

        :param node: the loop that is to be captured as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations. The
            two read here are ``"team_size"`` and ``"atomics"``; see
            :py:meth:`apply`.
        :type options: Optional[Dict[str, Any]]
        :param kwargs: additional keyword arguments for the base
            :py:meth:`~psyclone.psyGen.Transformation.validate`.
        :type kwargs: unwrapped dict

        :raises TransformationError: if ``node`` is not an LFRicLoop, or if
            its bounds, its kernel's metadata, its body, its array sections,
            its shape enquiries, its formal arguments, its local arrays or the
            module constants it reads fall outside the contract stated in this
            class's description.
        :raises TransformationError: if the ``"team_size"`` option is neither
            absent nor a positive integer.
        :raises TransformationError: if the ``"atomics"`` option and the
            loop's colouring contradict each other, as
            ``LFRicKokkosWriteMixin._validate_atomics_option`` states.
        """
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

        self._validate_atomics_option(node, options)
        team_size = (options or {}).get(self._TEAM_SIZE_OPTION)
        if team_size is not None and (
                isinstance(team_size, bool) or not isinstance(team_size, int)
                or team_size <= 0):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._TEAM_SIZE_OPTION}' option must be "
                f"a positive integer, but found '{team_size}'.")

        self._validate_loop(node)
        kernel = node.kernels()[0]
        self._validate_kernel_metadata(kernel)
        # Every rule below is asked of the body inlining leaves rather than
        # the one the kernel file holds: a callee brings its own loops,
        # locals, sections and constants in with it, and a rule asked before
        # the rewrite would be answering about a body that never reaches the
        # backend. The rewrite is made over a copy of the whole file, because
        # validate() must leave the schedule as it found it and because a
        # detached schedule has no Container for the callee to be found in.
        schedule = self._inlined_copy(self._schedule(kernel))
        self._validate_body(schedule)
        self._validate_sections(schedule)
        # The formals are judged with every assumed shape already measured,
        # because that is the shape apply() describes; on the copy taken
        # above, and so not on the kernel.
        self._resolve_assumed_shapes(schedule)
        self._validate_bounds(schedule)
        self._validate_formals(schedule)
        # Which names the launch reserves depends on which launch is selected,
        # and that is decided by the loops apply() will spread over the team.
        # Those are asked of a copy carrying the rewrites apply() makes,
        # because the lowering is itself a producer of loops: a kernel whose
        # only parallelisable loop is the one a section lowers to has none at
        # all until the copy is lowered. The two predicates above have already
        # shown that both rewrites succeed on this schedule. The copy is taken
        # the way _inlined_copy takes its own, and for the same reason: a
        # Routine copied alone loses the module scope its names resolve in.
        probe = self._rooted_copy(schedule)
        self._lower_allocations(probe)
        self._lower_sections(probe)
        self._substitute_bounds(probe)
        # The locals are judged on the probe rather than on the schedule,
        # because the allocation tier is what gives an allocated local its
        # shape: on the schedule it still has the deferred one the
        # declaration carried.
        parallel_loops = self._parallel_loops(probe)
        self._validate_locals(probe, parallel_loops)
        # Asked after the locals so that a kernel-local array a *cell* launch
        # could not place is still refused for the reason it always was; this
        # rule speaks only of what a dof launch has nowhere to put.
        self._validate_dof_body(node, probe, parallel_loops)
        self._validate_intrinsics(probe)
        # On the probe, and after the lowering, because the shape of an
        # update is what decides whether an atomic can carry it out and the
        # lowering is what settles that shape.
        if self._uses_atomics(node, options):
            self._validate_shared_updates(kernel, probe)
        self._constants(schedule)
        # The file-scope constants are described here as well as in apply(),
        # so that an array parameter the generated unit could not declare is
        # a refusal rather than a failure part-way through the capture.
        self._constant_arrays(schedule)

    def apply(self, node, options=None, **kwargs):
        """Generate C++ and replace ``node`` with the typed launch call.

        Unlike :py:meth:`validate`, this alters the kernel schedule: any
        array section it holds is lowered to an explicit loop, every shape
        enquiry is replaced by the bound its declaration gives, and every
        module-level ``parameter`` it reads is replaced by its value, before
        the region is described. The loops to spread over the team are chosen
        after all three, because the first two create loops.

        :param node: the loop to capture as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
            ``"team_size"`` sets the team the hierarchical launch asks for, as
            a positive integer; absent, the launch writes ``Kokkos::AUTO``.
        :type options: Optional[Dict[str, Any]]
        :param kwargs: additional keyword arguments for the base
            :py:meth:`~psyclone.psyGen.Transformation.apply`.
        :type kwargs: unwrapped dict

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
        self._inline_calls(schedule)
        self._lower_allocations(schedule)
        self._lower_sections(schedule)
        self._substitute_bounds(schedule)
        self._substitute_constants(schedule)
        region, actuals, constants = self._region(
            kernel, node, schedule, options)
        try:
            cpp = KokkosWriter()(region)
        except (VisitorError, ValueError, TypeError) as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot express '{kernel.name}' in the "
                f"Kokkos backend: {err}") from err
        self._call_region(node, region, actuals, constants)
        return cpp


__all__ = ["LFRicKokkosTrans"]
