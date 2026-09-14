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

"""Give an LFRic built-in the kernel schedule it never had, so that the
capture can treat it as the dof kernel it is.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``classmethod`` or ``staticmethod``.

An LFRic built-in -- ``setval_c``, ``inc_X_plus_Y``, ``real_to_real_X`` and
the rest of :py:data:`~psyclone.domain.lfric.lfric_builtins.BUILTIN_MAP`
-- is a loop over dofs whose body PSyclone generates rather than reads from
a kernel file. Every other rule of the capture asks its question of a
:py:class:`~psyclone.psyir.nodes.KernelSchedule`, and a built-in has none,
which is why the capture refused them by name until now. In the GungHo
model that refusal left the built-ins as the largest piece of host work the
device arm carries: a fifth of a timestep, every one of them a host
traversal of a field between two device launches, and the reason the
fields keep moving between host and card.

What this mixin adds is the schedule. PSyclone already knows the body of
every built-in -- :py:meth:`~psyclone.domain.lfric.lfric_builtins.\
LFRicBuiltIn.lower_to_language_level` writes it into the PSy layer as one
assignment over ``field_data(df)`` references -- and the region's argument
list is built by the same
:py:class:`~psyclone.domain.lfric.KernCallArgList` walk a coded dof kernel
uses, which already hands a dof kernel one dof of each field. So the
schedule synthesised here is the kernel file LFRic would have written for
the built-in had it written one: a routine with one scalar formal per
argument, at the kind the argument carries, whose single statement is the
built-in's own assignment with each per-dof reference replaced by the
formal it stands for. From there the dof launch, the ABI, the staging roles
and every rule of the contract apply unchanged, because nothing about the
loop that reaches them says "built-in" any more.

Two things are deliberately not built. A **reduction** (``sum_X``,
``X_innerproduct_Y``, ``inc_max_aX`` and the like) lowers to an
accumulation into a scalar, which the dof launch's flat ``parallel_for``
has no shape for; it is refused by name as before, and a reduction launch
is a capability of its own. And the schedule is **synthesised afresh** on
every call rather than cached on the built-in, because the copy it lowers
is cheap and a cached schedule would outlive any rewrite of the invoke
around it.

The region is named for the built-in *and* the kinds of its arguments --
``builtin_inc_x_plus_y_r_def_kokkos``, ``builtin_real_to_real_x_r_solver_\
r_def_kokkos`` -- because one built-in name covers every precision LFRic
instantiates it at and the generated C++ differs between them exactly as a
kind-polymorphic kernel's specific procedures do. The formals are named by
position, ``arg1`` to ``argN``, rather than after the algorithm's
variables, so that every call site of one built-in at one set of kinds
generates the same translation unit: the whole-model capture requires the
sites of one region symbol to agree on its text.
"""

from psyclone.core import AccessType
from psyclone.domain.lfric import LFRicConstants
from psyclone.domain.lfric.lfric_builtins import LFRicBuiltIn
from psyclone.psyGen import BuiltIn, InvokeSchedule
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Container, FileContainer, KernelSchedule,
    Literal, Reference)
from psyclone.psyir.symbols import (
    ArgumentInterface, ContainerSymbol, DataSymbol, ImportInterface,
    ScalarType, SymbolTable)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosBuiltinMixin:
    """Synthesise the kernel schedule of an LFRic built-in.

    See the module docstring for what is built, what is not and why.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The module a synthesised schedule's kind parameters are imported
    #: from. It is the module LFRic's own kernels import them from, so the
    #: names resolve through the same precision map a kernel file's do.
    _BUILTIN_KINDS_MODULE = "constants_mod"

    #: The prefix of every synthesised schedule's name, so that a region
    #: generated from a built-in says so in the manifest and the profiler.
    _BUILTIN_PREFIX = "builtin_"

    @staticmethod
    def _is_builtin(kernel):
        """Say whether ``kernel`` is an LFRic built-in.

        :param kernel: the kernel a loop holds.
        :type kernel: :py:class:`psyclone.psyGen.Kern`

        :returns: whether the kernel has no file and PSyclone writes its
            body.
        :rtype: bool
        """
        return isinstance(kernel, BuiltIn)

    @classmethod
    def _validate_builtin(cls, node):
        """Refuse a loop holding a built-in this mixin cannot give a body.

        LFRic writes a built-in as a loop over dofs, so admitting the dof
        iteration space reaches them. A built-in whose lowering is one
        assignment over its dofs is given a schedule by
        :py:meth:`_builtin_schedule` and passes; a reduction, whose lowering
        accumulates into a scalar, is refused here by name because the dof
        launch is a flat ``parallel_for`` with no shape for it. A built-in
        of an API other than LFRic has no lowering this mixin knows, and is
        refused too.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if any kernel in the loop is a
            reduction built-in, or a built-in that is not an LFRic one.
        """
        for kernel in node.kernels():
            if not cls._is_builtin(kernel):
                continue
            if not isinstance(kernel, LFRicBuiltIn):
                raise TransformationError(
                    f"LFRicKokkosTrans does not support the builtin "
                    f"'{kernel.name}': it is not an LFRic builtin.")
            if kernel.is_reduction:
                raise TransformationError(
                    f"LFRicKokkosTrans does not support the LFRic builtin "
                    f"'{kernel.name}': it is a reduction, and the dof "
                    "launch has no shape for one.")

    @classmethod
    def _builtin_schedule(cls, kernel):
        """Return the kernel schedule LFRic would have written for a built-in.

        The body is taken from PSyclone's own lowering of the built-in, on a
        copy of the invoke so that the invoke itself is left as it was: the
        lowering replaces the built-in with the assignment it generates,
        and the PSy layer must still hold the built-in when the loop is
        later replaced by the launch call.

        :param kernel: the built-in the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.lfric_builtins.\
LFRicBuiltIn`

        :returns: a schedule with one scalar formal per argument of the
            built-in, in the built-in's argument order and at each
            argument's kind, whose one statement is the built-in's
            assignment over those formals. It sits in a Container in a
            FileContainer, as a schedule read from a kernel file does, so
            that the copies the contract takes keep a scope chain.
        :rtype: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the built-in's lowering is not a
            single assignment.
        """
        assignment = cls._builtin_lowering(kernel)
        table = SymbolTable()
        kinds_module = ContainerSymbol(cls._BUILTIN_KINDS_MODULE)
        table.add(kinds_module)
        kinds = {}
        formals = []
        for index, argument in enumerate(kernel.arguments.args):
            kind = cls._builtin_kind(table, kinds, kinds_module, argument)
            access = (ArgumentInterface.Access.READ
                      if argument.access == AccessType.READ
                      else ArgumentInterface.Access.READWRITE)
            formal = DataSymbol(
                f"arg{index + 1}",
                ScalarType(cls._psyir_intrinsic(argument.intrinsic_type),
                           Reference(kind) if isinstance(kind, DataSymbol)
                           else kind),
                interface=ArgumentInterface(access))
            table.add(formal)
            formals.append(formal)
            cls._substitute_argument(assignment, argument, formal)
        table.specify_argument_list(formals)
        name = cls._builtin_schedule_name(kernel, kinds)
        schedule = KernelSchedule.create(name, table, [assignment])
        container = Container.create(f"{name[:-len('_code')]}_mod",
                                     SymbolTable(), [schedule])
        FileContainer.create(f"{name}.f90", SymbolTable(), [container])
        return schedule

    @staticmethod
    def _builtin_lowering(kernel):
        """Return the assignment PSyclone lowers ``kernel`` to, detached.

        Lowered on a copy of the whole invoke schedule rather than on the
        built-in, because the lowering resolves the field-data symbols and
        the dof index through the invoke's symbol table and replaces the
        built-in in its loop: neither may happen to the invoke being
        captured. The built-in is found again in the copy by its position
        among the invoke's built-ins.

        :param kernel: the built-in the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.lfric_builtins.\
LFRicBuiltIn`

        :returns: the built-in's own assignment, with no parent.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`

        :raises TransformationError: if the lowering is not one assignment,
            which is the shape every non-reduction built-in has.
        """
        invoke_schedule = kernel.ancestor(InvokeSchedule)
        # The field-data symbols the lowering looks up by tag are created
        # when the invoke prepares its PSy-layer symbols, which the code
        # generation does before it lowers anything; a capture may reach
        # the built-in first.
        invoke_schedule.invoke.setup_psy_layer_symbols()
        index = invoke_schedule.walk(LFRicBuiltIn).index(kernel)
        copy = invoke_schedule.copy().walk(LFRicBuiltIn)[index]
        lowered = copy.lower_to_language_level()
        if not isinstance(lowered, Assignment):
            raise TransformationError(
                f"LFRicKokkosTrans does not support the LFRic builtin "
                f"'{kernel.name}': its lowering is not a single assignment "
                f"but a {type(lowered).__name__}.")
        # The comment the lowering attaches names the built-in, and would
        # otherwise reach the generated C++ as a comment in the body.
        lowered.preceding_comment = ""
        return lowered.detach()

    @classmethod
    def _builtin_kind(cls, table, kinds, kinds_module, argument):
        """Return the kind symbol of one argument, creating it on first use.

        :param table: the synthesised schedule's symbol table.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param dict[str, DataSymbol] kinds: the kind symbols created so far,
            by name; extended in place, in the order the arguments name
            them.
        :param kinds_module: the module the kinds are imported from.
        :type kinds_module: :py:class:`psyclone.psyir.symbols.\
ContainerSymbol`
        :param argument: the built-in argument whose kind is wanted.
        :type argument: :py:class:`psyclone.domain.lfric.LFRicKernelArgument`

        :returns: the kind symbol, or the default precision where the
            argument names no kind.
        :rtype: Union[:py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.ScalarType.Precision`]
        """
        name = argument.precision
        if not name:
            return ScalarType.Precision.UNDEFINED
        if name not in kinds:
            kinds[name] = DataSymbol(
                name, ScalarType(ScalarType.Intrinsic.INTEGER,
                                 ScalarType.Precision.UNDEFINED),
                is_constant=True, interface=ImportInterface(kinds_module))
            table.add(kinds[name])
        return kinds[name]

    @staticmethod
    def _psyir_intrinsic(intrinsic_type):
        """Return the PSyIR intrinsic of an LFRic metadata intrinsic name.

        :param str intrinsic_type: ``real``, ``integer`` or ``logical``, as
            the metadata's data type names it.

        :returns: the scalar intrinsic.
        :rtype: :py:class:`psyclone.psyir.symbols.ScalarType.Intrinsic`
        """
        return {
            "real": ScalarType.Intrinsic.REAL,
            "integer": ScalarType.Intrinsic.INTEGER,
            "logical": ScalarType.Intrinsic.BOOLEAN,
        }[intrinsic_type]

    @staticmethod
    def _substitute_argument(assignment, argument, formal):
        """Replace one argument's references in the lowered body by a formal.

        A field argument reaches the lowering as ``<name>_data(df)``, an
        array reference to the field's data symbol subscripted by the dof
        index; every such reference becomes a reference to the formal, and
        the dof index leaves the body with them. A scalar argument reaches
        it as the algorithm's own expression -- a reference to a variable
        or a literal -- and is replaced by matching that expression: every
        reference to the variable, or the literals equal to the literal,
        one per occurrence in the order they appear so that two scalar
        arguments passed the same literal each claim one.

        :param assignment: the lowered body, rewritten in place.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`
        :param argument: the built-in argument being replaced.
        :type argument: :py:class:`psyclone.domain.lfric.LFRicKernelArgument`
        :param formal: the formal standing for it in the schedule.
        :type formal: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if argument.is_field:
            suffix = LFRicConstants().ARG_TYPE_SUFFIX_MAPPING[
                argument.argument_type]
            data_name = f"{argument.name}_{suffix}".lower()
            for reference in assignment.walk(ArrayReference):
                if reference.symbol.name.lower() == data_name:
                    reference.replace_with(Reference(formal))
            return
        actual = argument.psyir_expression()
        if isinstance(actual, Literal):
            matches = [literal for literal in assignment.walk(Literal)
                       if literal == actual]
            if matches:
                matches[0].replace_with(Reference(formal))
            return
        for reference in assignment.walk(Reference):
            # The exact type: an ArrayReference is a Reference too, and a
            # field's data reference must not be mistaken for the scalar.
            # pylint: disable-next=unidiomatic-typecheck
            if (type(reference) is Reference
                    and reference.symbol.name.lower()
                    == actual.symbol.name.lower()):
                reference.replace_with(Reference(formal))

    @classmethod
    def _builtin_schedule_name(cls, kernel, kinds):
        """Name the synthesised schedule for the built-in and its kinds.

        :param kernel: the built-in.
        :type kernel: :py:class:`psyclone.domain.lfric.lfric_builtins.\
LFRicBuiltIn`
        :param dict[str, DataSymbol] kinds: the kind symbols the arguments
            named, in first-use order.

        :returns: ``builtin_<name>_<kinds>_code``, which
            :py:meth:`LFRicKokkosArgumentMixin._region_name` turns into
            ``builtin_<name>_<kinds>_kokkos``.
        :rtype: str
        """
        parts = [cls._BUILTIN_PREFIX + kernel.name.lower()]
        parts.extend(kinds)
        return "_".join(parts) + "_code"


__all__ = ["LFRicKokkosBuiltinMixin"]
