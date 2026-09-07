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

"""Answer what module state a captured region may carry, and how.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

A kernel body reads names that are not its own arguments: a ``parameter``
declared beside it, an array of them, a variable of its own module, a constant
imported from a configuration module. Each has a different place in the
generated region, and choosing between them is the whole of what this module
does. What a *kind* maps to in C is ``LFRicKokkosTypesMixin``; what a
*declaration* says a shape is, is ``LFRicKokkosBoundsMixin``.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``. Calling a method here directly on this mixin is
therefore not supported.
"""

import re

from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Assignment, BinaryOperation, Container,
    Literal, Node, Reference, UnaryOperation)
from psyclone.psyir.symbols import (
    ArrayType, DataSymbol, ImportInterface, RoutineSymbol, ScalarType,
    StaticInterface, Symbol, UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosConstantsMixin:
    """Place the module state a kernel body reads into the generated region.

    Every question here is about a name the body reads that is not one of its
    arguments: whether the region carries its value, takes it across the ABI,
    or refuses it.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The nodes a value the region can carry is built from. Anything else --
    #: a call, an array constructor, a code block -- is not something a
    #: subscript-free substitution could put into the body and have mean the
    #: same thing, so an initialiser containing one is not folded at all.
    _FOLDABLE = (Literal, Reference, UnaryOperation, BinaryOperation)

    @staticmethod
    def _constant_value(symbol):
        """Return the value a named constant was declared with.

        An imported constant is resolved first, which is what makes a value
        declared in one module readable from the kernel that reads a second
        constant defined in terms of it.

        :param symbol: the symbol a value is wanted for.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the declared value, or ``None`` if the symbol is not a
            constant, or is one whose module could not be read.
        :rtype: Optional[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if isinstance(symbol.interface, ImportInterface):
            try:
                symbol = symbol.resolve_type()
            except Exception:                    # pylint: disable=W0703
                # Untypeable here is untypeable everywhere: the caller that
                # needs this symbol on the ABI reports it by name.
                return None
        if not isinstance(symbol, DataSymbol) or not symbol.is_constant:
            return None
        return symbol.initial_value

    @classmethod
    def _fold(cls, expression):
        """Return a copy of ``expression`` with each named constant replaced.

        ``face_order(n_faces) = [W, S, E, N, B]`` names five constants of
        another module, and ``half = 1.0_r_def / 2.0_r_def`` names an
        arithmetic the compiler will do. Both are values the generated region
        can carry, but only once every name in them has been replaced by what
        it was declared as: the region has no ``W`` to read.

        The replacement is recursive, because a constant may be declared in
        terms of another, and stops at anything that is not arithmetic over
        names and literals. It is a copy throughout: an initial value is a
        live piece of a symbol, and substituting it without copying would move
        it out of the symbol table and into the body.

        :param expression: the declared value to fold.
        :type expression: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the folded copy, or ``None`` if the expression is not
            arithmetic over constants and literals, or names a constant whose
            own value could not be read.
        :rtype: Optional[:py:class:`psyclone.psyir.nodes.Node`]
        """
        folded = expression.copy()
        if any(not isinstance(node, cls._FOLDABLE)
               or isinstance(node, ArrayReference)
               for node in folded.walk(Node)):
            return None
        for reference in folded.walk(Reference):
            value = cls._constant_value(reference.symbol)
            value = None if value is None else cls._fold(value)
            if value is None:
                return None
            if reference is folded:
                folded = value
            else:
                reference.replace_with(value)
        return folded

    @classmethod
    def _static_constant(cls, symbol):
        """Return the value a module-level scalar ``parameter`` was given.

        A kernel module routinely declares its own constants -- ``nfaces = 4``,
        ``tol = 1.0e-9_r_def`` -- beside the routine that reads them. These
        are compile-time values, so the region carries the *value* rather than
        an argument: importing the symbol into the PSy layer would not even
        compile, since a kernel module is ``private`` by default and makes
        only its ``_code`` routine public.

        The value need not be a literal. One written as an expression over
        other constants is folded by :py:meth:`_fold` and carried in the same
        way, because what the region cannot carry is a *name* it has no
        declaration for, not an arithmetic the compiler will do for it.

        An array ``parameter`` such as ``x_dofs(2) = (/ 1, 3 /)`` has no
        single value to substitute; it is carried by
        :py:meth:`_constant_arrays` instead.

        :param symbol: the symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the declared value, folded, or ``None`` if this symbol is
            not a module-level scalar constant with one.
        :rtype: Optional[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(symbol, DataSymbol):
            return None
        if not isinstance(symbol.interface, StaticInterface):
            return None
        if not symbol.is_constant or isinstance(symbol.datatype, ArrayType):
            return None
        value = symbol.initial_value
        return None if value is None else cls._fold(value)

    @classmethod
    def _constant_array(cls, symbol):
        """Return the values of a module-level ``parameter`` array.

        :param symbol: the symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: one folded value per element, or ``None`` if this symbol is
            not a one-dimensional ``parameter`` array declared from its
            origin with a constructor of values the region can carry. A rank
            above one is excluded because the generated declaration is a C
            array in Fortran storage order, which only agrees with the
            subscripts the body writes while there is one dimension.
        :rtype: Optional[tuple[:py:class:`psyclone.psyir.nodes.Node`, ...]]
        """
        if not (isinstance(symbol, DataSymbol)
                and isinstance(symbol.interface, StaticInterface)
                and symbol.is_constant):
            return None
        datatype = symbol.datatype
        if not (isinstance(datatype, ArrayType) and len(datatype.shape) == 1
                and isinstance(datatype.shape[0].lower, Literal)):
            return None
        value = symbol.initial_value
        if not isinstance(value, ArrayConstructor):
            return None
        values = tuple(cls._fold(child) for child in value.children)
        return None if any(item is None for item in values) else values

    @classmethod
    def _constant_arrays(cls, schedule):
        """Describe the ``parameter`` arrays the body reads.

        Each becomes a constant the generated launch body declares for itself
        rather than an argument: its values are in the Fortran, and a kernel
        module is ``private`` by default, so there is nothing for the PSy
        layer to import and pass even if passing it were worth the argument.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: one description per array, name-ordered.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos_constant.KokkosConstant`,
            ...]

        :raises TransformationError: for an array of a kind the generated
            unit has no type for.
        """
        described = {}
        for reference in schedule.walk(Reference):
            symbol = reference.symbol
            if symbol.name in described:
                continue
            values = cls._constant_array(symbol)
            if values is None:
                continue
            c_type = cls._c_type(symbol)
            if c_type is None:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot carry '{symbol.name}': only "
                    f"{cls._supported_kinds()} arrays have a place in the "
                    "generated translation unit.")
            described[symbol.name] = KokkosConstant(
                symbol.name, c_type, values,
                index_offsets=(int(symbol.datatype.shape[0].lower.value),))
        return tuple(described[name] for name in sorted(described))

    @classmethod
    def _substitute_constants(cls, schedule):
        """Replace each module-level ``parameter`` by the value it was given.

        The companion of :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_bounds_mixin.LFRicKokkosBoundsMixin._substitute_bounds`, and
        for the same
        reason: the Fortran has already stated the value, so the region
        carries it rather than asking for it. :py:meth:`_constants` skips
        exactly what this replaces, so the two cannot disagree about which
        symbols reach the ABI.

        :param schedule: the kernel schedule being captured, modified in
            place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        """
        for reference in schedule.walk(Reference):
            value = cls._static_constant(reference.symbol)
            if value is None:
                continue
            reference.replace_with(value)

    @classmethod
    def _constants(cls, schedule):
        """Return the module constants the kernel body reads.

        These are not kernel arguments, so the PSy layer has to import each
        one and pass it by value into the region.

        Three kinds of non-local reference are not data the ABI carries, and
        are skipped rather than described:

        * an intrinsic's ``kind`` argument, which names a type -- see
          :py:meth:`_kind_argument`;
        * the routine of a ``Call``, which names something to call. Before
          ``resolve_type`` is reached these arrive as a plain ``Symbol``
          rather than a ``RoutineSymbol``, so the check is structural;
        * a module-level ``parameter``, which :py:meth:`_substitute_constants`
          writes into the body or :py:meth:`_constant_arrays` declares in the
          generated unit instead.

        The first two are refused by :py:meth:`_validate_body` and by the
        backend well before this, so skipping them here loses no check. What
        it buys is that each refusal names one fact: a kernel calling a
        function was reported both as an uncapturable call and as a module
        constant that could not be passed by value.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: ``(name, container, orig_name, c_type, symbol)`` per
            datum, ordered by the name the kernel body reads.
        :rtype: list[tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]]

        :raises TransformationError: as :py:meth:`_describe_constant` and
            :py:meth:`_describe_module_variable` do, for anything that has no
            place on the generated C ABI.
        """
        table = schedule.symbol_table
        local = {symbol.name for symbol in table.argument_list}
        local |= {symbol.name for symbol in table.automatic_datasymbols}
        constants = {}
        for reference in schedule.walk(Reference):
            symbol = reference.symbol
            if symbol.name in local or isinstance(symbol, RoutineSymbol):
                continue
            if cls._called_routine(reference) or cls._kind_argument(reference):
                continue
            if cls._static_constant(symbol) is not None:
                continue
            if cls._constant_array(symbol) is not None:
                continue
            if symbol.name in constants:
                continue
            if isinstance(symbol.interface, ImportInterface):
                constants[symbol.name] = cls._describe_constant(symbol)
            else:
                constants[symbol.name] = cls._describe_module_variable(
                    symbol, schedule)
        return [constants[name] for name in sorted(constants)]

    @classmethod
    def _describe_constant(cls, symbol):
        """Resolve one imported symbol onto the generated C ABI.

        The caller has established that the symbol is imported: a symbol the
        kernel did not import is the business of
        :py:meth:`_describe_module_variable`, and :py:meth:`_constants`
        chooses between the two on the interface, so that each answers about
        one kind of name.

        :param symbol: the imported symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        A name the kernel renamed on import -- ``use water_mod, only: lv =>
        latent_heat`` -- is carried under both names: the region and the
        kernel body know it as ``lv``, and the PSy layer has to repeat the
        rename to import it at all, because ``water_mod`` has no ``lv``.

        :returns: the symbol's name, the container it is imported from, the
            name it has *in* that container if the import renamed it, the C
            type it crosses the ABI as, and the symbol itself, which is what
            :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_call_mixin.LFRicKokkosCallMixin._region_arguments` reads a shape
            from.
        :rtype: tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]

        :raises TransformationError: if its type cannot be resolved, which
            means the source of its container is not on the module search
            path.
        :raises TransformationError: if it turns out to name a routine, which
            only becomes visible once its container has been read.
        :raises TransformationError: if its kind is not one
            :py:attr:`_C_TYPES` maps.
        """
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
        if isinstance(symbol, RoutineSymbol):
            # resolve_type specialises the symbol, so a name the frontend
            # could only guess at is known to be a routine by now even though
            # _constants could not tell when it walked past it.
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}' from "
                f"'{container}': it is a routine rather than data, and a "
                "routine the body calls is refused by the check on calls "
                "rather than passed by value.")
        c_type = cls._c_type(symbol)
        if c_type is None:
            c_type = cls._declared_c_type(symbol)
        if c_type is None:
            raise TransformationError(
                f"LFRicKokkosTrans cannot pass '{symbol.name}' from "
                f"'{container}' by value: only {cls._supported_kinds()} "
                "scalars have a place on the generated C ABI.")
        cls._carried_extents(symbol)
        return (symbol.name, container, symbol.interface.orig_name,
                c_type, symbol)

    @classmethod
    def _carried_extents(cls, symbol):
        """Return the extents a module datum is carried across the ABI with.

        A scalar reports none and crosses by value. An array crosses by
        reference, as a read-only View the generated region sizes for itself
        -- and it can only size it from what it can evaluate, which is
        literals. A name in a declared extent is a name the region has no
        argument for, so an array declared ``(n_profile)`` is refused where
        one declared ``(100)`` is carried.

        :param symbol: the module datum the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: one C extent expression per dimension, empty for a scalar.
        :rtype: tuple[str, ...]

        :raises TransformationError: if a declared extent names a size the
            region could not evaluate.
        :raises TransformationError: as
            :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_bounds_mixin.LFRicKokkosBoundsMixin._bounds` does, for a
            declaration with no explicit bounds -- a deferred shape among
            them.
        """
        extents = cls._extents(symbol)
        names = cls._extent_names(symbol) if extents else set()
        if names:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}': the "
                f"region would have to size it from "
                f"{', '.join(sorted(names))}, which it has no argument for. "
                "Only a module array whose declared extents are literal can "
                "be carried.")
        return extents

    @classmethod
    def _describe_module_variable(cls, symbol, schedule):
        """Resolve one variable of the kernel's own module onto the C ABI.

        A kernel module may declare state beside its routine and read it from
        the body -- ``real(r_def), public :: profile_heights(100)`` is
        gungho's shape. It is not a constant, so the region cannot carry its
        value; it is not an argument, so the PSy layer is not passing it
        already. What it *is* is a name the PSy layer can ``use``, so the
        region takes it as a formal of its own: a by-value scalar for a
        scalar, and a read-only View of the declared extents for an array.
        The actual is read where the launch is made, so the value at region
        entry is the value the region sees.

        That is the whole of the contract, and it is why a write is refused:
        a by-value scalar could not carry an assignment back to the module,
        and a region that silently dropped one would be wrong rather than
        unsupported.

        :param symbol: the non-imported symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param schedule: the kernel schedule being captured, which is what
            says whether the symbol belongs to the module or to the routine.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: as :py:meth:`_describe_constant` does, with no original
            name: a module variable is read under the name it is declared
            with, there being no ``use`` in the kernel to have renamed it.
        :rtype: tuple[str, str, Optional[str], str,
            :py:class:`psyclone.psyir.symbols.DataSymbol`]

        :raises TransformationError: if the symbol is a module-level
            ``parameter`` whose value is not one the region could carry.
        :raises TransformationError: if it belongs to the routine rather than
            to the module, which a local with an initialiser does, Fortran
            giving that one the ``SAVE`` attribute.
        :raises TransformationError: if it is neither, and so is nothing the
            generated code could reach.
        :raises TransformationError: if the module declares it ``private``,
            leaving the PSy layer no name to import.
        :raises TransformationError: if the region assigns to it.
        :raises TransformationError: if its declaration is one PSyIR could
            not model -- ``allocatable`` and a derived type among them -- or
            states a kind :py:attr:`_C_TYPES` does not map.
        :raises TransformationError: as :py:meth:`_carried_extents` does.
        """
        if isinstance(symbol.interface, StaticInterface) and getattr(
                symbol, "is_constant", False):
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}': a "
                "module-level constant is written into the region as its "
                "value, and this one was not declared with a literal "
                "value the region could carry.")
        table = symbol.find_symbol_table(schedule)
        scope = None if table is None else table.node
        if not isinstance(scope, Container):
            if isinstance(symbol.interface, StaticInterface):
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture '{symbol.name}': it is "
                    "declared in the routine with an initialiser, which "
                    "Fortran gives the SAVE attribute. Nothing outside the "
                    "routine declares it, so there is no name for the PSy "
                    "layer to pass and no way for the region to keep a value "
                    "from one call to the next.")
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}': it is "
                "neither a kernel argument nor imported from a module.")
        if symbol.visibility is not Symbol.Visibility.PUBLIC:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}' from "
                f"'{scope.name}': the module declares it private, so the PSy "
                "layer cannot import it to pass it to the region.")
        if any(isinstance(assignment.lhs, Reference)
               and assignment.lhs.symbol is symbol
               for assignment in schedule.walk(Assignment)):
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}' from "
                f"'{scope.name}': the body assigns to it, and the region is "
                "given the value the module holds at entry rather than a "
                "share in the module's own storage.")
        c_type = cls._c_type(symbol)
        if c_type is None:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}' from "
                f"'{scope.name}': only {cls._supported_kinds()} scalars and "
                "arrays of them have a place on the generated C ABI, and "
                f"'{symbol.datatype}' is not one.")
        cls._carried_extents(symbol)
        return (symbol.name, scope.name, None, c_type, symbol)

    @classmethod
    def _declared_c_type(cls, symbol):
        """Recover a C type from a declaration PSyIR could not model.

        LFRic module constants routinely carry attributes -- ``PROTECTED``,
        an initialiser -- that leave the frontend with an
        :py:class:`UnsupportedFortranType` holding the original text. The kind
        is still stated there, so read it rather than give up; anything with a
        shape is refused, because only scalars are passed by value.

        A declaration that names no kind is read too, and handed to
        :py:meth:`_map_kind` with ``None`` so that it is answered by the one
        table the typed path is answered by. What that costs is a regular
        expression that has to tell ``INTEGER, PUBLIC`` from ``INTEGER(i_def)``
        rather than only looking for ``KIND=``: a positional kind is a kind,
        and reading it as the default would put a silently narrowed value on
        the ABI, so the type name is accepted bare only where the declaration
        goes straight on to its attributes or its ``::``.

        :param symbol: the imported symbol whose declaration is to be read.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C type named by the declaration text, or ``None`` if
            the declaration is not one PSyIR failed to model, has a shape, is
            of no intrinsic the ABI carries, names a kind :py:attr:`_C_TYPES`
            does not map, states one positionally rather than as ``KIND=``, or
            names none where :py:attr:`_DEFAULT_KINDS` admits none.
        :rtype: Optional[str]
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, UnsupportedFortranType):
            return None
        attributes = datatype.declaration.split("::")[0]
        if "DIMENSION" in attributes.upper():
            return None
        match = re.match(
            r"\s*(REAL|INTEGER|LOGICAL)\s*"
            r"(?:\(\s*KIND\s*=\s*(\w+)\s*\)|(?=,|$))",
            attributes, re.IGNORECASE)
        if not match:
            return None
        intrinsic = {
            "real": ScalarType.Intrinsic.REAL,
            "integer": ScalarType.Intrinsic.INTEGER,
            "logical": ScalarType.Intrinsic.BOOLEAN,
        }[match.group(1).lower()]
        kind = match.group(2)
        return cls._map_kind(intrinsic, kind.lower() if kind else None)


__all__ = ["LFRicKokkosConstantsMixin"]
