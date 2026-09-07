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

from psyclone.psyir.nodes import Literal, Reference
from psyclone.psyir.symbols import (
    DataSymbol, ImportInterface, RoutineSymbol, ScalarType, StaticInterface,
    UnsupportedFortranType)
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

    @staticmethod
    def _static_constant(symbol):
        """Return the literal value a module-level ``parameter`` was given.

        A kernel module routinely declares its own constants -- ``nfaces = 4``,
        ``tol = 1.0e-9_r_def`` -- beside the routine that reads them. These
        are compile-time values, so the region carries the *value* rather than
        an argument: importing the symbol into the PSy layer would not even
        compile, since a kernel module is ``private`` by default and makes
        only its ``_code`` routine public.

        An array ``parameter`` such as ``x_dofs(2) = (/ 1, 3 /)`` has no
        literal to substitute and is not a scalar the ABI could carry either,
        so it is left for :py:meth:`_describe_constant` to refuse.

        :param symbol: the symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the declared value, or ``None`` if this symbol is not a
            module-level constant declared with a literal.
        :rtype: Optional[:py:class:`psyclone.psyir.nodes.Literal`]
        """
        if not isinstance(symbol, DataSymbol):
            return None
        if not isinstance(symbol.interface, StaticInterface):
            return None
        if not symbol.is_constant:
            return None
        value = symbol.initial_value
        return value if isinstance(value, Literal) else None

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
            # The initial value is a live piece of the symbol, as a declared
            # bound is; substituting it without copying would move it out of
            # the symbol table and into the body.
            reference.replace_with(value.copy())

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
        * a module-level ``parameter`` declared with a literal value, which
          :py:meth:`_substitute_constants` writes into the body instead.

        The first two are refused by :py:meth:`_validate_body` and by the
        backend well before this, so skipping them here loses no check. What
        it buys is that each refusal names one fact: a kernel calling a
        function was reported both as an uncapturable call and as a module
        constant that could not be passed by value.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: ``(name, container, c_type)`` per constant, name-ordered.
        :rtype: list[tuple[str, str, str]]

        :raises TransformationError: as :py:meth:`_describe_constant` does,
            for any constant that has no place on the generated C ABI.
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
            if symbol.name in constants:
                continue
            constants[symbol.name] = cls._describe_constant(symbol)
        return [constants[name] for name in sorted(constants)]

    @classmethod
    def _describe_constant(cls, symbol):
        """Resolve one non-local symbol onto the generated C ABI.

        :param symbol: the imported symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the symbol's name, the container it is imported from, and
            the C type it is passed by value as.
        :rtype: tuple[str, str, str]

        :raises TransformationError: if the symbol is a module-level
            ``parameter`` whose value is not a literal, such as an array.
        :raises TransformationError: if the symbol is neither a kernel
            argument nor imported from a module.
        :raises TransformationError: if its type cannot be resolved, which
            means the source of its container is not on the module search
            path.
        :raises TransformationError: if it turns out to name a routine, which
            only becomes visible once its container has been read.
        :raises TransformationError: if its kind is not one
            :py:attr:`_C_TYPES` maps.
        """
        if not isinstance(symbol.interface, ImportInterface):
            if isinstance(symbol.interface, StaticInterface) and getattr(
                    symbol, "is_constant", False):
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture '{symbol.name}': a "
                    "module-level constant is written into the region as its "
                    "value, and this one was not declared with a literal "
                    "value the region could carry.")
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
        return (symbol.name, container, c_type)

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
