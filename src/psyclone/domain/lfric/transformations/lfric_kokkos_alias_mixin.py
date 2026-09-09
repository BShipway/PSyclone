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

"""Capture a helper's aliasing pointer local as a Kokkos View handle.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object. It sits beside
:py:class:`~psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin.\
LFRicKokkosInlineMixin`, whose :py:meth:`_inline_calls` calls
:py:meth:`~LFRicKokkosAliasMixin._alias_locals` and which reads the
declaration attributes this one asks it for; the two are one rewrite in two
files because a module of a thousand lines is one pylint refuses.

A Fortran pointer that is only ever aimed at a whole array is not storage.
``field_ptr => log_field`` gives ``field_ptr`` a second name for storage
``log_field`` owns, and every ``field_ptr(k)`` after it reads ``log_field(k)``.
A Kokkos ``View`` says exactly that: a copy of a ``View`` handle shares the
elements of the original, so the whole of the translation is one handle
assignment and no rewriting of the subscripts at all.

That equivalence is narrow, and everything outside it is refused by name:
:py:meth:`~LFRicKokkosAliasMixin._alias_pointer_refusal` lists the uses of a
pointer a ``View`` handle does not reproduce.
"""

from psyclone.psyir.nodes import (
    Assignment, Call, CodeBlock, IntrinsicCall, Reference)
from psyclone.psyir.symbols import (
    ArrayType, AutomaticInterface, UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosAliasMixin:
    """Let a callee's whole-array pointer local be captured as a View handle.

    **What is accepted.** A local declared ``real`` or ``integer`` with a
    ``POINTER`` attribute and nothing else unmodelled, aimed at least once at
    a whole array of the same intrinsic, kind and rank -- a dummy, another
    local, or anything else the region already carries -- and otherwise only
    subscripted. It is re-declared as an array of deferred shape before
    :py:class:`~psyclone.psyir.transformations.InlineTrans` sees it, which is
    what lets the routine be inlined at all, and that deferred shape is the
    mark the rest of the transformation reads it back by.

    **What is refused, and why each is not a handle copy.** A target that is
    a section or an expression aims the pointer at part of an array rather
    than at the array; ``associated`` asks a question a ``View`` has no
    answer to; ``nullify``, ``allocate`` and ``deallocate`` make the pointer
    storage of its own; passing it as an actual hands the question to a
    routine this cannot see; and targets of differing rank or kind are more
    than one handle can hold.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The attributes a local may carry beside ``POINTER`` and still be
    #: captured as an alias. ``DIMENSION`` is matched because it is already
    #: in the partial datatype, as its shape, and so is not an attribute
    #: being dropped. Matched by prefix, since it is written with a
    #: parenthesised value.
    _ALIASABLE_ATTRIBUTES = ("POINTER", "DIMENSION")

    #: The intrinsics that ask a pointer something a ``View`` handle cannot
    #: answer, by the name a refusal spells them with. ``NULLIFY`` is not
    #: among them although
    #: :py:class:`~psyclone.psyir.nodes.IntrinsicCall` names it: the fparser2
    #: frontend does not build one, so a ``nullify`` arrives as a
    #: :py:class:`~psyclone.psyir.nodes.CodeBlock` and is refused with every
    #: other statement PSyclone could not model.
    _ALIAS_REFUSED_INTRINSICS = {
        "ASSOCIATED": IntrinsicCall.Intrinsic.ASSOCIATED,
        "ALLOCATE": IntrinsicCall.Intrinsic.ALLOCATE,
        "DEALLOCATE": IntrinsicCall.Intrinsic.DEALLOCATE,
    }

    @staticmethod
    def _array_type(symbol):
        """Return the array type of ``symbol``, however it is declared.

        A declaration PSyclone could not model keeps the type it *could*
        parse as the partial datatype of its
        :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`, and that
        is the type this asks for: an aliasing pointer and a ``TARGET``
        array are both unmodelled, and both carry the intrinsic, the kind and
        the rank behind the attribute that made them so.

        :param symbol: the symbol to read the type of.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the symbol's array type, or ``None`` where it has none --
            a scalar, or a declaration nothing could be parsed from.
        :rtype: Optional[:py:class:`psyclone.psyir.symbols.ArrayType`]
        """
        datatype = getattr(symbol, "datatype", None)
        if isinstance(datatype, UnsupportedFortranType):
            datatype = datatype.partial_datatype
        return datatype if isinstance(datatype, ArrayType) else None

    @classmethod
    def _array_shape(cls, symbol):
        """Return what two aliased arrays have to agree on.

        Intrinsic, kind and rank, and not the extents: a pointer aimed at one
        array and then at another says nothing about their sizes, and the
        ``View`` handle the alias becomes carries the target's extents with
        it, so ``size(p)`` after ``p => x`` is ``size(x)`` in the generated
        C++ as it is in the Fortran. The kind is compared by name where it is
        a symbol, because two scopes' ``r_tran`` are two symbols naming one
        kind -- and after inlining the callee's declarations sit beside the
        kernel's.

        :param symbol: the symbol to describe.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the intrinsic, the kind's name and the rank, or ``None``
            where the symbol is not an array this could read.
        :rtype: Optional[Tuple]
        """
        array = cls._array_type(symbol)
        if array is None:
            return None
        precision = array.precision
        return (array.intrinsic, getattr(precision, "name", precision),
                len(array.shape))

    @classmethod
    def _alias_pointer_refusal(cls, routine, symbol, assignments):
        """Say why ``symbol`` cannot be captured as an alias, or ``None``.

        Every clause here is a use of the pointer that means something the
        generated ``View`` handle would not mean. A section or an expression
        as a target aims the pointer at part of an array rather than at the
        array, which a handle copy does not express; ``associated`` asks
        whether it is aimed anywhere, which a ``View`` has no answer for;
        ``allocate`` and ``deallocate`` make it storage of its own rather
        than a second name for someone else's; a statement PSyclone could
        not model -- ``nullify`` among them -- may do either without saying
        so; and passing it as an actual hands that question to a routine
        this one cannot see.

        The pointer being written at all is required as well as the rest: a
        pointer nothing aims is read before it names anything, which is a
        program error rather than a shape to capture.

        :param routine: the callee the pointer is declared in.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the pointer local.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param assignments: the pointer assignments to it, in tree order.
        :type assignments: List[:py:class:`psyclone.psyir.nodes.Assignment`]

        :returns: the reason it is refused, or ``None`` where it is accepted.
        :rtype: Optional[str]
        """
        # A refusal is a list of the uses a View handle does not reproduce,
        # and reads better as one than as an arbitrary split into halves.
        # pylint: disable=too-many-return-statements, too-many-branches
        if cls._array_type(symbol) is None:
            return ("its declaration is not one an array type could be read "
                    "from")
        if not assignments:
            return "nothing in the routine aims it at an array"
        for name, intrinsic in cls._ALIAS_REFUSED_INTRINSICS.items():
            for call in routine.walk(IntrinsicCall):
                if call.intrinsic is not intrinsic:
                    continue
                if any(reference.symbol is symbol
                       for reference in call.walk(Reference)):
                    return f"'{name.lower()}' is called on it"
        for block in routine.walk(CodeBlock):
            if symbol.name in block.get_symbol_names():
                return ("a statement PSyclone does not model names it, so "
                        "what it is aimed at afterwards cannot be read")
        for call in routine.walk(Call):
            if isinstance(call, IntrinsicCall):
                continue
            if any(argument.symbol is symbol
                   for argument in call.arguments
                   if isinstance(argument, Reference)):
                return "it is passed as an actual argument"
        for assignment in assignments:
            # ``type(...) is Reference`` and not ``isinstance``: an
            # ArrayReference is a Reference, and it is exactly the case being
            # refused on both sides.
            # pylint: disable-next=unidiomatic-typecheck
            if type(assignment.lhs) is not Reference:  # noqa: E721
                return "it is aimed at through a subscript"
            # pylint: disable-next=unidiomatic-typecheck
            if type(assignment.rhs) is not Reference:  # noqa: E721
                return ("it is aimed at a section or an expression rather "
                        "than at a whole array")
        shapes = {cls._array_shape(symbol)}
        for assignment in assignments:
            shapes.add(cls._array_shape(assignment.rhs.symbol))
        if None in shapes:
            return "it is aimed at something that is not a declared array"
        if len(shapes) > 1:
            return ("it is aimed at arrays that differ in intrinsic, kind "
                    "or rank")
        return None

    @classmethod
    def _alias_locals(cls, call):
        """Re-declare every aliasing pointer local of ``call``'s callee.

        A helper that chooses between two whole arrays with a pointer --
        ``field_ptr => log_field`` in one branch and ``field_ptr => field``
        in the other, and then ``field_ptr(:)`` everywhere after -- is the
        shape LFRic's FFSL schemes use to run one column of arithmetic over
        either of two inputs. The PSyIR does not model ``POINTER``, so the
        local arrives as an
        :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType` with an
        :py:class:`~psyclone.psyir.symbols.UnknownInterface`, and
        :py:class:`~psyclone.psyir.transformations.InlineTrans` refuses the
        routine for declaring a local it cannot place.

        What this does is give such a local the type the frontend did parse
        -- an array of deferred shape -- and an automatic interface, which is
        what lets the routine be inlined. Nothing about the body changes: the
        pointer assignments stay pointer assignments, and the Kokkos writer
        generates each as a ``View`` handle copy, which is what a whole-array
        pointer assignment means. The deferred shape is deliberate and is the
        mark the rest of the transformation reads the alias back by, since a
        local array of any other kind has extents.

        A ``TARGET`` local that the pointer is aimed at is relaxed with it,
        for the same reason
        :py:meth:`_relax_target_arguments` relaxes a ``TARGET`` formal and
        by the same test -- ``log_field`` in the vertical-support helpers is
        a local, not a dummy, and inlining says no more about a local's
        aliasing than it says about a formal's.

        The rewrite is made on the callee as the call site sees it, after
        :py:meth:`_module_inline` has run, so a later capture of another
        kernel calling the same helper meets the routine as its own module
        declares it.

        :param call: the call whose callee is to be rewritten.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :raises TransformationError: if the callee declares a pointer local
            that is not an alias of whole arrays, naming the pointer, the
            routine and which of the clauses in
            :py:meth:`_alias_pointer_refusal` it fell to.
        """
        for routine in cls._local_callees(call):
            for symbol in list(routine.symbol_table.symbols):
                datatype = getattr(symbol, "datatype", None)
                if (symbol.is_argument
                        or not isinstance(datatype, UnsupportedFortranType)):
                    continue
                attributes = cls._declaration_attributes(datatype.declaration)
                if "POINTER" not in attributes:
                    continue
                assignments = [
                    assignment for assignment in routine.walk(Assignment)
                    if assignment.is_pointer
                    and assignment.lhs.symbol is symbol]
                extra = [
                    attribute for attribute in attributes
                    if not attribute.startswith(cls._ALIASABLE_ATTRIBUTES)]
                refusal = (
                    f"its declaration also carries {extra[0]}" if extra
                    else cls._alias_pointer_refusal(
                        routine, symbol, assignments))
                if refusal is not None:
                    raise TransformationError(
                        f"LFRicKokkosTrans cannot capture the pointer "
                        f"'{symbol.name}' of '{routine.name}' as a Kokkos "
                        f"View handle because {refusal}.")
                symbol.datatype = datatype.partial_datatype
                symbol.interface = AutomaticInterface()
                for assignment in assignments:
                    cls._relax_alias_target(assignment.rhs.symbol)

    @classmethod
    def _relax_alias_target(cls, symbol):
        """Give a ``TARGET`` local aimed at by a pointer its own type.

        :py:meth:`_relax_target_arguments` does this for a formal and only
        for a formal, because a formal declared ``TARGET`` is refused by
        ``InlineTrans`` for its type where a local is refused for its
        interface, and the two are separate rewrites. This is the second of
        them, narrowed to the locals an accepted alias is actually aimed at
        rather than to every ``TARGET`` local a callee happens to declare.

        A symbol that is already modelled -- a formal the first rewrite has
        reached, or an array declared without attributes -- is left alone.

        :param symbol: the array the pointer is aimed at.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, UnsupportedFortranType):
            return
        attributes = cls._declaration_attributes(datatype.declaration)
        if datatype.partial_datatype is None or "TARGET" not in attributes:
            return
        if all(attribute.startswith(cls._BINDABLE_ATTRIBUTES)
               for attribute in attributes):
            symbol.datatype = datatype.partial_datatype
            if symbol.is_unresolved or not symbol.is_argument:
                symbol.interface = AutomaticInterface()

    @classmethod
    def _alias_targets(cls, schedule):
        """Return the aliasing pointers of ``schedule`` and what they alias.

        Read back from the body rather than carried alongside it, because
        inlining is a rewrite of the tree and not of anything this mixin
        holds: ``InlineTrans`` renames a callee's local where the caller has
        the name already, and the renamed symbol is still the one the
        pointer assignments name.

        An alias is recognised by the two marks :py:meth:`_alias_locals`
        leaves: a local array a dimension of which is an
        :py:class:`~psyclone.psyir.symbols.ArrayType.Extent` rather than a
        pair of bounds -- ``ATTRIBUTE`` for the ``(:)`` a pointer is usually
        written with, ``DEFERRED`` for the ``dimension(:)`` form -- which
        nothing else among a captured kernel's locals is, since a local with
        no extent is one the launch could reserve no scratch for; and at
        least one pointer assignment aiming it at a whole array.

        :param schedule: the kernel schedule after inlining.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the arrays each aliasing pointer is aimed at, in the order
            the body aims it, keyed by the pointer's name and ordered by
            declaration.
        :rtype: Dict[str, Tuple[str, ...]]
        """
        aliases = {}
        for symbol in schedule.symbol_table.automatic_datasymbols:
            datatype = symbol.datatype
            if not isinstance(datatype, ArrayType):
                continue
            if not any(isinstance(dimension, ArrayType.Extent)
                       for dimension in datatype.shape):
                continue
            targets = []
            for assignment in schedule.walk(Assignment):
                if (assignment.is_pointer
                        and assignment.lhs.symbol is symbol
                        and assignment.rhs.name not in targets):
                    targets.append(assignment.rhs.name)
            if targets:
                aliases[symbol.name] = tuple(targets)
        return aliases

    @classmethod
    def _validate_alias_spaces(cls, schedule):
        """Check that each alias' targets are Views of one Kokkos space.

        The generated region declares an alias ``decltype(t) p;`` for the
        first array ``t`` it is aimed at, and a kernel argument and a
        kernel-local array do not give the same ``t``. An argument is a
        ``View`` of the space the region's data is in and a local is a
        ``View`` of the launch's scratch, so a pointer aimed at one of each
        -- which is the shape LFRic's vertical-support helpers have, choosing
        between a field passed in and a column worked out on the way -- would
        generate a handle assignment the C++ compiler rejects with a template
        error naming neither the pointer nor the kernel. It is refused here
        instead, by name.

        :param schedule: the kernel schedule after inlining.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if an aliasing pointer is aimed both at
            a kernel argument and at a kernel-local array.
        """
        table = schedule.symbol_table
        for name, targets in sorted(cls._alias_targets(schedule).items()):
            if len({table.lookup(target).is_argument
                    for target in targets}) > 1:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture the pointer '{name}' "
                    "as a Kokkos View handle because it is aimed both at a "
                    "kernel argument and at a kernel-local array. The first "
                    "is a View of the space the region's data is in and the "
                    "second a View of the launch's scratch, and no one "
                    "handle can hold both.")


__all__ = ["LFRicKokkosAliasMixin"]
