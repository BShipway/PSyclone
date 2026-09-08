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

"""The intrinsics whose value is an array, written as generated loops.

``MATMUL``, ``DOT_PRODUCT``, ``SUM``, ``MINVAL``, ``MAXVAL``, ``TRANSPOSE``
and ``RESHAPE`` have no counterpart in C++ and no Kokkos function to call:
each is a statement's worth of loops rather than an expression. They are also
the largest single blocker in the LFRic catalogue, because the model writes
its basis changes and its column sums with them.

The observation this module is built on is that **none of them needs an array
temporary**. One element of a contraction or a reduction is a *scalar*, so it
is accumulated in a scalar declared beside the loop that produces it;
``TRANSPOSE`` and ``RESHAPE`` do not compute anything at all, being index maps
that are applied to the operand's subscripts where the operand is read. That
matters beyond tidiness: a temporary would have to be part of the launch's
``shmem_size`` request, which
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` writes *before* it
generates the body that would discover the need for one.

It also settles ``MATMUL(TRANSPOSE(m), x)``, which the model writes wherever
it changes basis: the transpose is two subscripts swapped in the operand of
the product, and no matrix is ever laid out the other way round.

:py:class:`KokkosArrayIntrinsics` is a collaborator of
:py:class:`~psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpression` rather than a mixin on the writer, because it answers
about a whole expression -- its shape, and the statements its value needs --
where a visitor handler answers about one node with nowhere to put a
statement. That module owns the nest; this one says what goes in it.
"""

from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, IntrinsicCall, Literal, Range, Reference)
from psyclone.psyir.nodes.array_mixin import ArrayMixin
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType


#: The type of every index a generated loop or an index map produces. Its
#: precision is deliberately undefined: the index is the back-end's own and
#: crosses no interface, and ``gen_declaration`` renders a symbol whose kind
#: the region did not describe as a plain ``int``, which is what it must be.
INDEX_TYPE = ScalarType(ScalarType.Intrinsic.INTEGER,
                        ScalarType.Precision.UNDEFINED)


class KokkosArrayIntrinsics:
    """Write Fortran's array-valued intrinsics as loops over their operands.

    One instance serves one region, alongside the
    :py:class:`~psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpression` that owns it: the names it generates are numbered, and
    a second region must start again from zero.

    Each intrinsic is one of three kinds, and which one it is decides what is
    generated:

    * A **contraction** -- ``MATMUL`` or ``DOT_PRODUCT`` -- whose element is a
      sum of products over one index the two operands share. The index becomes
      a generated loop and the sum a generated scalar.
    * A **reduction** -- ``SUM``, ``MINVAL`` or ``MAXVAL`` -- whose element
      folds one or more of its operand's dimensions into a scalar. With a
      ``dim`` argument one dimension is folded and the rest survive as the
      result's shape; without one, all of them go and the result is a scalar.
    * An **index map** -- ``TRANSPOSE`` or ``RESHAPE`` -- which computes
      nothing, and is applied to the subscripts of the operand underneath it.

    Every reduction starts from ``Kokkos::reduction_identity``, a
    ``constexpr`` device-callable trait, rather than from a literal written
    here. A written ``-DBL_MAX`` would be right for one C type and wrong for
    every other, and would have to be spelt again for each; the trait is right
    for whatever type the accumulator was declared with.

    :param owner: the lowering this belongs to, which owns the nest.
    :type owner: :py:class:`psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpression`
    :param writer: the writer generating the region.
    :type writer: :py:class:`psyclone.psyir.backend.kokkos.KokkosWriter`
    """

    #: The intrinsics whose element is a sum of products over a shared index.
    _CONTRACTIONS = (IntrinsicCall.Intrinsic.MATMUL,
                     IntrinsicCall.Intrinsic.DOT_PRODUCT)

    #: The intrinsics that fold dimensions away, and the ``reduction_identity``
    #: member each one's identity is taken from. The member names are also the
    #: ``Kokkos::`` function each non-additive fold accumulates with.
    _REDUCTIONS = {
        IntrinsicCall.Intrinsic.SUM: "sum",
        IntrinsicCall.Intrinsic.MINVAL: "min",
        IntrinsicCall.Intrinsic.MAXVAL: "max",
        }

    #: The intrinsics that compute nothing and are applied to the subscripts
    #: of whatever they are reading.
    _MAPS = (IntrinsicCall.Intrinsic.TRANSPOSE,
             IntrinsicCall.Intrinsic.RESHAPE)

    #: Every intrinsic this tier answers for.
    INTRINSICS = frozenset(
        set(_CONTRACTIONS) | set(_REDUCTIONS) | set(_MAPS))

    def __init__(self, owner, writer):
        self._owner = owner
        self._writer = writer
        self._results = 0
        self._indices = 0

    # ------------------------------------------------------------------
    # What this tier answers for
    # ------------------------------------------------------------------
    @classmethod
    def handles(cls, node) -> bool:
        """Return whether ``node`` is an intrinsic this tier writes.

        :param node: the node to judge.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether it is one of :py:attr:`INTRINSICS`.
        :rtype: bool
        """
        return (isinstance(node, IntrinsicCall)
                and node.intrinsic in cls.INTRINSICS)

    @classmethod
    def holds(cls, node) -> bool:
        """Return whether ``node`` reads an intrinsic this tier writes.

        :param node: the expression to search.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether any of it is one of :py:attr:`INTRINSICS`.
        :rtype: bool
        """
        return any(cls.handles(call) for call in node.walk(IntrinsicCall))

    @classmethod
    def _outermost(cls, tree):
        """Return the intrinsics of this tier that no other one encloses.

        An intrinsic inside another is that one's operand and is written by
        it -- the ``TRANSPOSE`` of a ``MATMUL(TRANSPOSE(m), x)`` never becomes
        anything of its own -- so only the outermost are generated.

        :param tree: the expression to search.
        :type tree: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the outermost handled calls, outermost-first.
        :rtype: list[:py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        """
        outermost = []
        for call in tree.walk(IntrinsicCall):
            if not cls.handles(call):
                continue
            enclosing = call.ancestor(IntrinsicCall)
            while enclosing is not None and not cls.handles(enclosing):
                enclosing = enclosing.ancestor(IntrinsicCall)
            if enclosing is None:
                outermost.append(call)
        return outermost

    def space(self, node):
        """Return the shape this tier gives an expression, if it gives one.

        A statement may hold several of these intrinsics -- ``lhs =
        matmul(a, b) + sgn * matmul(c, d)`` is the shape the model writes --
        and every array-valued one of them has the shape of the statement, so
        the first is as good an answer as any. A statement whose intrinsics
        are all scalar-valued, ``a(:) = b(:) * dot_product(p, q)``, takes its
        shape from the sections beside them instead and is answered here with
        the empty tuple.

        :param node: the expression whose shape is wanted.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the shape, as
            :py:meth:`~psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpression._space` gives one, or the empty tuple.
        :rtype: Tuple[Tuple[str, str, str], ...]

        :raises VisitorError: as :py:meth:`shape` does.
        """
        for call in self._outermost(node):
            shape = self.shape(call)
            if shape:
                return shape
        return ()

    def consumed(self, node):
        """Return the identity of every array access this tier reads.

        An operand of one of these intrinsics is not a section of the
        statement it appears in: ``matmul(m3(ik,:,:), p_e)`` is rank 1 and its
        operand is rank 2, so a caller taking the statement's shape from the
        first section it finds would take the wrong one. This says which
        accesses to step over while looking.

        :param node: the expression to search.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: ``id()`` of each array access an intrinsic of this tier
            encloses.
        :rtype: set[int]
        """
        return {id(access)
                for call in self._outermost(node)
                for access in call.walk(ArrayMixin)}

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    def shape(self, node):
        """Return the shape one of these intrinsics produces.

        :param node: the intrinsic call, which :py:meth:`handles` accepts.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: one ``(first index, last index, extent)`` triple per
            dimension, and the empty tuple where the value is a scalar.
        :rtype: Tuple[Tuple[str, str, str], ...]

        :raises VisitorError: if the operands' ranks are not ones the
            intrinsic is defined over, if a reduction's ``dim`` is not a
            literal naming one of its operand's dimensions, or if an operand
            is not an array the region described.
        """
        intrinsic = node.intrinsic
        if intrinsic in self._CONTRACTIONS:
            return self._contraction_shape(node)
        if intrinsic in self._REDUCTIONS:
            space = self._operand_space(node.arguments[0])
            dimension = self._dimension(node)
            if dimension is None:
                return ()
            return space[:dimension - 1] + space[dimension:]
        if intrinsic is IntrinsicCall.Intrinsic.TRANSPOSE:
            space = self._operand_space(node.arguments[0])
            if len(space) != 2:
                raise VisitorError(
                    f"The Kokkos back-end cannot write 'TRANSPOSE' of a "
                    f"rank-{len(space)} operand: it is defined over a matrix.")
            return tuple(reversed(space))
        self._reshaped(node)
        return tuple(("1", extent, extent)
                     for extent in self._reshape_extents(node))

    def _contraction_shape(self, node):
        """Return the shape a contraction produces.

        ``DOT_PRODUCT`` produces a scalar; a ``MATMUL`` keeps whichever of its
        operands' outer dimensions the contraction did not consume.

        :param node: the contraction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the shape, as :py:meth:`shape` gives one.
        :rtype: Tuple[Tuple[str, str, str], ...]

        :raises VisitorError: as :py:meth:`_contracted` does.
        """
        # Called for its refusals as much as for its answer: the ranks are
        # checked in one place whether or not the result has a shape.
        self._contracted(node)
        if node.intrinsic is IntrinsicCall.Intrinsic.DOT_PRODUCT:
            return ()
        left = self._operand_space(node.arguments[0])
        right = self._operand_space(node.arguments[1])
        if len(left) == 2 and len(right) == 2:
            return (left[0], right[1])
        return (left[0],) if len(left) == 2 else (right[1],)

    def _operand_space(self, node):
        """Return the shape of one operand of one of these intrinsics.

        An operand is normally a whole array named without any subscript at
        all -- ``matmul(inv_monomial, delta)`` -- whose shape is the one the
        region described for it. A section, ``matmul(m3(ik,:,:), p_e)``, is
        answered by the dimensions its
        :py:class:`~psyclone.psyir.nodes.Range` subscripts name, and a nested
        intrinsic by :py:meth:`shape`.

        :param node: the operand.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: its shape, as :py:meth:`shape` gives one.
        :rtype: Tuple[Tuple[str, str, str], ...]

        :raises VisitorError: if the operand is a whole array the region
            described no extents for, or is not array-valued at all.
        """
        # pylint: disable=protected-access
        if self.handles(node):
            return self.shape(node)
        if isinstance(node, ArrayMixin):
            positions = [position
                         for position, index in enumerate(node.indices)
                         if isinstance(index, Range)]
            if positions:
                return tuple(self._owner._dimension(node, position)
                             for position in positions)
        elif isinstance(node, Reference) and isinstance(
                node.symbol.datatype, ArrayType):
            view = self._writer._views.get(node.symbol.name)
            if view is None:
                raise VisitorError(
                    f"The Kokkos back-end cannot write an array-valued "
                    f"intrinsic over '{node.symbol.name}': the region "
                    "described no array of that name, so there are no extents "
                    "to generate its loops from.")
            return tuple((f"{offset}", f"({offset} + {extent} - 1)", extent)
                         for offset, extent
                         in zip(view.index_offsets, view.extents))
        raise VisitorError(
            f"The Kokkos back-end cannot take the shape of "
            f"'{node.debug_string().strip()}': an array-valued intrinsic's "
            "operand must be a whole array or a section of one.")

    def _dimension(self, node):
        """Return which axis a reduction folds, or ``None`` for all of them.

        :param node: the reduction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the ``dim`` argument's value, counted from one, or ``None``
            where the call has none and folds every dimension.
        :rtype: Optional[int]

        :raises VisitorError: if the call carries an argument other than its
            array and its ``dim``, if the ``dim`` is not an integer literal,
            or if it names no dimension the operand has.
        """
        found = None
        for name, argument in zip(node.argument_names[1:], node.arguments[1:]):
            if name != "dim":
                raise VisitorError(
                    f"The Kokkos back-end cannot write "
                    f"'{node.intrinsic.name}' with a '{name}' argument: it "
                    "generates a fold over whole dimensions and nothing "
                    "narrower.")
            if not (isinstance(argument, Literal)
                    and argument.datatype.intrinsic
                    is ScalarType.Intrinsic.INTEGER):
                raise VisitorError(
                    f"The Kokkos back-end cannot write an array-valued "
                    f"intrinsic: '{node.intrinsic.name}' takes its 'dim' "
                    f"from '{argument.debug_string().strip()}' rather than "
                    "from a literal, so which axis it folds is not known "
                    "where the loops are generated.")
            found = int(argument.value)
        if found is None:
            return None
        rank = len(self._operand_space(node.arguments[0]))
        if not 1 <= found <= rank:
            raise VisitorError(
                f"The Kokkos back-end cannot write "
                f"'{node.intrinsic.name}' with a 'dim' of {found}: its "
                f"operand has rank {rank}.")
        return found

    def _contracted(self, node):
        """Return the dimension a contraction sums over.

        ``MATMUL`` contracts the last dimension of its left operand against
        the first of its right, and ``DOT_PRODUCT`` the only dimension of
        each; in both cases the left operand's last dimension is the one, and
        Fortran requires the right operand's to agree with it.

        :param node: the contraction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the ``(first index, last index, extent)`` of the summed
            dimension.
        :rtype: Tuple[str, str, str]

        :raises VisitorError: if the operands' ranks are not a pair the
            intrinsic is defined over.
        """
        left = self._operand_space(node.arguments[0])
        right = self._operand_space(node.arguments[1])
        ranks = (len(left), len(right))
        allowed = (((1, 1),) if node.intrinsic
                   is IntrinsicCall.Intrinsic.DOT_PRODUCT
                   else ((1, 2), (2, 1), (2, 2)))
        if ranks not in allowed:
            raise VisitorError(
                f"The Kokkos back-end cannot write "
                f"'{node.intrinsic.name}' of a rank-{ranks[0]} and a "
                f"rank-{ranks[1]} operand.")
        return left[-1]

    def _reshaped(self, node):
        """Return the array a ``RESHAPE`` re-views, checking that it may.

        A reshape costs nothing here exactly because it moves no element: the
        result's subscripts are turned into the source's linear position, and
        Fortran's column-major order and ``LayoutLeft`` agree about what that
        position holds. A ``pad`` or an ``order`` breaks that correspondence,
        and a source of rank two or more would need the position unpacked as
        well as packed, which is arithmetic no kernel in the model asks for.

        :param node: the ``RESHAPE`` call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the source array.
        :rtype: :py:class:`psyclone.psyir.nodes.Node`

        :raises VisitorError: if the call carries a ``pad`` or an ``order``,
            or if its source is not rank 1.
        """
        for name in node.argument_names:
            if name in ("pad", "order"):
                raise VisitorError(
                    "The Kokkos back-end cannot write 'RESHAPE' with a 'pad' "
                    "or an 'order': both change which source element a result "
                    "subscript names, and this tier generates a reshape only "
                    "where it changes none.")
        source = node.arguments[0]
        rank = len(self._operand_space(source))
        if rank != 1:
            raise VisitorError(
                f"The Kokkos back-end cannot write 'RESHAPE' of a rank-{rank} "
                "source: only a rank-1 source's linear position is a single "
                "subscript.")
        return source

    def _reshape_extents(self, node):
        """Return the extents a ``RESHAPE`` was asked for.

        :param node: the ``RESHAPE`` call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: one generated extent per dimension of the result.
        :rtype: Tuple[str, ...]

        :raises VisitorError: if the shape is not written as a list of
            extents.
        """
        # pylint: disable=protected-access
        shape = node.arguments[1]
        if not isinstance(shape, ArrayConstructor):
            raise VisitorError(
                f"The Kokkos back-end cannot write an array-valued "
                f"intrinsic: 'RESHAPE' takes its shape from "
                f"'{shape.debug_string().strip()}' rather than from a list of "
                "extents, so the loops it needs cannot be sized.")
        return tuple(self._writer._visit(extent)
                     for extent in shape.children)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def hoist(self, tree, variables, indent):
        """Replace every intrinsic of this tier in ``tree`` by its value.

        ``tree`` is a copy the caller owns, and it is rewritten in place: each
        outermost handled call becomes a reference to the accumulator its
        loops leave the value in, or -- for an index map, which needs no
        loops -- to the operand access the map resolves to. The statements
        those loops are made of are returned for the caller to place ahead of
        the one that reads them, because a loop is not an expression in C++
        and cannot appear where the operand did.

        The value is spliced back as a
        :py:class:`~psyclone.psyir.nodes.Reference` whose symbol's *name* is
        the generated text. A symbol's name is not required to be an
        identifier and the visitor renders a reference as that name, so this
        is how already-generated C++ re-enters a tree that is about to be
        visited; nothing else about the symbol is read.

        :param tree: the expression to rewrite, which the caller owns.
        :type tree: :py:class:`psyclone.psyir.nodes.Node`
        :param variables: the generated name of each index of the nest the
            value is being produced inside, empty where there is no nest.
        :type variables: Sequence[str]
        :param str indent: the indentation the statements are written at.

        :returns: the statements the values need, and the value of ``tree``
            itself where ``tree`` is one of these intrinsics and so could not
            be replaced in place.
        :rtype: Tuple[str, Optional[str]]

        :raises VisitorError: as :py:meth:`shape` and :py:meth:`element` do.
        """
        statements = []
        value = None
        for call in self._outermost(tree):
            indices = list(variables) if self.shape(call) else []
            if call.intrinsic in self._MAPS:
                text = self.element(call, indices)
            else:
                text, generated = self._reduce(call, indices, indent)
                statements.append(generated)
            if call.parent is None:
                value = text
            else:
                call.replace_with(self._generated(text))
        return "".join(statements), value

    def element(self, node, indices):
        """Return the C++ for one element of an operand.

        An index map is resolved here rather than generated: ``TRANSPOSE``
        reverses the subscripts asked for and ``RESHAPE`` packs them into the
        source's linear position, and in both cases what is finally read is
        the array underneath.

        :param node: the operand, or an index map over one.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param indices: the generated index of each of its dimensions, as
            Fortran subscripts: the origin of the array is removed by
            :py:meth:`~psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpressionMixin.arrayreference_node`, which every access reaches.
        :type indices: list[str]

        :returns: the element access.
        :rtype: str

        :raises VisitorError: if a contraction or a reduction appears as an
            operand of another of these intrinsics, or if the operand is not
            an array access at all.
        """
        # pylint: disable=protected-access
        if self.handles(node):
            if node.intrinsic is IntrinsicCall.Intrinsic.TRANSPOSE:
                return self.element(node.arguments[0], list(reversed(indices)))
            if node.intrinsic is IntrinsicCall.Intrinsic.RESHAPE:
                return self.element(self._reshaped(node),
                                    [self._linear(node, indices)])
            enclosing = node.ancestor(IntrinsicCall)
            raise VisitorError(
                f"The Kokkos back-end cannot write '{node.intrinsic.name}' "
                f"inside '{enclosing.intrinsic.name}': the accumulator of a "
                "reduction has to be declared outside the loops that index "
                "it, and this tier hoists one level only.")
        if isinstance(node, ArrayMixin):
            clone = node.copy()
            position = 0
            for index in list(clone.indices):
                if isinstance(index, Range):
                    index.replace_with(self._generated(indices[position]))
                    position += 1
            return self._writer._visit(clone)
        if isinstance(node, Reference) and isinstance(
                node.symbol.datatype, ArrayType):
            return self._writer._visit(ArrayReference.create(
                node.symbol, [self._generated(index) for index in indices]))
        raise VisitorError(
            f"The Kokkos back-end cannot read an element of "
            f"'{node.debug_string().strip()}': an array-valued intrinsic's "
            "operand must be a whole array or a section of one.")

    def _linear(self, node, indices):
        """Return the source subscript a reshaped subscript names.

        The result's subscripts are packed into a zero-based position in the
        leading-dimension-fastest order both Fortran and ``LayoutLeft`` use,
        and the source's own origin is added back, so that the access reaches
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression.\
KokkosArrayExpressionMixin.arrayreference_node` as a Fortran subscript like
        every other.

        :param node: the ``RESHAPE`` call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param indices: the generated index of each dimension of the result.
        :type indices: list[str]

        :returns: the subscript of the source the element lives at.
        :rtype: str
        """
        extents = self._reshape_extents(node)
        first = self._operand_space(self._reshaped(node))[0][0]
        linear = f"({indices[-1]} - 1)"
        for position in reversed(range(len(indices) - 1)):
            linear = (f"(({indices[position]} - 1) + "
                      f"{extents[position]} * {linear})")
        return f"({first} + {linear})"

    def _reduce(self, node, indices, indent):
        """Generate the loops one contraction or reduction is made of.

        :param node: the contraction or reduction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param indices: the generated index of each dimension of the result,
            empty where the value is a scalar.
        :type indices: list[str]
        :param str indent: the indentation the statements are written at.

        :returns: the accumulator's name, and the statements declaring and
            filling it.
        :rtype: Tuple[str, str]

        :raises VisitorError: if the region described no C type for the kind
            of the value, leaving the accumulator with no type to declare.
        """
        # pylint: disable=protected-access
        c_type = self._writer._kind_c_type(node.datatype)
        if c_type is None:
            raise VisitorError(
                f"Cannot accumulate '{node.intrinsic.name}' over "
                f"'{node.debug_string().strip()}': the region described no C "
                "type for its kind.")
        name = self._name("r", "results")
        if node.intrinsic in self._CONTRACTIONS:
            space = (self._contracted(node),)
        else:
            space = self._folded(node)
        variables = [self._name("j", "indices") for _ in space]
        fold = ("sum" if node.intrinsic in self._CONTRACTIONS
                else self._REDUCTIONS[node.intrinsic])
        element = (self._product(node, indices, variables[0])
                   if node.intrinsic in self._CONTRACTIONS
                   else self._folded_element(node, indices, variables))
        update = (f"{name} += {element};\n" if fold == "sum"
                  else f"{name} = Kokkos::{fold}({name}, {element});\n")
        body = f"{indent}{'  ' * len(space)}{update}"
        return name, (
            f"{indent}{c_type} {name} = "
            f"Kokkos::reduction_identity<{c_type}>::{fold}();\n"
            + self._owner._nest(space, variables, body, indent))

    def _folded(self, node):
        """Return the dimensions a reduction folds away.

        :param node: the reduction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: one ``(first index, last index, extent)`` triple per
            generated loop.
        :rtype: Tuple[Tuple[str, str, str], ...]
        """
        space = self._operand_space(node.arguments[0])
        dimension = self._dimension(node)
        if dimension is None:
            return space
        return (space[dimension - 1],)

    def _product(self, node, indices, variable):
        """Return the product a contraction accumulates one term of.

        :param node: the contraction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param indices: the generated index of each dimension of the result.
        :type indices: list[str]
        :param str variable: the generated index of the summed dimension.

        :returns: the product of the two operands' elements.
        :rtype: str
        """
        left, right = node.arguments[0], node.arguments[1]
        left_rank = len(self._operand_space(left))
        right_rank = len(self._operand_space(right))
        first = self.element(
            left, [indices[0], variable] if left_rank == 2 else [variable])
        second = self.element(
            right, [variable, indices[-1]] if right_rank == 2 else [variable])
        return f"{first} * {second}"

    def _folded_element(self, node, indices, variables):
        """Return the operand element a reduction folds in.

        :param node: the reduction call.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param indices: the generated index of each surviving dimension.
        :type indices: list[str]
        :param variables: the generated index of each folded dimension.
        :type variables: list[str]

        :returns: the element access.
        :rtype: str
        """
        dimension = self._dimension(node)
        if dimension is None:
            return self.element(node.arguments[0], list(variables))
        subscripts = list(indices)
        subscripts.insert(dimension - 1, variables[0])
        return self.element(node.arguments[0], subscripts)

    def _name(self, letter, counter):
        """Return the next generated name of one kind.

        :param str letter: the letter the kind of name is written with.
        :param str counter: the attribute counting names of that kind.

        :returns: the name, which no Fortran identifier can collide with.
        :rtype: str
        """
        index = getattr(self, f"_{counter}")
        setattr(self, f"_{counter}", index + 1)
        return f"{self._owner.PREFIX}_{letter}{index}"

    @staticmethod
    def _generated(text):
        """Return a node the visitor renders as already-generated C++.

        :param str text: the C++ to render.

        :returns: a reference whose symbol's name is that text.
        :rtype: :py:class:`psyclone.psyir.nodes.Reference`
        """
        return Reference(DataSymbol(text, INDEX_TYPE))


__all__ = ["INDEX_TYPE", "KokkosArrayIntrinsics"]
