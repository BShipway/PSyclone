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

"""What an array-valued expression lowers to in the generated Kokkos source.

An array expression -- a whole-array assignment such as ``pv_at_quad = ...``,
a section such as ``a(i:j)``, or an array constructor -- has no counterpart in
C++. It lowers to a *nest*: one generated ``for`` per dimension of the shape
it produces, whose body assigns one element, writing either into an array the
caller named or into a temporary this module allocates.

This module is the lowering itself and nothing else:
:py:class:`KokkosArrayExpression` decides between a nest, a subview and a
copy, generates whichever it chose, and describes each temporary it allocates
as a :py:class:`KokkosScratch` so that the launch can ask for the memory. It
holds no visitor handler and is not mixed into the writer; it is handed an
expression and answers with C++. The writer side -- which node kinds the
Kokkos writer claims, and when a statement is handed here at all -- is
:py:class:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin`, in a module of its own.

Split out of :py:mod:`psyclone.psyir.backend.kokkos` rather than added to it,
which was within thirty lines of the size a module of this project may reach.
"""

from dataclasses import dataclass
from typing import Tuple, Union

from psyclone.psyir.backend.kokkos_array_intrinsics import (
    INDEX_TYPE as _INDEX_TYPE, KokkosArrayIntrinsics)
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import ArrayReference, Range, Reference
from psyclone.psyir.nodes.array_mixin import ArrayMixin
from psyclone.psyir.symbols import ArrayType, DataSymbol


@dataclass(frozen=True)
class KokkosScratch:
    """A kernel-local array placed in Kokkos team scratch.

    A Fortran automatic local such as ``real(r_def), dimension(nlayers) ::
    x_new`` crosses no interface, so it is described here rather than among
    the region's arguments: it must not appear in the generated C ABI, and it
    is not a kernel formal the region has to account for. Its extents are
    integer expressions over the region's scalar arguments, such as
    ``nlayers``, ``4`` or ``(nlayers + 1)``, which is what lets the generated
    C++ size it: a scratch size is computed on the host before the launch,
    where only those scalars are in scope.

    ``index_offsets`` and ``extra_indices`` are carried, and the latter is
    always empty, so that one array-reference table can hold both Views and
    scratch and be read without a type test.
    """

    name: str
    c_type: str
    extents: Tuple[str, ...]
    #: As :py:attr:`KokkosView.index_offsets`.
    index_offsets: Tuple[Union[int, str], ...] = ()
    extra_indices: Tuple[str, ...] = ()


class KokkosArrayExpression:
    """Lower an array-valued expression to a nest, a subview or a copy.

    One instance serves one region, because the names it generates are
    numbered and a second region must start again from zero. The writer holds
    it, hands it every array-valued expression it meets, and gives it back the
    state it needs: the region's Views, its kinds, and the visitor by which a
    sub-expression becomes C++.

    Three answers are possible, and which one is given is a property of the
    expression rather than a choice:

    * A **subview**, for a contiguous whole-dimension slice of a described
      View wanted as a value. ``Kokkos::subview`` of a ``LayoutLeft`` View
      over its leading dimensions is itself a View over the same storage, so
      nothing is copied and nothing is allocated.
    * A **copy**, for a slice that is not contiguous in the layout. The
      elements are gathered into a temporary by a nest, and the generated
      source says in a comment which dimension made the slice non-contiguous,
      so a reader of the C++ can see which of the two happened without
      reading this class.
    * A **nest**, for everything else: one generated ``for`` per dimension of
      the shape, innermost over the dimension the layout makes fastest.

    The loop and temporary names all begin with an underscore, which no
    Fortran identifier may, so a generated name cannot collide with one the
    kernel wrote. They are block-scope names in the generated C++, where a
    single leading underscore is not reserved.

    :param writer: the writer generating the region this belongs to.
    :type writer: :py:class:`psyclone.psyir.backend.kokkos.KokkosWriter`
    """

    #: The prefix every generated name carries. Reserved: a kernel cannot
    #: write it, because Fortran has no identifier starting with ``_``.
    PREFIX = "_kae"

    def __init__(self, writer):
        self._writer = writer
        self._temporaries = []
        self._counter = 0
        self._result = None
        self._intrinsics = KokkosArrayIntrinsics(self, writer)

    @property
    def temporaries(self):
        """Return the temporaries lowering has allocated so far.

        The launch, not this class, declares them: a scratch array must be in
        the region's ``shmem_size`` request, which is written before the body
        that needs it. So a caller that lowers an expression with no
        destination has to add these to the region's scratch itself. Nothing
        in the tree does yet, every array assignment on the LFRic path being
        rewritten into loops before the writer sees it, and every
        :py:meth:`lower` the writer performs having a destination.

        :returns: one description per temporary, in the order allocated.
        :rtype: Tuple[:py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`,
            ...]
        """
        return tuple(self._temporaries)

    @property
    def result(self):
        """Return the name the last :py:meth:`lower` wrote its value to.

        Only meaningful where that call was given no destination, which is
        the case in which the caller cannot know the name in advance.

        :returns: the generated name, or ``None`` if nothing has been lowered
            or the last value went somewhere the caller named.
        :rtype: Optional[str]
        """
        return self._result

    @staticmethod
    def holds(node) -> bool:
        """Return whether an expression reads an array-valued intrinsic.

        A statement that does has to be lowered whether or not it carries a
        section, because the value of ``MATMUL`` or ``DOT_PRODUCT`` is
        accumulated by loops rather than written as an expression.

        :param node: the expression to search.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether any of it is an intrinsic of that tier.
        :rtype: bool
        """
        return KokkosArrayIntrinsics.holds(node)

    def unshapeable(self, node) -> Tuple[str, ...]:
        """Return why an expression could not be lowered, if it could not.

        Asked before there is anything to generate, by a caller deciding
        whether to accept a body at all: what it reports is what
        :py:meth:`lower` would have raised as it took the expression's shape.
        The answer is the writer's own wording, so that a caller reporting it
        keeps no second account of what this can write.

        :param node: the expression to ask about.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: one refusal per distinct wording, in the order met, and the
            empty tuple for an expression this can shape.
        :rtype: Tuple[str, ...]
        """
        return self._intrinsics.unshapeable(node)

    def ranks(self, node) -> Tuple[str, ...]:
        """Return the shape ``node`` produces, one extent per dimension.

        The extents are integer expressions in the region's own scalars, as
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosView.extents` are and
        for the same reason: they are emitted into the generated C++
        verbatim, and a dimension of an LFRic array is ``ndf_w3`` far more
        often than it is a number. A whole dimension of a described array
        gives that array's own extent; a partial section gives the count its
        bounds imply.

        :param node: the expression whose shape is wanted.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the extent of each dimension, outermost first, and the
            empty tuple for an expression that is not array-valued.
        :rtype: Tuple[str, ...]
        """
        return tuple(extent for _, _, extent in self._space(node))

    def lower(self, node, into=None) -> str:
        """Lower an array-valued expression into generated C++.

        :param node: the array-valued expression to lower.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param into: the array the value is written to, as a reference whose
            :py:class:`~psyclone.psyir.nodes.Range` subscripts say which of
            its elements are written, or ``None`` to have a temporary
            allocated and named in :py:attr:`result`.
        :type into: Optional[:py:class:`psyclone.psyir.nodes.Reference`]

        :returns: the generated nest, subview or copy, indented for the
            writer's current depth and ending in a newline.
        :rtype: str

        :raises VisitorError: if ``node`` is neither array-valued nor a
            scalar-valued intrinsic of the array tier, if it writes an array
            it also reads at a different subscript, so that the nest would
            carry a dependence between its iterations, or if a temporary is
            needed and the expression names no type the region described.
        """
        # pylint: disable=protected-access
        # A scalar right-hand side is array-valued only by the destination it
        # is spread over -- ``m(:,:) = 0.0`` -- so where the expression names
        # no space of its own the destination's is the one to generate.
        space = self._space(node)
        if not space and into is not None:
            space = self._space(into)
        if not space:
            # ``x = dot_product(p, q)`` has no shape at all, and yet is not
            # an expression either: its value is accumulated by loops, which
            # go ahead of the assignment rather than inside it.
            if into is not None and self._intrinsics.holds(node):
                return self._scalar(node, into)
            raise VisitorError(
                f"Cannot lower '{node.debug_string().strip()}' to a nest: it "
                "is not an array-valued expression.")
        if into is not None:
            self._check_dependence(node, into)
            self._result = None
        elif self._is_contiguous(node):
            return self._subview(node)

        comment = "" if into is not None else self._copy_comment(node)
        target, offsets = self._target(node, into, space)
        variables = [f"{self.PREFIX}_i{index}" for index in range(len(space))]
        indent = "  " * (self._writer._depth + len(space))
        statements, element = self._element(node, variables, indent)
        assignment = (
            f"{self._access(target, variables, offsets, indent)} = "
            f"{element};\n")
        body = f"{statements}{indent}{assignment}"
        return comment + self._nest(
            space, variables, body, "  " * self._writer._depth)

    def _scalar(self, node, into) -> str:
        """Lower a scalar-valued intrinsic of the array tier to statements.

        :param node: the expression to lower, which reads a reduction.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param into: the scalar the value is assigned to.
        :type into: :py:class:`psyclone.psyir.nodes.Reference`

        :returns: the accumulating loops and the assignment that reads them,
            indented for the writer's current depth.
        :rtype: str
        """
        # pylint: disable=protected-access
        indent = "  " * self._writer._depth
        self._result = None
        clone = node.copy()
        statements, value = self._intrinsics.hoist(clone, (), indent)
        if value is None:
            value = self._writer._visit(clone)
        return (f"{statements}{indent}{self._writer._visit(into)} = "
                f"{value};\n")

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    def _space(self, node):
        """Return the iteration space ``node``'s value is produced over.

        The space is taken from the first array access in the expression that
        carries any :py:class:`~psyclone.psyir.nodes.Range`, because a
        conforming Fortran expression gives every array-valued operand in it
        the same shape.

        :param node: the expression whose space is wanted.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: one ``(first index, last index, extent)`` triple per
            dimension, each already generated as C++, and the empty tuple for
            an expression that is not array-valued.
        :rtype: Tuple[Tuple[str, str, str], ...]
        """
        space = self._intrinsics.space(node)
        if space:
            return space
        # An operand of one of those intrinsics is not a section of this
        # expression -- ``matmul(m3(ik,:,:), p_e)`` is rank 1 where its
        # operand is rank 2 -- so the accesses inside one are stepped over.
        consumed = self._intrinsics.consumed(node)
        for access in node.walk(ArrayMixin):
            if id(access) in consumed:
                continue
            positions = [position
                         for position, index in enumerate(access.indices)
                         if isinstance(index, Range)]
            if positions:
                return tuple(self._dimension(access, position)
                             for position in positions)
        return ()

    def _dimension(self, access, position):
        """Describe one dimension of an access's iteration space.

        :param access: the array access the dimension belongs to.
        :type access: :py:class:`psyclone.psyir.nodes.ArrayReference`
        :param int position: which of its subscripts, counted from zero.

        :returns: the first index, the last index and the extent, as C++.
        :rtype: Tuple[str, str, str]
        """
        # pylint: disable=protected-access
        view = self._writer._views.get(access.name)
        if view is not None and access.is_full_range(position):
            # Answered from the description rather than from the section,
            # whose bounds are ``LBOUND`` and ``UBOUND`` calls that this
            # back-end has no translation for: a whole dimension of a
            # described array runs from its declared origin for its declared
            # extent, which is what those two calls would have returned.
            offset = view.index_offsets[position]
            extent = view.extents[position]
            return (f"{offset}", f"({offset} + {extent} - 1)", extent)
        section = access.indices[position]
        start = self._writer._visit(section.start)
        stop = self._writer._visit(section.stop)
        return (start, stop, f"(({stop}) - ({start}) + 1)")

    # ------------------------------------------------------------------
    # The three answers
    # ------------------------------------------------------------------
    def _is_contiguous(self, node):
        """Return whether ``node`` is a slice that is a View in its own right.

        A ``LayoutLeft`` View has its leading dimension fastest, so a slice of
        it is contiguous exactly when the dimensions it takes whole are the
        leading ones and every later subscript picks a single element. Two
        further conditions are this class's rather than the layout's: the
        slice must be of a described View, since a subview must name one, and
        each of its dimensions must be taken whole, since the generated
        subview asks for ``Kokkos::ALL`` and nothing narrower.

        :param node: the expression to judge.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether it can be passed as a subview.
        :rtype: bool
        """
        # pylint: disable=protected-access
        if not isinstance(node, ArrayReference):
            return False
        if node.name not in self._writer._views:
            return False
        whole = [isinstance(index, Range) and node.is_full_range(position)
                 for position, index in enumerate(node.indices)]
        if not any(whole):
            return False
        if any(isinstance(index, Range) and not taken
               for index, taken in zip(node.indices, whole)):
            return False
        # Leading dimensions first: ``sorted(reverse=True)`` puts every True
        # before every False, which is the shape a contiguous slice has.
        return whole == sorted(whole, reverse=True)

    def _subview(self, node):
        """Return the declaration binding a name to a slice of a View.

        The subview shares the storage it is taken from, so it is not a copy
        and it is not an allocation; ``auto`` is used because the type
        ``Kokkos::subview`` returns is not one this back-end can spell.

        :param node: the contiguous slice to bind.
        :type node: :py:class:`psyclone.psyir.nodes.ArrayReference`

        :returns: the declaration, indented and ending in a newline.
        :rtype: str
        """
        # pylint: disable=protected-access
        view = self._writer._views[node.name]
        arguments = []
        for position, index in enumerate(node.indices):
            if isinstance(index, Range):
                arguments.append("Kokkos::ALL")
                continue
            expression = self._writer._visit(index)
            offset = view.index_offsets[position]
            arguments.append(
                f"({expression} - {offset})" if offset else expression)
        space = self._space(node)
        name = self._reserve(node, space, subview=True)
        return (f"{self._writer._nindent}auto {name} = Kokkos::subview("
                f"{node.name}, {', '.join(arguments)});\n")

    def _copy_comment(self, node):
        """Return the comment a copied slice carries into the generated C++.

        A reader of the generated source can see a nest and a temporary, but
        not why the section was not passed as a subview instead. Saying so
        here is the only place that reason survives the translation.

        :param node: the expression being copied.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the comment, indented and ending in a newline, or the empty
            string where the value being produced is not a slice at all and
            so was never a candidate for a subview.
        :rtype: str
        """
        # pylint: disable=protected-access
        if not isinstance(node, ArrayReference):
            return ""
        return (
            f"{self._writer._nindent}// '"
            f"{node.debug_string().strip()}' is copied rather than passed as "
            "a Kokkos::subview: it is not\n"
            f"{self._writer._nindent}// contiguous in this View's "
            "LayoutLeft, whose leading dimension is the fast one.\n")

    def _nest(self, space, variables, body, base):
        """Wrap the statements producing one element in a loop per dimension.

        The innermost loop runs over the dimension the layout makes fastest,
        which for the ``LayoutLeft`` Views this back-end declares is the
        leading one. Nothing downstream checks this: a nest written the other
        way round computes the same answer and reads the memory in the worst
        possible order, so it is asserted by a test and stated here.

        The body arrives indented rather than being indented here, because a
        nest of the array-valued intrinsic tier holds several statements at
        several depths of its own and only its writer knows where they sit.

        :param space: the iteration space, as :py:meth:`_space` gives it.
        :type space: Tuple[Tuple[str, str, str], ...]
        :param variables: the generated name of each dimension's index.
        :type variables: List[str]
        :param str body: the statements the innermost loop runs, each
            indented and ending in a newline.
        :param str base: the indentation of the outermost loop.

        :returns: the nest, indented and ending in a newline.
        :rtype: str
        """
        text = body
        for position, (start, stop, _) in enumerate(space):
            variable = variables[position]
            # The loop that varies fastest is written last and so indented
            # deepest: position zero is the innermost of the nest, not the
            # outermost, because it is the dimension the layout makes fast.
            indent = base + "  " * (len(space) - 1 - position)
            text = (
                f"{indent}for (int {variable} = {start}; "
                f"{variable} <= {stop}; {variable}++) {{\n"
                f"{text}"
                f"{indent}}}\n")
        return text

    # ------------------------------------------------------------------
    # Temporaries and element access
    # ------------------------------------------------------------------
    def _reserve(self, node, space, subview=False):
        """Name the array a lowered value is produced into, and describe it.

        The description joins the region's scratch, which is what makes the
        launch ask for the memory; it also joins the writer's array table, so
        that a later read of the temporary is subscripted exactly as a read
        of any other described array is. A subview is described but not
        allocated: it is a second name for storage the region already has.

        :param node: the expression whose value the array holds.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param space: the iteration space, as :py:meth:`_space` gives it.
        :type space: Tuple[Tuple[str, str, str], ...]
        :param bool subview: whether the array is a slice of another rather
            than storage of its own.

        :returns: the generated name.
        :rtype: str

        :raises VisitorError: if the expression names no type the region
            described, so the temporary could not be declared.
        """
        # pylint: disable=protected-access
        name = f"{self.PREFIX}_{'sub' if subview else 'tmp'}{self._counter}"
        self._counter += 1
        c_type = self._writer._kind_c_type(node.datatype)
        if c_type is None:
            raise VisitorError(
                f"Cannot allocate a temporary for "
                f"'{node.debug_string().strip()}': the region described no C "
                "type for its kind.")
        description = KokkosScratch(
            name=name, c_type=c_type,
            extents=tuple(extent for _, _, extent in space),
            index_offsets=tuple(start for start, _, _ in space))
        if not subview:
            self._temporaries.append(description)
        self._writer._views[name] = description
        self._result = name
        return name

    def _target(self, node, into, space):
        """Return the array a lowered value is written to and its origins.

        :param node: the expression being lowered.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param into: the destination the caller named, or ``None``.
        :type into: Optional[:py:class:`psyclone.psyir.nodes.Reference`]
        :param space: the iteration space, as :py:meth:`_space` gives it.
        :type space: Tuple[Tuple[str, str, str], ...]

        :returns: the destination, and the origin of each of its dimensions
            where this class has to apply them itself.
        :rtype: Tuple[Union[str,
            :py:class:`psyclone.psyir.nodes.Reference`], Tuple[str, ...]]
        """
        if into is not None:
            return (into, ())
        return (self._reserve(node, space),
                tuple(start for start, _, _ in space))

    def _access(self, target, variables, offsets, indent):
        """Return the C++ subscripting one element of the destination.

        :param target: the destination, as :py:meth:`_target` gives it.
        :type target: Union[str,
            :py:class:`psyclone.psyir.nodes.Reference`]
        :param variables: the generated name of each dimension's index.
        :type variables: List[str]
        :param offsets: the origin of each dimension, empty where the
            destination is a reference whose own description carries them.
        :type offsets: Tuple[str, ...]
        :param str indent: the indentation statements are written at.

        :returns: the element access.
        :rtype: str
        """
        if not isinstance(target, str):
            return self._element(target, variables, indent)[1]
        indices = ", ".join(f"({variable} - {offset})"
                            for variable, offset in zip(variables, offsets))
        return f"{target}({indices})"

    def _element(self, node, variables, indent):
        """Return the C++ for one element of an array-valued expression.

        Every :py:class:`~psyclone.psyir.nodes.Range` in a copy of the
        expression is replaced by a reference to the loop index of its own
        dimension, and the copy is then generated as any other expression is.
        The index generated is the Fortran subscript, not the zero-based View
        one, so that every access still reaches
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.arrayreference_node` and has its
        array's declared origin removed there rather than here.

        A name Fortran allows to stand for its whole array carries no
        subscripts to replace, and is the same value as the section spelling
        it: ``lhs_e = matmul(a, b)`` and ``lhs_e(:) = matmul(a, b)`` mean one
        thing. So such a name is given the nest's index in every one of its
        dimensions, rather than being generated as it stands. Leaving it
        alone would name the View itself where an element was wanted, on
        either side of the assignment -- ``lhs_e = _kae_r0`` assigns a
        ``double`` to a View, and ``_kae_r0 + q`` adds one to the other --
        and neither is an operation Kokkos defines.

        An intrinsic of the array-valued tier is taken out of the copy first,
        while its operands' own :py:class:`~psyclone.psyir.nodes.Range`
        subscripts are still there for it to read, and leaves behind both a
        value and the statements that value is accumulated by.

        :param node: the array-valued expression.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param variables: the generated name of each dimension's index.
        :type variables: List[str]
        :param str indent: the indentation statements are written at.

        :returns: the statements the element needs, and the element
            expression itself.
        :rtype: Tuple[str, str]
        """
        # pylint: disable=protected-access
        clone = node.copy()
        statements, value = self._intrinsics.hoist(clone, variables, indent)
        if value is not None:
            return statements, value
        for access in clone.walk(ArrayMixin):
            position = 0
            for index in list(access.indices):
                if isinstance(index, Range):
                    index.replace_with(Reference(DataSymbol(
                        variables[position], _INDEX_TYPE)))
                    position += 1
        clone = self._subscript_whole_arrays(clone, variables)
        return statements, self._writer._visit(clone)

    @staticmethod
    def _subscript_whole_arrays(clone, variables):
        """Subscript every unsubscripted array name in a copy of an element.

        The rank is required to be the nest's own, which a conforming Fortran
        expression gives every whole-array name in it. A name of some other
        rank is left as it stands rather than subscripted wrongly, because
        this nest has no index to give it: only non-conforming source can
        produce one, and the subscript that would be invented for it would be
        a wrong element rather than a missing one.

        :param clone: the copy being rewritten, which may itself be the name.
        :type clone: :py:class:`psyclone.psyir.nodes.Node`
        :param variables: the generated name of each dimension's index.
        :type variables: List[str]

        :returns: the copy, or the access replacing it where the whole of it
            was one unsubscripted name.
        :rtype: :py:class:`psyclone.psyir.nodes.Node`
        """
        for reference in clone.walk(Reference):
            if isinstance(reference, ArrayMixin):
                continue
            datatype = getattr(reference.symbol, "datatype", None)
            if not isinstance(datatype, ArrayType):
                continue
            if len(datatype.shape) != len(variables):
                continue
            access = ArrayReference.create(
                reference.symbol,
                [Reference(DataSymbol(variable, _INDEX_TYPE))
                 for variable in variables])
            if reference is clone:
                clone = access
            else:
                reference.replace_with(access)
        return clone

    def _check_dependence(self, node, into):
        """Refuse an assignment whose nest would not have independent steps.

        A section that reads the array it writes at some other subscript --
        ``tracer(2:n) = tracer(1:n - 1)`` -- is a copy in Fortran, where the
        right-hand side is evaluated whole before anything is assigned, and a
        recurrence in a nest, where it is not. There is no order of the
        generated loops that recovers the Fortran meaning, so this is refused
        rather than lowered.

        :param node: the expression being lowered.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param into: the destination it is written to.
        :type into: :py:class:`psyclone.psyir.nodes.Reference`

        :raises VisitorError: if the destination is read at a different
            subscript anywhere in the expression.
        """
        written = into.debug_string().strip()
        for access in node.walk(Reference):
            if access.symbol is not into.symbol:
                continue
            read = access.debug_string().strip()
            if read != written:
                raise VisitorError(
                    f"Cannot lower '{written} = "
                    f"{node.debug_string().strip()}' to a nest: "
                    f"'{into.symbol.name}' is written at '{written}' and read "
                    f"at '{read}', so the generated loops would carry a "
                    "dependence from one iteration to the next that Fortran's "
                    "whole-array assignment does not have.")


__all__ = ["KokkosArrayExpression", "KokkosScratch"]
