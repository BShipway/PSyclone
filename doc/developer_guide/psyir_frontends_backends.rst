.. -----------------------------------------------------------------------------
   BSD 3-Clause License

   Copyright (c) 2017-2026, Science and Technology Facilities Council.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
   -----------------------------------------------------------------------------
   Authors: R. W. Ford, A. R. Porter, S. Siso and N. Nobre, STFC Daresbury Lab


PSyIR Frontend and Backends
###########################

Instead of creating PSyIR nodes manually, PSyclone provides
:ref:`psyir-frontends` and :ref:`psyir-backends` to translate from/to
other representations (such as languages like Fortran or C, or other
Intermediate Representations like SIR).

The set of PSyIR nodes that the frontends and backends recognise and translate
to/from are known as the language_level nodes. PSyclone also provide
:ref:`uplifting-lowering` to support higher-level or domain-specific abstractions.


.. _psyir-frontends:

PSyIR Frontends
===============

Currently two Fortran frontends are available:

fparser2:
  This is the main Fortran frontend based on
  `fparser2 <https://github.com/stfc/fparser>`_ and it is currently the only
  recommended stable option.

treesitter:
  This is a highly experimental frontend based on
  `the treesitter parser <https://tree-sitter.github.io>`_ which has the
  potential to be faster, but is currently severely incomplete and untested.

The frontend is selected with the ``psyclone --frontend <frontend>`` flag.

Loop exits
----------

The fparser2 frontend maps an unlabelled ``EXIT`` to an
:ref_guide:`Exit psyclone.psyir.nodes.html#psyclone.psyir.nodes.Exit` node.
Which loop it leaves is decided by where it sits rather than by anything it
says, so ``_exit_handler`` accepts one only where the tree being built
already has an enclosing ``Loop`` or ``WhileLoop``; an ``EXIT`` whose DO is
itself in a ``CodeBlock`` belongs in that ``CodeBlock`` too.

An ``EXIT`` that names its construct, ``EXIT outer``, is not modelled. It may
leave a loop other than the innermost one, which no ``Exit`` node can mean.
In practice the DO handler refuses first -- a named DO whose name is
referenced inside it becomes a ``CodeBlock`` whole -- so the refusal in
``_exit_handler`` is what keeps the two consistent rather than what a parsed
program usually reaches. ``CYCLE`` is not modelled at all, labelled or not.

.. _psyir-backends:

PSyIR Back-ends
===============

PSyIR back-ends translate PSyIR into another form (such as Fortran, C or
OpenCL) using a ``Visitor`` pattern approach.This approach separates the code to
traverse a tree from the tree being visited. The back-end visitor code
is stored in ``psyclone/psyir/backend``.

Visitor Base code
-----------------

``visitor.py`` in ``psyclone/psyir/backend`` provides a base class -
`PSyIRVisitor` - that implements the visitor pattern and is designed
to be subclassed by each back-end.

``PSyIRVisitor`` is implemented in such a way that the PSyIR classes do
not need to be modified. This is achieved by translating the class
name of the object being visited in the PSyIR tree into the method
name that the visitor attempts to call (using the Python ``eval``
function). ``_node`` is postfixed to the method name to avoid name
clashes with Python keywords.

For example, an instance of the ``Loop`` PSyIR class would result in
``PSyIRVisitor`` attempting to call a ``loop_node`` method with the PSyIR
instance as an argument. Note the names are always translated to lower
case. Therefore, a particular back-end needs to subclass
``PSyIRVisitor``, provide a ``loop_node`` method (in this particular example)
and this method would then be called when the visitor finds an instance of
``Loop``. For example:

.. code-block:: python

    from psyclone.psyir.visitor import PSyIRVisitor
    class TestVisitor(PSyIRVisitor):
        ''' Example implementation of a back-end visitor. '''

        def loop_node(self, node):
            ''' This method is called if the visitor finds a loop. '''
            print("Found a loop node")

    test_visitor = TestVisitor()
    test_visitor._visit(psyir_tree)

It is up to the sub-class to call any children of the particular
node. This approach was chosen as it allows the sub-class to control
when and how to call children. For example:


.. code-block:: python

    from psyclone.psyir.visitor import PSyIRVisitor
    class TestVisitor(PSyIRVisitor):
        ''' Example implementation of a back-end visitor. '''

        def loop_node(self, node):
            ''' This method is called if the visitor finds a loop. '''
            print("Found a loop node")
            for child in node.children:
                self._visit(child)

    test_visitor = TestVisitor()
    test_visitor._visit(psyir_tree)

If a ``node`` is called that does not have an associated method defined
then ``PSyIRVisitor`` will raise a ``VisitorError`` exception. This
behaviour can be changed by setting the ``skip_nodes`` option to ``True``
when initialising the visitor i.e.:

.. code-block:: python

    test_visitor = TestVisitor(skip_nodes=True)

Any unsupported nodes will then be ignored and their children will be
called in the order that they appear in the tree.

PSyIR nodes might not be direct subclasses of ``Node``. For example,
``GOKernelSchedule`` subclasses ``KernelSchedule`` which subclasses
``Routine`` which subclasses ``Schedule`` which subclasses ``Node``. This can
cause a problem as a
back-end would need to have a different method for each class e.g. both
a ``gokernelschedule_node`` and a ``kernelschedule_node`` method, even if the
required behaviour is the same. Even worse, expecting someone to have
to implement a new method in all back-ends when they subclass a node
(if they don't require the back-end output to change) is overly
restrictive.

To get round the above problem, if the attempt to call a method with
the name of the PSyIR class (with ``_node`` appended) fails, then the
``PSyIRVisitor`` will subsequently call the method name of its parent
(with ``_node`` appended). This will continue with the ``PSyIRVisitor``
working its way through the class hierarchy in method resolution order
until it is successful (or fails for all names and raises an
exception).

This implementation gives the behaviour one would expect from standard
inheritance rules. For example, if a ``kernelschedule_node`` method is
implemented in the back-end and a ``GOKernelSchedule`` is found then a
``gokernelschedule_node`` method is first tried which fails, then a
``kernelschedule_node`` method is called which succeeds. Therefore all
subclasses of ``KernelSchedule`` will call the ``kernelschedule_node``
method (if their particular specialisation has not been added).

One example of the power of this approach makes use of the fact that
all PSyIR nodes have ``Node`` as a parent class. Therefore, some base
functionality can be added there and all nodes that do not have a
specific method implemented will call this. To see the
class hierarchy, the following code can be written:

.. code-block:: python

    class PrintHierarchy(PSyIRVisitor):
        ''' Example of a visitor that prints the PSyIR node hierarchy. '''

        def node_node(self, node):
        ''' This method is called if no specific methods have been
            written. '''
            print(f"[ {type(node).__name__} start]")
            for child in node.children:
                self._visit(child)
            print(f"[ {type(node).__name__} end]")

    print_hierarchy = PrintHierarchy()
    print_hierarchy._visit(psyir_tree)

In the examples presented up to now, the information from a back-end
has been printed. However, a back-end will generally not want to use
print statements. Output from a ``PSyIRVisitor`` is supported by
allowing each method call to return a string. Reimplementing the
previous example using strings would give the following:

.. code-block:: python

    class PrintHierarchy(PSyIRVisitor):
        ''' Example of a visitor that prints the PSyIR node hierarchy'''

        def node_node(self, node):
            ''' This method is called if the visitor finds a loop '''
            result = f"[ {type(node).__name__} start ]"
            for child in node.children:
                result += self._visit(child)
            result += f"[ {type(node).__name__} end ]"
            return result

    print_hierarchy = PrintHierarchy()
    result = print_hierarchy._visit(psyir_tree)
    print(result)

As most back-ends are expected to indent their output based in some
way on the PSyIR node hierarchy, the ``PSyIRVisitor`` provides support
for this. The ``self._nindent`` variable contains the current
indentation as a string and the indentation can be increased by
increasing the value of the ``self._depth`` variable. The initial depth
defaults to 0 and the initial indentation defaults to two
spaces. These defaults can be changed when creating the back-end
instance. For example:

.. code-block:: python

    print_hierarchy = PrintHierarchy(initial_indent_depth=2,
                                     indent_string="***")

The ``PrintHierarchy`` example can be modified to support indenting by
writing the following:


.. code-block:: python

    class PrintHierarchy(PSyIRVisitor):
        ''' Example of a visitor that prints the PSyIR node hierarchy
        with indentation'''

        def node_node(self, node):
            ''' This method is called if the visitor finds a loop '''
            result = f"{self._nindent}[ {type(node).__name__} start ]\n"
        self._depth += 1
        for child in node.children:
            result += self._visit(child)
        self._depth -= 1
        result += f"{self._nindent}[ {type(node).__name__} end ]\n"
        return result

    print_hierarchy = PrintHierarchy()
    result = print_hierarchy._visit(psyir_tree)
    print(result)

As a visitor instance always calls the ``_visit`` method, an alternative
(functor) implementation is provided via the ``__call__`` method in the
base class. This allows the above example to be called in the
following simplified way (as if it were a function):

.. code-block:: python

    print_hierarchy = PrintHierarchy()
    result = print_hierarchy(psyir_tree)
    print(result)

The primary reason for providing the above (functor) interface is to
hide users from the use of the visitor pattern. This is the interface
to expose to users (which is why ``_visit`` is used for the visitor
method, rather than ``visit``). An important characteristic of the ``__call__``
method is that it will manage the lowering of DSL-concepts because the
backends should not provide specific visitors for concepts that do not relate
directly to the language domain (more information about the lowering step is
provided in the :ref:`uplifting-lowering` section below). This step is done
internally without exposing side effects (e.g. modifications to the provided
tree). This is important because it permits the generation of backend code
without altering the existing PSyIR tree, thus simplifying debugging and
development. For instance the walk statement in the following example will
return the same nodes, regardless of whether or not the print statement
is commented out:

.. code-block:: python

    print_hierarchy = PrintHierarchy()
    # print(print_hierarchy(psyir_tree))
    psyir_tree.walk(APIHaloExchange)

.. note::
    The property of not having side effects is implemented by making a copy
    of the whole tree provided as an argument to the visitor functor. An
    alternative that was explored was modifying the lowering implementation
    so that it returned a new sub-tree instead of modifying the current one
    in-place. This turned out to be complicated as the lowering method doesn't
    have a well defined region where the modification can happen (e.g. a DSL
    concept could need the addition of imports and new symbols defined in
    an ancestor symbol table).


PSyIR Validation
----------------

Although some validation is performed during the Node creation and when
applying transformations, there are often constraints that can only be checked
once the tree is complete, i.e. at the point that a backend is used to generate
code. One such example is that an OpenMP `do` directive must appear within an
OpenMP `parallel` region.

For this reason, the base PSyVisitor class provides support for this global
validation by calling the ``validate_global_constraints()`` method of each
Node that it visits. The ``Node`` base class contains an empty implementation
of this method. Therefore, if a subclass of ``Node`` is subject to certain
global constraints then it must override this method and implement the
required checks. If those checks fail then the method should raise a
``GenerationError``.

Note that, if required, this validation may be disabled by passing
``check_global_constraints=False`` when constructing the PSyIRVisitor
instance::

    print_hierarchy = PrintHierarchy(check_global_constraints=False)

 
Available back-ends
-------------------

Currently, there are two back-ends capable of generating Kernel
code (a KernelSchedule with all its children), these are:

- `FortranWriter()` in `psyclone.psyir.backend.fortran`
- `OpenCLWriter()` in `psyclone.psyir.backend.opencl`

Additionally, there are three partially-implemented back-ends

- `CWriter()` in `psyclone.psyir.backend.c` which handles assignments,
  literals, references, if-blocks, loops, while loops, exits, array
  constructors, unary and binary operations, a subset of intrinsics, and
  directives. It
  has no handler for a `Routine`,
  so it generates the statements and expressions of a body rather than a
  whole kernel. A loop's continuation test follows the sign of its step
  where that sign is visible in the tree, so a Fortran countdown such as
  `do k = n, 1, -1` becomes `for(k=n; k>=1; k+=-1)`; a step that is a
  runtime value is taken to be positive. An array constructor is written
  only where it fills an array, and the C back-end section below says why
  that is the only position it can be written in. A power by a small integer
  literal is written as the multiplications gfortran makes rather than as a
  call to `pow`, so that `x ** 2` is `(x * x)`; which exponents that covers,
  and why the rest keep `pow`, is set out there too, as is which intrinsics
  the subset contains and why four of them depend on their argument's type.
  The three handlers that answer from those tables -- unary operations,
  binary operations and intrinsic calls -- are `CIntrinsicsMixin` in
  `psyclone.psyir.backend.c_intrinsics_mixin`, inherited by `CWriter` ahead
  of `LanguageWriter`, so that the part of the backend that grows one table
  entry at a time is separate from the statements and control flow around it.
- `KokkosWriter()` in `psyclone.psyir.backend.kokkos` which extends
  `CWriter` to generate a complete C++/Kokkos translation unit. It is not
  called on a PSyIR node: it is called on a `KokkosRegion` holding the
  region's name, its scalar arguments, its unmanaged Views, its
  `kind_types`, its `scratch`, the `parallel_loops` and `team_size`
  that select and size the hierarchical launch, the `dof` that selects the
  dof launch, the `cell_start` naming the formal a launch that does not
  begin at zero starts from, the `cell_position` naming the one formal it
  declares rather than takes, and the `colour_map` a region captured from a
  coloured loop declares its cell from, and it visits the loop body through
  `CWriter`. That description -- `KokkosRegion` itself and the
  `KokkosScalar`, `KokkosView`, `KokkosScratch`, `KokkosConstant` and
  `KokkosColourMap` it is made of -- is defined in
  `psyclone.psyir.backend.kokkos_region` and re-exported from
  `psyclone.psyir.backend.kokkos`, so either module may be imported from.
  A `colour_map` and a `cell_start` are not combined: that map's index
  counts the cells of one colour and a first cell counts the mesh's, so a
  region naming both is refused rather than generated, and no LFRic loop
  produces the pair because a coloured loop always begins at `start`.
  Two parts of the back-end live beside it because they
  grow as regions are captured while the writer's own job does not:
  `psyclone.psyir.backend.kokkos_launch` renders three of the four launch
  shapes, one function per shape, with the fourth in
  `psyclone.psyir.backend.kokkos_launch_dof`, and `KokkosIntrinsicsMixin` in
  `psyclone.psyir.backend.kokkos_intrinsics_mixin` holds the intrinsics,
  inherited ahead of `CWriter` so its handlers are found first and fall
  through to `CWriter`'s. `KokkosConstant` in
  `psyclone.psyir.backend.kokkos_constant` is a third,
  `KokkosArrayExpressionMixin` in
  `psyclone.psyir.backend.kokkos_array_expression_mixin` a fourth,
  `KokkosArrayExpression` in
  `psyclone.psyir.backend.kokkos_array_expression` -- the lowering that
  fourth hands an array expression to -- a fifth, and
  `KokkosArrayIntrinsics` in
  `psyclone.psyir.backend.kokkos_array_intrinsics` -- which that fifth owns
  one of per region -- a sixth, and `team_private_scalars` in
  `psyclone.psyir.backend.kokkos_team_scalars` -- which decides, before any
  code is generated, which of the body's scalars belong to one member of a
  team -- a seventh; all are described below. The
  description is built by the LFRic transformation `LFRicKokkosTrans` (see
  the Transformations section of the LFRic chapter in the User Guide), which
  also fixes the C ABI the region is generated against. `kind_types` is
  there because the back-end cannot resolve an LFRic kind to a width: a
  kernel names `r_solver`, and only the LFRic configuration's precision map
  says whether that is 4 bytes or 8. The transformation resolves each kind
  once and hands the answer over as `(kind name, C type)` pairs; an empty
  tuple means "generate as `CWriter` would", which is what every region
  built before the field existed did.
- `SIRWriter()` in `psyclone.psyir.backend.sir` which can generate
  valid SIR from simple Fortran code conforming to the NEMO API.

C back-end
++++++++++

The three handlers below are inherited from `CIntrinsicsMixin` in
`psyclone.psyir.backend.c_intrinsics_mixin` rather than written in
`psyclone.psyir.backend.c`, and are named `CWriter.<handler>` here because
that is where a caller reaches them. Operators and intrinsics share the one
module because they share decisions: `MOD` is an intrinsic spelt as C's `%`
operator, `**` is an operator spelt as the `pow` function, and four
intrinsics are chosen by their argument's type.

`CWriter.intrinsiccall_node` translates an `IntrinsicCall` through a table
that gives each supported intrinsic a C spelling and one of five formatters:
an infix operator, a function call, a cast, a cast wrapped round a function
call, and a right-to-left fold. `NINT` and `FLOOR` need the fourth, because
both return an integer in Fortran while neither `round` nor `floor` does in
C; without the cast, `FLOOR(x)` would silently stay a real.

Four intrinsics have no single right spelling, because Fortran overloads them
on their argument's type and C does not. `REAL_INTRINSIC_ALTERNATIVES` gives
the real spelling of each -- `ABS` becomes `fabs`, `MOD` becomes `fmod`,
`MAX` and `MIN` become `fmax` and `fmin` -- and the module-level helper
`_is_real_argument`, in the same module, chooses between that and the table
entry. Choosing wrongly
is silent in one direction and loud in the other: `abs` binds `::abs(int)` and
truncates a real, while `%` does not compile for one.

`_is_real_argument` answers "no" for every reason it might not know, including
an `UnresolvedType` and a `datatype` property that raises on a tree assembled
by hand. A caller probing the writer with synthetic arguments is asking which
intrinsics it supports rather than what one particular expression is, so the
kind-blind path has to stay reachable rather than becoming an error.

Integer `MAX` and `MIN` are refused rather than translated, which is why
neither has a table entry and both are reachable only through the real
dispatch. C has no standard integer maximum, `fmax` returns a double, and a
conditional expression would evaluate its arguments twice. The refusal is
this writer's alone: `Kokkos::max` and `Kokkos::min` are type-generic, so
`KokkosWriter` overrides `intrinsiccall_node` and generates both.

A cast accepts a second argument and discards it. That argument is a Fortran
kind, so `real(x, r_solver)` and `real(x, r_def)` are both `(double)x` here.
Discarding it is safe only because each cast target is the widest of its
intrinsic, so the value is never narrowed below what was asked for. Honouring
it needs a writer that has been told what each kind's width is, which is what
`kind_types` gives the Kokkos back-end below; that back-end overrides this
method and casts at the width the Fortran asked for.

`CWriter.binaryoperation_node` does not write a constant integer power as a
call to `pow`. `x ** 2` becomes `(x * x)` and `x ** 3` becomes `((x * x) * x)`,
which is what gfortran generates for the same source: its front end never
calls the library for an integer literal exponent, expanding the power by the
binary method into multiplications instead. Each of those multiplications is
correctly rounded by IEEE-754, so the multiplication tree has one right
answer; `pow` has no such guarantee, and glibc's differs from it often enough
to matter. Measured over 200,000 pseudo-random operands, `pow(x, 2)` differs
from the product in the last bit for about one in a thousand of them and
`pow(x, 3)` for about a quarter, and that difference was what stopped a
generated LFRic region from reproducing the model's checksums bit for bit.

`psyclone.psyir.backend.c_integer_power` holds the rule, in three functions so
that each is testable on its own: `literal_exponent` recognises the exponents
the rule applies to, `power_tree` builds the text, and `integer_power` joins
them and answers `None` where `pow` is to be kept. The exponent must be an
integer `Literal`, optionally under a unary sign -- the Fortran frontend
writes `x ** (-2)` as a `MINUS` over `Literal("2")`, so the sign is a node and
not part of the literal's value. A negative exponent over a real base becomes
`(1 / tree)`; `1` rather than `1.0` so that a single-precision base is not
widened by the division. Every product is parenthesised, so the text is
unambiguous wherever it is placed and no operand's own precedence can reach
into it.

What the rule declines is as deliberate as what it writes. A non-literal
exponent, a real exponent, and an exponent of zero all stay `pow`. A negative
exponent over an integer base stays `pow` too, because Fortran evaluates that
in integer arithmetic and the reciprocal would not. So does any exponent above
eight in magnitude: nothing was measured there, and `pow` is the honest answer
where the shape gfortran uses has not been checked. Exponents of five and six
are written, but they are the two the prototype's generated-code probe does
not compare against Fortran: gfortran's chain for them depends on the
optimisation level -- the binary method at `-O0` and `-Og`, GCC's
`powi_table` addition chain from `-O1` -- so there is no single tree that
matches Fortran at every level. The exponents the model's kernels use, two
through four and seven and eight, are the same text under both methods.

What the rule declines becomes a call, and the *name* of that call belongs to
the back-end. `CIntrinsicsMixin._POW_FUNCTION` is `pow` for the C writer,
which is right there: C's `pow` takes and returns `double`, and C has no
Fortran kind for it to disagree with. It is wrong for a back-end whose
operands carry one. A single-precision `x ** y` written as `pow` is computed
in double and rounded back to `float`, where gfortran calls `powf` and rounds
once; the two differ in the last bit for the same reason the product tree
exists at all. The Kokkos back-end overrides the attribute with `Kokkos::pow`,
which is overloaded, so the width follows the operands as it does in Fortran,
and a double power stays the library call it already was. That difference,
like the integer one, was found by a generated LFRic region failing to
reproduce the model's checksums bit for bit, and by nothing else.

`CWriter.arrayconstructor_node` writes an array constructor as one
assignment per element rather than as a value, because C has no array-valued
expression. `assignment_node` therefore hands an assignment whose right-hand
side is a constructor to that callback whole, and the callback returns
statements rather than an expression -- the only handler in this writer that
does. A braced initialiser is not the alternative it appears to be: C accepts
one only on a declaration, and the array being assigned to has been declared
earlier in the body.

That leaves one position the constructor can be written in, and
`_constructor_target` decides whether the assignment is in it. The target must
be a whole rank-1 array, `x = [...]`, or a full-extent section of one
dimension of an array, `x(:) = [...]` or `v(:,1,q) = [...]`, and that
dimension's declared lower bound must be a literal. The elements are then
placed from that lower bound rather than from zero, so the subscript each
element assignment carries is the Fortran one and a writer that re-bases
subscripts -- `KokkosWriter.arrayreference_node`, applying a View's index
offsets -- subtracts the origin exactly once, on the same path as every other
subscript. Every other position -- an actual argument, an operand, a nested
constructor -- raises a `VisitorError` naming the position, because the value
would have to survive as an array of its own and this writer creates no
temporary to hold one. The refusal is raised before any child is visited, so
a caller can use it as a predicate on a constructor it has not otherwise
prepared, which is how the prototype's coverage survey asks whether a kernel's
constructors can be written.

An implied-do constructor is not one of these cases. The fparser2 frontend
does not model an implied do, so `[ (i, i=1,n) ]` becomes a `CodeBlock`
holding the whole constructor and never reaches this handler at all.

`CWriter.exit_node` writes an `Exit` as C's `break`. Fortran's unlabelled
`EXIT` and C's `break` both leave the innermost enclosing loop, so nothing
else is needed to carry one across, and an `EXIT` that names the construct it
leaves never reaches this handler: the frontend leaves it in a `CodeBlock`.
The node validates that it has a loop to leave, so a `break` cannot escape
into a body that has none. A back-end that puts a loop body somewhere a
`break` does not mean the same thing -- a lambda, in the Kokkos back-end's
hierarchical launch -- has to keep such a loop out of that shape itself;
`LFRicKokkosTrans._parallel_loops` is where that is done. It is inherited from
`LFRicKokkosScheduleMixin`, in
`domain/lfric/transformations/lfric_kokkos_schedule_mixin.py`, which holds the
steps that choose the kernel implementation to capture and put its body into
the shape the region is described from.

Kokkos back-end
+++++++++++++++

The Kokkos back-end is limited in the same way as the transformation that
drives it. Only the types in `KokkosWriter._SUPPORTED_TYPES` -- `bool`,
`int`, `float` and `double` -- may appear in a region's signature or in its
`kind_types`, every array becomes an unmanaged `LayoutLeft` View over
storage the caller owns, and the region is entered through an `extern "C"`
function so that Fortran can call it with a `bind(C)` interface. A region
description that breaks those rules raises a `ValueError` before any code is
generated, and a node in the body that `CWriter` has no handler for raises
the usual `VisitorError`. Either reaching a caller means the driving
transformation's own validation was too weak, since it is that validation,
not this back-end, which decides what may be captured.

`kind_types` governs the body alone: `gen_declaration` declares a local at
its own kind's width and `literal_node` suffixes a `float` literal, so a
single-precision kernel is not promoted by `CWriter`'s default of `double`.
Each argument keeps the C type its own description carries, so a region
whose ABI and whose kinds disagree generates the disagreement rather than
hiding it.

That split follows the division of labour the generated code relies on.
Arguments cross the ABI, so the Fortran compiler already checks them: the
`bind(C)` interface names an `iso_c_binding` kind where the PSy layer names
an LFRic one, and a mismatch is a compile error without anything being
generated to make it so. Locals and literals cross nothing, so no compiler
can check them -- which is why they are generated from `kind_types`, and why
`LFRicKokkosTrans` emits a compile-time width assertion per kind into the
interface. The assertion is a `parameter` whose kind is a `merge` over a
`storage_size` comparison, so a false comparison asks for kind `-1` and the
declaration itself is the error.

`bool` is the exception, and is the only supported type with no assertion
behind it. A Fortran `logical` reaches the ABI by conversion rather than by
matching widths: `LFRicKokkosTrans` declares the dummy `logical(c_bool),
value` and wraps the actual in `LOGICAL(..., c_bool)`, which the compiler
performs. There is therefore no width to assert, and asserting one would fail
on exactly the builds this admits -- LFRic's `l_def` is `kind(.false.)` and
measures 4 bytes where PSyclone's precision map records 1, which is issue
#1941. `LFRicKokkosInterfaceMixin._kind_assertions` filters a logical kind out of
both the assertions and their `use constants_mod` line for that reason, and
`_C_LOGICAL_TYPE` is deliberately a separate attribute rather than a
`_C_TYPES` row, since that table is keyed by width and a row would have to
name one.

Intrinsics
~~~~~~~~~~

`KokkosWriter.intrinsiccall_node` overrides `CWriter`'s and tries three
handlers in turn -- a cast, a numeric limit, a function -- falling through to
`CWriter` when none of them recognises the intrinsic. Anything it does write
is qualified `Kokkos::`, because the body becomes a device lambda and the
unqualified `<cmath>` names are host functions; `Kokkos::` picks the device
implementation on a GPU and forwards to `<cmath>` on a host build.

`_KOKKOS_FUNCTIONS` gives the eleven intrinsics whose spelling depends on
nothing: `ACOS`, `ASIN`, `ATAN`, `ATAN2`, `COS`, `EXP`, `LOG`, `SIN`, `SQRT`
and `TAN` keep their names, and `SIGN` becomes `copysign`. `ABS` and `MOD`
are not in it, because they are the two that `CWriter` already spells by
their argument's type: this writer reuses `_is_real_argument` from
`psyclone.psyir.backend.c_intrinsics_mixin` and generates
`Kokkos::fabs` or `Kokkos::abs`, and `Kokkos::fmod` for a real `MOD` while
leaving an integer one to `CWriter`'s `%`.

`MAX` and `MIN` are folded right to left into nested two-argument calls, so
`max(a, b, c)` becomes `Kokkos::max(a, Kokkos::max(b, c))`. This is where the
integer refusal described in the C back-end section above stops applying:
`Kokkos::max` and `Kokkos::min` are templates, so one spelling serves both
types and neither needs a table entry per type. A fold over fewer than two
arguments raises a `VisitorError` rather than generating a call Kokkos has no
overload for.

`FLOOR` and `NINT` keep the cast that `CWriter` wraps round them, since
`Kokkos::floor` and `Kokkos::round` return a real just as their C
counterparts do. Both take an optional second argument naming the integer
kind of their result, and `_kokkos_cast_function` checks it rather than
generating it: the cast is written `(int)` and nothing else, so a region
mapping that kind to any other C type would silently discard the width the
kernel asked for and a request for a 64-bit result would compile as a 32-bit
one that truncates. The kind is read from the second argument's own symbol
and not from the call's datatype, because PSyIR gives `NINT(x, i_def)` an
undefined precision -- the result kind of these two is not carried into the
type they report.

A cast is generated at the width `kind_types` gives, so `real(x, r_solver)`
becomes `(float)x` in a region that describes `r_solver` as `float` where
`CWriter` would write `(double)x`. Two things have to hold before the kind is
honoured: it has to resolve through `kind_types` at all -- which for a kind
no declaration in the body repeats means the transformation collected it from
the cast itself, reading the intrinsic from the call's own datatype since
`i_def` and `r_def` are alike until the cast says integer or real -- and its
C type has
to be one the intrinsic could cast to, which `_KOKKOS_CAST_TYPES` records as
`double` or `float` for `REAL` and `int` for `INT`. The second test is not
redundant. A kindless `real(i)` over an integer `i` has a PSyIR datatype of
`Scalar<REAL, Reference[i_def]>` -- the precision is inherited from the
argument rather than defaulted -- so resolving that kind and using it would
generate `(int)i` and discard the conversion the Fortran asked for. A kind
failing either test is treated as undescribed and the cast is left to
`CWriter`, whose target is the widest of the intrinsic and so never narrows.

`EPSILON` becomes `Kokkos::Experimental::epsilon_v<T>` at the argument's own
width, and is the one intrinsic here that refuses instead of falling through.
There is no kind-blind spelling to fall back to: the trait is a template over
the type, so a region that does not describe the argument's kind raises a
`VisitorError` naming `kind_types` rather than guessing at `double`.

`**` is not an intrinsic call but an operator, and the only part of it this
writer touches is the name of the function `CWriter` falls back to when the
integer-power tree declines. `_POW_FUNCTION` overrides that name with
`Kokkos::pow`. The qualification is the one every function above carries, and
the overloading is the reason it matters here beyond device code: C's `pow`
binds `double` whatever it is given, so a power over `real(kind=r_solver)`
operands was computed in double and rounded back, where gfortran called
`powf` and rounded once. `Kokkos::pow` lets the width follow the operands, and
a double power resolves to the library call it always was. The tree itself is
`CWriter`'s and is not touched: this writer has no `binaryoperation_node`.

`unsupported_intrinsics(schedule, kind_types)` answers the opposite question
-- which of a body's intrinsics this writer could not spell -- so that a
transformation can refuse a kernel when it is asked rather than discover the
refusal part-way through generating. The tables above are not readable from
outside: several handlers search them, one builds its map as a local, and a
second list kept beside them would be a second list to keep in step. Each
call is therefore put to the writer itself, and what raises is reported as
`NAME/arity` in the order first met, without repetition.

It is put as a *probe*: the call is copied and every argument replaced by a
`Reference` to a `DataSymbol` of that argument's own datatype, named
`_PROBE`. The type has to survive because `EPSILON` and the casts are
answered from their argument's kind; the argument itself must not, because an
array the region never described would fail the View lookup and an
unwritable argument would stand in for the call it sits under. `LBOUND`,
`UBOUND` and `SIZE` are not asked -- they are resolved by the array lowering
before the writer sees them -- and neither are the array-valued intrinsics of
the next section, which no handler writes and which lowering replaces first.

Four launch shapes
~~~~~~~~~~~~~~~~~~

The back-end generates one of four launches. A region naming `dof` takes the
dof launch; otherwise one naming any
`parallel_loops` takes the hierarchical launch; otherwise one describing any
`scratch` takes the flat team launch; otherwise the range launch. The
selection is a chain rather than a separate selector field, so nothing can
disagree with it, and each shape is reached only by a region carrying the
field it selects on -- a region built before any of those fields existed
generates
exactly the source it generated then. The launches themselves are rendered by
`psyclone.psyir.backend.kokkos_launch`, one function per shape, except the
dof launch which is `dof_launch` in
`psyclone.psyir.backend.kokkos_launch_dof`; `KokkosWriter` chooses among
them.

A region with no scratch launches over `Kokkos::RangePolicy<>(0, ncells)`
with a `KOKKOS_LAMBDA(const int cell)`. This is the shape every region had
before scratch existed and it is generated unchanged, because a kernel with
no local arrays has nothing to place.

A region naming `dof` launches over `Kokkos::RangePolicy<>(0, ndofs)` with a
`KOKKOS_LAMBDA(const int df)`, and that is the whole of it: the Fortran loop
it replaces iterated over the degrees of freedom of a function space rather
than over cell columns, so the index is a position in each of the Views the
region carries and reaches no dofmap. The shape has no team, so a dof region
naming `scratch` or `parallel_loops` is rejected by `_validate` rather than
launched over a shape that would drop them.

A region naming `cell_start` begins at that formal instead of at zero, which
is what a Fortran loop over the halo cells alone needs. `launch_offsets` in
`psyclone.psyir.backend.kokkos_launch` returns the three pieces of text this
contributes and every shape but the dof one asks it: the range shape uses
the first cell as the policy's own lower bound, and the two team shapes add
it back to the cell they compute from a league rank while shortening the
league by the cells being skipped, so one worker still takes one cell. A
region with no `cell_start` gets `"0"`, no offset and the count unchanged,
which is the text every shape wrote before the field existed.

The index is the region's `cell_index` rather than a fixed `cell`. It
defaults to `cell`, which is what the paragraph above describes, but a
caller renames it when the kernel declares that name itself: the lambda
parameter and the kernel's own declarations share one C++ scope, so a
kernel with an `integer :: cell` local would produce a `conflicting
declaration` error rather than a wrong answer. `LFRicKokkosTrans` chooses
the name with `next_available_name` against the kernel's symbol table, which
gives `cell_1` in that case and `cell` otherwise. Renaming the index alone
is not enough: the per-cell Views' `extra_indices` name it too, and the
back-end takes them as given rather than rewriting them, so whatever builds
the region has to use one name for both.

A region with scratch launches over `Kokkos::TeamPolicy<>`. A Fortran
automatic local such as `real(r_def), dimension(nlayers) :: x_new` has an
extent that is a runtime value, so it cannot become a C++ local; it becomes
a `Kokkos::View` over `team.thread_scratch(0)`, sized by `shmem_size`. That
is the allocation Kokkos provides for exactly this case -- no heap traffic,
no storage outliving the launch, and shared memory rather than global on a
GPU.

Cells are tiled across the ranks of a team rather than given a team each, so
that each rank takes one cell with its own per-thread scratch and the
parallelism stays what the `RangePolicy` shape has: one cell per worker.
The league is sized by rounding up, so the generated body returns early for
a rank whose cell is past the end. The team size is asked of a probe policy
already carrying the scratch request, with `team_size_recommended`: the team
the backend recommends for a `parallel_for` of this functor, which on the
OpenMP backend is one thread, so that the leagues rather than the ranks
carry the parallelism. `team_size_max` is not asked instead. It returns the
whole thread pool there whatever the scratch request, which puts one team on
the whole league with a rendezvous between consecutive cells.

A region naming `parallel_loops` launches over `Kokkos::TeamPolicy<>` too,
but divides the work the other way round: one team per cell, `cell =
team.league_rank()`, and the team's members spread over the cell's levels
rather than over cells. That is the division an LFRic kernel is shaped for,
since a cell's levels are the inner dimension of every field it reads, so
members of one team touch neighbouring elements rather than columns a stride
apart. Each loop the region named becomes a
`Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start, stop + 1), ...)`
-- the `+ 1` converting Fortran's inclusive bound to Kokkos's half-open
range -- followed by an unconditional `team.team_barrier()`. `TeamVectorRange`
rather than `TeamThreadRange` because the two are the same loop on a host and
the former spreads over the whole team on a device whatever shape the team
has. The barrier is unconditional because a statement after the loop may read
what the loop wrote, and deciding whether one does is a second dependence
analysis the back-end does not perform.

Every member of the team executes the rest of the body on its own copy of the
locals, which is right for a scalar and wrong for an array: an element
assigned outside every one of these loops would be written by all the members
at once, which is a race in the memory model even though the values agree. So
`KokkosWriter.assignment_node` wraps such an assignment in
`Kokkos::single(Kokkos::PerTeam(team), ...)` and a barrier, wherever it sits
-- at the top of the body, inside an `if`, or inside a serial loop, where
each iteration's write is wrapped on its own. Scratch moves with the same
change of meaning: `team.team_scratch(0)` with `PerTeam`, since a per-cell
temporary is team-shared once a team is a cell.

A scalar is a third case, and the one whose correctness rests on where its
declaration is written. Every member has its own copy of a local declared at
the top of the functor, so a scalar a spread loop writes and reads back
within an iteration is already the member's own and races with nothing. What
that placement does not survive is a read *after* the loop: with a team of
more than one member the value left in a member's copy is the one the last
iteration that member ran left there, and the member which reads it after the
barrier need not be the member which ran the loop's last iteration. Under
`Kokkos::AUTO` on the OpenMP backend a team has one member, the two are the
same value, and nothing on a host says otherwise.

`kokkos_team_scalars.team_private_scalars` settles this before generation
rather than leaving it to that coincidence. For each loop the region names,
it classifies every scalar the loop writes:

- written before anything in the loop reads it, and not live at the loop's
  exit -- nothing that runs afterwards reads it without writing it first:
  **loop-private**. Its declaration is generated inside that loop's
  `TeamVectorRange` lambda, which gives each iteration its own and takes the
  name out of scope after the loop. The counter of
  a serial loop nested inside a spread loop is private on the same terms,
  its initialisation being the write;
- anything else: **shared**, and the region is refused -- `ValueError` from
  the writer, which `LFRicKokkosTrans` reports as a `TransformationError` and
  the LFRic capture script records as an unmodelled region. Refusing the
  region rather than leaving the declaration where it was is the point: the
  value such a statement wants belongs to the loop's last iteration, and
  after a spread no member holds it.

The second half of that test is liveness rather than "used nowhere else",
because an LFRic kernel reuses its counters and its one or two temporaries
all through a body -- `df` driving a serial loop at team level and then a
nested loop inside a spread one, `t1` holding a difference at team level and
a sum inside a level -- and each of those uses writes before it reads. So a
private declaration *shadows* the region-scope one wherever the body needs
both, and the region-scope declaration is dropped only where nothing outside
the spread loops mentions the name. This is the shadowing the lambda's own
parameter already does to the loop variable's declaration, for the same
reason and with the same consequence: the generated code is not compiled
with `-Wshadow`.

A scalar the loop only reads is neither, and stays where it is: every member
computed it redundantly at team level, so every member's copy holds the same
value.

The classification is a flow analysis over the loop body rather than a pair
of walks, because the property is order-sensitive in a way a walk cannot see.
A loop whose only write to a scalar is inside an `if` with no `else` reads
the previous iteration's value whenever the branch is not taken; an `if` with
an `else` that writes in both arms does not. A construct the analysis does
not model -- a `while`, a call -- contributes reads and no writes, so a
scalar it writes is refused for being unknown rather than made private for
being unexamined. `DependencyTools._is_scalar_parallelisable`, which
`LFRicKokkosTrans` uses when choosing the loops, takes the first access to a
scalar anywhere in the loop and does not ask which branches reach it, so the
back-end's answer is the narrower of the two.

Which loops may be spread is otherwise a dependence judgement, and it is made
by `LFRicKokkosTrans` rather than here: the back-end renders the loops it is
given, and judges only what they do to the body's scalars. It does check what it is given, since none of
the four mistakes announces itself downstream -- an entry must be a `Loop` of
the region's own schedule, not nested inside another chosen loop, and of unit
step, `TeamVectorRange` having no stride. `team_size` renders as a literal in
the policy, `Kokkos::AUTO` when it is `None`; the two flat shapes ignore it,
the range launch having no team and the flat team launch taking the size its
own probe recommends.

A `KokkosScratch` is described separately from the region's arguments, and
deliberately so. It crosses no interface, so it must not appear in the C ABI
or in the generated `bind(C)` interface, and it is not a kernel formal that
the region has to account for. It does carry `index_offsets` and an
always-empty `extra_indices`, so that `arrayreference_node` can resolve Views
and scratch through one table without a type test; a scratch symbol is
skipped when the kernel's locals are declared, since it is already declared
as its View.

Carried constants
~~~~~~~~~~~~~~~~~

A `KokkosConstant` describes a Fortran `parameter` array the kernel body
reads -- `integer(i_def), parameter :: face_order(4) = [1, 2, 3, 4]`. It is
neither an argument nor scratch: the values are stated by the source, and a
kernel module is `private` by default, so there is nothing for the PSy layer
to import and pass even if an argument were worth adding. The writer declares
each one among the body's other locals, first, ahead of the cell position:

.. code-block:: c++

    const int face_order[4] = {1, 2, 3, 4};

**Not at file scope**, which is where a `parameter` beside a kernel would
most naturally go. A namespace-scope array is host data, and `nvcc` compiles
a device lambda that subscripts one and then rejects it --
`identifier "face_order" is undefined in device code` -- for `static const`
and `static constexpr` alike, with and without `--expt-relaxed-constexpr`.
Inside the body it is a local the compiler materialises wherever the body
runs, and that is the only spelling a host and a device execution space both
accept. A file-scope constant compiles for OpenMP and stops the CUDA build of
the same region dead.

The values are held as PSyIR nodes and generated by the writer rather than
carried as text, so that a literal takes the region's own `kind_types` width
rather than `CWriter`'s default. Like `KokkosScratch`, a constant carries
`index_offsets` and an always-empty `extra_indices` and joins the same
array-reference table, so one lookup resolves Views, scratch and constants
alike; the one difference is punctuation, since a constant is a C array and
is subscripted `face_order[(face - 1)]` rather than as a View. It is one
dimensional, because a C array is written in row-major order and only in one
dimension does that agree with the subscripts the Fortran body writes.
`KokkosConstant` lives in `psyclone.psyir.backend.kokkos_constant` rather
than beside the other descriptions because it was moved out first, when
`kokkos.py` was at the size limit this project sets; the rest followed it
into `psyclone.psyir.backend.kokkos_region`, and both are re-exported from
`kokkos` so that neither move is visible to an importer.

The cell position
~~~~~~~~~~~~~~~~~

`cell_position` names the one kernel formal a region declares rather than
takes. LFRic passes its cell index to every kernel that takes an operator,
because the kernel finds its own slice of the operator's local stencil
arithmetically -- `ik = (cell - 1) * nlayers + 1` -- rather than being handed
the slice. That is a formal like any other, and the region's `arguments` do
not describe it; instead the back-end generates

.. code-block:: c++

    const int cell = cell_1 + 1;

as the first line of the launch body, from the launch's own index. The `+ 1`
is the whole of the conversion: the launch index is zero-based and LFRic's
cell is one-based.

The declaration is prepended to the kernel's local declarations rather than
written into each launch, because all three shapes place those declarations
immediately after establishing the index this one reads. `None`, the default,
generates nothing, so a region built before the field existed generates
exactly the source it generated then.

Passing it across the ABI instead is what makes this worth a field. The
value would then be a launch parameter, fixed for the whole region, and the
PSy layer has nothing sensible to pass: its own loop counter is the obvious
candidate and by then it has been lowered away, so every cell computes `ik`
from an unassigned variable. That compiles, links and runs, and produces a
wrong answer, which is why `_validate_cell_position` checks four things none
of which a build would catch -- the name must be a C++ identifier, must be
one of the schedule's formals, must not also be described as a region
argument, and must not be the launch's own cell index, since `const int cell
= cell + 1;` initialises an object from itself.

A write two cells share
~~~~~~~~~~~~~~~~~~~~~~~

Two things in the description answer a write that two cells of one launch
would make to the same element, and a region carries one of them or neither.

The first is `KokkosView.atomic`. A View marked with it has every
read-modify-write of its elements generated as a `Kokkos::atomic_*` call
rather than as an assignment, by `_atomic_update` in
`KokkosArrayExpressionMixin`; `ATOMIC_UPDATES` there maps the four PSyIR
operators Kokkos has an atomic for onto `Kokkos::atomic_add`, `_sub`, `_mul`
and `_div`, and says for each whether the operand may be either side of the
operator, since `a = b - a` is not an `atomic_sub`. The flag is per View, so
a region writing an incremented field and a private one generates an atomic
for the first and an assignment for the second, and it is the update alone
that is atomic: a statement that reads the element elsewhere in its own
right-hand side reads it plainly. An assignment the flag applies to that is
not one of those four shapes, or that is a whole-array expression, raises a
`VisitorError` naming the assignment rather than generating an update that
is not atomic. A View that is both `atomic` and `read_only` is rejected by
`_validate_view`, an atomic on data nothing writes being a contradiction in
the description.

The second is `KokkosColourMap`, which is how a region generated from an
already-coloured loop finds its cell. Such a region is launched over the
cells of one colour rather than over the mesh, so the launch index is no
longer the cell: `launch_index` in `kokkos_launch` returns the colour map's
own `index` name for it, and `KokkosWriter` prepends

.. code-block:: c++

    const int cell = cmap(colour - 1, cell_in_colour) - 1;

to the launch body, in front of the `cell_position` declaration if there is
one. The map is described as a rank-2 read-only View like any other
argument, strided by the number of colours, and the colour being launched is
a scalar; the caller enters the region once per colour. No atomic is
generated for such a region, because the cells of one colour meet at no dof,
and the two fields are alternatives rather than a sequence.

A kernel that takes an operator is given the cell index as an argument, and
under colouring the PSy layer supplies it as `cmap(colour, cell)` rather than
as `cell`. `LFRicKokkosContractMixin._is_the_loops_cell` recognises both
spellings, so the region drops that actual and declares the position from the
map's answer: `const int cell = cell_1 + 1;` follows the lookup above, where
`cell_1` is the mesh cell and `cell` the kernel's own one-based formal. Every
`matrix_vector` call site in GungHo is of this shape, so refusing it would
leave the coloured arm without the commonest shared write there is.

One consequence is worth knowing before a debug build reports it. A coloured
region's per-cell Views -- its dofmaps, and a sliced actual like a stencil's
-- are strided by the launch's cell count, which is the cells of this colour,
while the index they are read at is the mesh cell the map returned. Under
`LayoutLeft` the trailing extent takes no part in an address, so only the
leading one has to be exact and the addresses are the ones the Fortran
computes; a build with `KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK` would nonetheless
report the index as out of range.

Extents
~~~~~~~

A View's and a scratch array's `extents` are strings written into the
generated C++ verbatim, one per dimension. An extent may be an integer
expression over named sizes: `nlayers`, `4` and `(nlayers + 1)` are all
accepted, built from identifiers, decimal literals, `+`, `-`, `*`, `/` and
balanced parentheses. `is_extent` is the predicate, and `extent_names`
reports the identifiers an extent is sized from; both are module-level
functions of `psyclone.psyir.backend.kokkos`, because the transformation
that builds a region has to apply the same grammar before it gets here.

Division is carried rather than refused. Fortran and C++ agree about the
rounding of an integer quotient -- both truncate toward zero -- so
`(stencil_size + 1) / 2` sizes a View as the Fortran declaring it meant. What
neither language defines is an allocation of negative size, and division is
the only operator in the grammar that can reach one from operands a Fortran
declaration would accept. `scratch_guard` therefore emits, for a region whose
scratch extents divide and for no other, a comment stating the truncation
rule it relies on and a `Kokkos::abort` on a negative extent. A region whose
scratch does not divide generates exactly the source it generated before
division was admitted.

What the grammar still refuses is a call. `max(nlayers, 1)` and
`pow(nlayers, 2)` are rejected on the comma, which a `shmem_size` argument
list may not carry, and the refusal is reported rather than left to the C++
compiler.

A scratch array is restricted further than a View: every name in its extents
must be a scalar argument of the region. A scratch size is computed on the
host before the launch, where only the region's scalars are in scope.

Neither shape assumes a lower bound of 1. `index_offsets` carries one value
per dimension, subtracted from each Fortran subscript on the way to the
zero-based element, and it may be an integer or an extent expression --
`is_offset` is the predicate, and admits both. A Fortran local declared
`dimension(0:nlayers-1)` is described by an extent of `nlayers` and an offset
of `0`; one declared `dimension(1:nlayers)` by an extent of `nlayers` and an
offset of `1`. The offset is applied in `arrayreference_node` and nowhere
else, so no path can reach an element of a View without it.

The offset is written out even when it is zero -- `u_e(k - 0)` rather than
`u_e(k)` -- because the alternative is a special case in the one routine
where a missing subtraction is a silent wrong answer rather than a compile
error. The optimiser folds it; a reader of the generated source can see which
origin each subscript was written against.

A View with `extra_indices` and no `index_offsets` is the one case where a
*name* rather than a subscript has to be rewritten. `LFRicKokkosTrans` uses
that shape for a formal the kernel declares as a scalar and the PSy layer
fills per cell -- a 1-D or region stencil's `stencil_size`, which reaches
the region as the whole rank-1 array rather than as one cell's value.
`reference_node` recognises exactly that shape and generates
`smap_size(cell)` where the kernel wrote `smap_size`, so the body reads the
current cell's value without the kernel's source being rewritten. Every
other reference goes to `CWriter`, and a View that carries offsets is
subscripted by `arrayreference_node` as before.

Array-valued expressions
~~~~~~~~~~~~~~~~~~~~~~~~

Kokkos has no whole-array assignment. `a(:) = b(:) + 1.0` and `m(:,:) = 0.0`
therefore have to become explicit loops before they can be generated at all,
and `KokkosArrayExpression` in
`psyclone.psyir.backend.kokkos_array_expression` is what generates them. Its
public surface is two methods: `ranks(node)` reports the shape an expression
produces, one extent per dimension in the same string form a View's `extents`
take, and `lower(node, into=None)` generates the source that evaluates it.
`KokkosArrayExpressionMixin`, in
`psyclone.psyir.backend.kokkos_array_expression_mixin`, supplies the writer's
`assignment_node` and the handlers the nest needs, and is inherited ahead of
`CWriter` for the same reason `KokkosIntrinsicsMixin` is. The two are separate
modules because they grow for different reasons: the lowering gains code when
a new shape of array expression has to be generated, and the mixin when the
writer claims a new node kind or changes what it hands on to `CWriter`.

The destination decides which of two shapes is generated. Given an `into`,
the value is written straight into it and nothing is allocated:

.. code-block:: c++

    for (int _kae_i1 = 1; _kae_i1 <= (1 + nlayers - 1); _kae_i1++) {
      for (int _kae_i0 = 1; _kae_i0 <= (1 + 3 - 1); _kae_i0++) {
        m((_kae_i0 - 1), (_kae_i1 - 1)) = 0.0;
      }
    }

**The leading subscript is the innermost loop**, and that is not a matter of
taste. Every View this back-end declares is `Kokkos::LayoutLeft`, whose
leading dimension is the contiguous one, so the leading subscript is the one
that must vary fastest. A nest written the other way round computes exactly
the same answer while striding across memory on every step; no compiler
warns, no test fails, and nothing downstream can tell. The order is asserted
by a test of its own, because there is nowhere else it could be caught.

A destination need not carry a section for any of this. Fortran lets a name
stand for the whole of the array it was declared as, and `lhs_e = MATMUL(a,
b)` means what `lhs_e(:) = MATMUL(a, b)` means, so the two are lowered to the
same nest. The rule that makes them agree is applied to every array-valued
name in the statement rather than to the destination alone: a name carrying
no subscripts is given the nest's index in each of its dimensions, on the
right of the assignment as well as on the left. So `lhs_e = MATMUL(a, b) +
2.0 * c_e` becomes `lhs_e((_kae_i0 - 1)) = (_kae_r0 + (2.0 * c_e((_kae_i0 -
1))))`, because a View is not a value: assigning a `double` to one, or adding
one to a `double`, is an operation Kokkos does not define, and a translation
unit that asked for either would not compile. A name whose rank is not the
nest's is left as it stands, which only non-conforming source can produce.

Because each iteration reads the element it writes, `lhs_e = lhs_e +
MATMUL(a, b)` is lowered rather than refused by the rule below: it is the one
shape in which Fortran's evaluate-then-assign and the nest's element-by-
element order agree.

Given no `into`, the value has to go somewhere, and where it goes depends on
whether the expression is a section that could be named rather than copied. A
section taking its leading dimensions whole -- `jac(:,1,df)`, and any slice
whose `Kokkos::ALL` subscripts come first -- is contiguous in `LayoutLeft`,
so it is passed as a subview:

.. code-block:: c++

    auto _kae_sub0 = Kokkos::subview(jac, Kokkos::ALL, (1 - 1), (df - 1));

`auto` because the type `Kokkos::subview` returns is not one this back-end
can spell: it carries the layout the slice inherits, and a callee taking it
by template parameter is what keeps the write reaching the storage the caller
named. Nothing is allocated and no element is read.

Anything else is copied into a temporary, and the generated C++ says why:

.. code-block:: c++

    // jac(df,:,:) is copied rather than passed as a Kokkos::subview
    // because it is not contiguous in this View's LayoutLeft.
    for (int _kae_i1 = 1; ...) {
      for (int _kae_i0 = 1; ...) {
        _kae_tmp0((_kae_i0 - 1), (_kae_i1 - 1)) =
            jac((df - 1), (_kae_i0 - 1), (_kae_i1 - 1));
      }
    }

The comment is generated rather than left to the reader because a reader who
sees a nest where the neighbouring section got a subview has no way to tell
whether the difference was reasoned about. Three shapes reach it: a section
that skips the leading dimension, one that narrows any dimension it takes --
`Kokkos::subview` is generated with `Kokkos::ALL` and nothing narrower -- and
an expression that is not a section at all.

A temporary is described as a `KokkosScratch` and reported through
`temporaries`, rather than declared by this class, because that is what the
launch asks the run time for: an array allocated behind the launch's back
would not be in its `shmem_size` sum. `result` names where the last value
went. Names are generated `_kae_sub0`, `_kae_tmp0` and so on from one
counter, and `_kae_i0` upwards for the loop indices; the prefix is
`KokkosArrayExpression.PREFIX`.

An assignment reading the array it writes is refused. Fortran evaluates the
whole right-hand side before assigning any of it and a nest does not, so
`a(2:nlayers) = a(1:nlayers-1)` has no order of generated loops that means
what the Fortran meant. The refusal names the array and both subscripts.

On the LFRic path most array assignments never reach this class:
`LFRicKokkosTrans` applies `ArrayAssignment2LoopsTrans` first, which rewrites
them into PSyIR loops the back-end then generates as any other loop. What is
left for the lowering here is the shapes that transformation declines, and
the sections that are arguments rather than assignments.

Array-valued intrinsics
~~~~~~~~~~~~~~~~~~~~~~~

`MATMUL`, `DOT_PRODUCT`, `SUM`, `MINVAL`, `MAXVAL`, `TRANSPOSE` and `RESHAPE`
produce arrays, and Kokkos has no operator for any of them.
`KokkosArrayIntrinsics` in `psyclone.psyir.backend.kokkos_array_intrinsics`
generates them as loops. `KokkosArrayExpression` owns one instance per
region, because the names both classes generate are numbered from a counter
that a second region must start again from zero.

The class turns on one observation: *one element* of any of these is not an
array. An element of `MATMUL` or `DOT_PRODUCT` is a sum of products over the
index the operands share; an element of `SUM`, `MINVAL` or `MAXVAL` is that
same fold over the dimensions the `dim` argument does not keep; an element of
`TRANSPOSE` or `RESHAPE` is an element of the operand, at subscripts computed
from the ones asked for. So no array temporary is ever built. This is not
only economy. Scratch is asked of the launch by `shmem_size` *before* the
body is generated, so a temporary discovered while generating would arrive
too late to be counted; a nest that needs none sidesteps the ordering
problem entirely. It also composes for free: `MATMUL(TRANSPOSE(m), x)`
resolves the transpose into the subscripts the contraction reads with, and
`p(:) = MATMUL(a, b) + s * MATMUL(c, d)` is two accumulators in one nest.

Three methods carry the work. `shape` says what shape a call produces, in the
`(start, stop, step)` form `KokkosArrayExpression` uses for its own sections,
so a statement's nest is sized from the intrinsic where no section states
it. `hoist` rewrites a copy of the statement in place, replacing each
outermost handled call by the accumulator its loops leave the value in and
returning those loops for the caller to place ahead of the statement -- a
loop is not an expression in C++ and cannot appear where the operand did.
`element` resolves an index map and reads the array underneath it. A fold
starts from `Kokkos::reduction_identity<T>::sum()`, `::min()` or `::max()`
rather than from a literal, since a written `-DBL_MAX` would be right for one
C type and would have to be spelt again for each of the others.

A hoisted value re-enters the tree as a `Reference` whose symbol's *name* is
the generated C++. A `DataSymbol` does not validate its name and
`reference_node` writes the name it is given, so this is how text that is
already generated survives being visited again. Nothing else about the symbol
is read.

On the LFRic path an assignment holding one of these is kept from the section
lowering rather than passed through it. `ArrayAssignment2LoopsTrans` takes only
a right-hand side that is scalar-valued or elemental, and a contraction is
neither, so `exner_e(:) = MATMUL(m, rhs_e)` -- the shape `set_exner_code` is
written in -- reaches the writer as it stands and is generated here.
`LFRicKokkosIntrinsicMixin._written_as_a_nest` is what
`LFRicKokkosContractMixin._is_array_valued` asks, alongside the array
constructor it excludes for the same reason.

`space` and `consumed` are what `KokkosArrayExpression` asks before it sizes
a nest. An operand is not a section of the statement it appears in --
`matmul(m3(ik,:,:), p_e)` is rank 1 while its operand is rank 2 -- so a
caller taking the statement's shape from the first section it meets would
take the wrong one. `space` gives the shape of the first array-valued call
instead, and answers the empty tuple where every call in the statement is
scalar-valued, as in `a(:) = b(:) * dot_product(p, q)`, whose shape comes
from the sections beside it. `consumed` gives the `id()` of every access
under one of these calls, which is what the section walk steps over.

What is refused is the fold the generated shape cannot express: a `dim` that
is not a literal or that names no dimension the operand has, any argument
beside `dim` -- a `mask`, most often -- a reduction directly inside another,
and an operand that is neither a whole array nor a section of one.
`TRANSPOSE` is refused over anything that is not a matrix, and `RESHAPE` is
generated only from a rank-1 source with a literal shape, its element being
index arithmetic on the source's linear position.

Lowering order
~~~~~~~~~~~~~~

`LFRicKokkosTrans` replaces an `LFRicLoop` with a plain `Call`, and that is a
constraint on more than the loop. A domain node that resolves part of itself by
walking the tree for other domain nodes can only do so while they are still
there, so anything whose lowering depends on the loop has to be lowered before
the loop is replaced rather than after.

`LFRicHaloExchange` is the case that arises. It does not know its own depth: it
computes one from the accesses that read the field it exchanges, and those are
LFRic kernel arguments on the loop. Lowered after the replacement it finds no
reader at all and PSyclone raises `InternalError` from
`_compute_halo_read_info`. `LFRicKokkosTrans._lower_halo_exchanges` therefore
lowers every exchange in the invoke first, which is the order whole-container
lowering would have used anyway since an exchange precedes the loop it feeds.

A transformation that introduces a new such dependency has the same obligation.
The symptom is an `InternalError` from a node the transformation never touched,
which is easy to read as a bug in that node rather than as an ordering
constraint on this one.

SIR back-end
++++++++++++

The SIR back-end is limited in a number of ways:

- only Fortran code containing 3 dimensional directly addressed
  arrays, with simple stencil accesses, iterated with triply nested
  loops is supported. Imperfectly nested loops, doubly nested loops,
  etc will cause a ``VisitorError`` exception.
- anything other than real arrays (integer, logical etc.) will cause
  incorrect SIR code to be produced (see issue #468).
- calls are not supported (and will cause a VisitorError exception).
- loop bounds are not analysed so it is not possible to add in offset
  and loop ordering for the vertical. This also means that the ordering
  of loops (lat/lon/levels) is currently assumed.
- Fortran literals such as `0.0d0` are output directly in the
  generated code (but this could also be a frontend issue).
- the only unary operator currently supported is '-'.

The current implementation also outputs text rather than running Dawn
directly. This text needs to be pasted into another script in order to
run Dawn, see :ref:`nemo-eg4-sir` the NEMO API example 4.

Currently there is no way to tell PSyclone to output SIR. Outputting
SIR is achieved by writing a script which creates an SIRWriter and
outputs the SIR (for kernels) from the PSyIR. Whilst the main
'psyclone' program could have a '-backend' option added it is not
clear this would be useful here as it is expected that the SIR will be
output only for certain parts of the PSyIR and (an)other back-end(s)
used for the rest. It is not yet clear how best to do this - perhaps
mark regions using a transformation.

It is unlikely that the SIR will be able to accept full NEMO code due
to its complexities (hence the comment about using different
back-ends in the previous paragraph). Therefore the approach that will
be taken is to use PSyclone to transform NEMO to make regions that
conform to the SIR constraints and to make these as large as
possible. Once this is done then PSyclone will be used to generate and
optimise the code that the SIR is not able to optimise and will let
the SIR generate code for the bits that it is able to do. This
approach seems a robust one but would require interface code between
the Dawn generated cuda (or other) code and the PSyclone generated
Fortran. In theory PSyclone could translate the remaining code to C
but this would require no codeblocks in the PSyIR when parsing NEMO
(which is a difficult thing to achieve), or interface code between
codeblocks and the rest of the PSyIR.

As suggested by the Dawn developers, PSyIR local scalar variables are
translated into temporary SIR fields (which are 3D arrays by
default). The reason for doing this is that it is easy to specify
variables in the SIR this way (whereas I did not manage to get scalar
declarations working) and Dawn optimises a temporary field, reducing
it to its required dimensionality (so PSyIR local scalar variables are
output as scalars by the Dawn back end even though they are specified
as fields). A limitation of the current translation from PSyIR to SIR
is that all PSyIR scalars are assumed to be local and all PSyIR arrays
are assumed to be global, which may not be the case. This limitation
is captured in issue #521.


.. _uplifting-lowering:

Uplifting and lowering mechanism
================================

The additional complexity of the PSy-layer comes from the fact that it
contains multiple domain-specific concepts and parallel concepts that are not
part of the target languages. Instead of dealing with these concepts in the
visitors we require that any domain-specific concept introduced on top
of the core PSyIR constructs contains the logic to lower this concept
into language level constructs. The reasons for choosing a method instead
of a visitor for this transformation are:

- Each concept introduced by the API-developer will need lowering instructions,
  and this is better implied by an abstract class in the node that needs to
  be filled.
- The lowering is done in-place. A method fits better with modifying the AST
  in-place because it can use and modify the nodes private fields.

The current proposed solution is to create a 2-phase generation workflow where
a domain-specific PSyIR is first lowered to a language-level version of the
PSyIR using the ``lower_to_language_level`` node method and then processed by
the Visitor to generate the target language.
The language-level PSyIR is still the same IR but restricted to the subset
of Nodes that have a direct translation into target language concepts.

.. image:: 2level_psyir.png
