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
  literals, references, if-blocks, loops, unary and binary operations, a
  subset of intrinsics, and directives. It has no handler for a `Routine`,
  so it generates the statements and expressions of a body rather than a
  whole kernel. A loop's continuation test follows the sign of its step
  where that sign is visible in the tree, so a Fortran countdown such as
  `do k = n, 1, -1` becomes `for(k=n; k>=1; k+=-1)`; a step that is a
  runtime value is taken to be positive. Which intrinsics that subset
  contains, and why four of them depend on their argument's type, is set out
  in the C back-end section below.
- `KokkosWriter()` in `psyclone.psyir.backend.kokkos` which extends
  `CWriter` to generate a complete C++/Kokkos translation unit. It is not
  called on a PSyIR node: it is called on a `KokkosRegion` holding the
  region's name, its scalar arguments, its unmanaged Views, its
  `kind_types`, its `scratch`, the `parallel_loops` and `team_size`
  that select and size the hierarchical launch, and the `cell_position`
  naming the one formal it declares rather than takes, and it visits the
  loop body through `CWriter`. Two parts of the back-end live beside it because they
  grow as regions are captured while the writer's own job does not:
  `psyclone.psyir.backend.kokkos_launch` renders the launch shapes, one
  function per shape, and `KokkosIntrinsicsMixin` in
  `psyclone.psyir.backend.kokkos_intrinsics_mixin` holds the intrinsics,
  inherited ahead of `CWriter` so its handlers are found first and fall
  through to `CWriter`'s. `KokkosConstant` in
  `psyclone.psyir.backend.kokkos_constant` is a third, described below. The
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
`_is_real_argument` chooses between that and the table entry. Choosing wrongly
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
#1941. `LFRicKokkosCallMixin._kind_assertions` filters a logical kind out of
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
their argument's type: this writer reuses `_is_real_argument` and generates
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
counterparts do.

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

Three launch shapes
~~~~~~~~~~~~~~~~~~~

The back-end generates one of three launches. A region naming any
`parallel_loops` takes the hierarchical launch; otherwise one describing any
`scratch` takes the flat team launch; otherwise the range launch. The
selection is a chain rather than a separate selector field, so nothing can
disagree with it, and each shape is reached only by a region carrying the
field it selects on -- a region built before either field existed generates
exactly the source it generated then. The launches themselves are rendered by
`psyclone.psyir.backend.kokkos_launch`, one function per shape, and
`KokkosWriter` chooses among them.

A region with no scratch launches over `Kokkos::RangePolicy<>(0, ncells)`
with a `KOKKOS_LAMBDA(const int cell)`. This is the shape every region had
before scratch existed and it is generated unchanged, because a kernel with
no local arrays has nothing to place.

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

Which loops may be spread is a dependence judgement, and it is made by
`LFRicKokkosTrans` rather than here: the back-end renders the loops it is
given and does not judge them. It does check what it is given, since none of
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
`KokkosConstant` lives in its own module rather than beside the other
descriptions because `kokkos.py` is at the size limit this project sets, and
the descriptions there are to move out of it wholesale.

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

Extents
~~~~~~~

A View's and a scratch array's `extents` are strings written into the
generated C++ verbatim, one per dimension. An extent may be an integer
expression over named sizes: `nlayers`, `4` and `(nlayers + 1)` are all
accepted, built from identifiers, decimal literals, `+`, `-`, `*` and
balanced parentheses. `is_extent` is the predicate, and `extent_names`
reports the identifiers an extent is sized from; both are module-level
functions of `psyclone.psyir.backend.kokkos`, because the transformation
that builds a region has to apply the same grammar before it gets here.

Division is refused rather than merely unsupported. Fortran and C++ can
disagree about the rounding of an integer division, and an extent is one of
the few places where that disagreement would produce a wrongly sized
allocation instead of a compile error.

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
