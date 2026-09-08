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

"""Which of a captured body's scalars belong to one member of a team.

Every member of a Kokkos team executes the body of a hierarchical launch, so
a scalar declared at the top of that body is already one object per member and
not one the team shares. What a ``TeamVectorRange`` inside the body changes is
which iterations of a loop each member runs.

A scalar the spread loop writes therefore holds, when the loop ends, the value
of the last iteration *that member* happened to run -- and in no member the
value of the loop's last iteration. A statement after the loop that reads it
is right on at most one member and reads a value from the middle of the loop
on the rest. Under ``Kokkos::AUTO`` on the OpenMP back-end a team has one
member, a ``TeamVectorRange`` is a serial loop, and the value is the one the
kernel author meant: nothing on such a host, checksums included, distinguishes
a body that would be wrong with four members from one that would not.

Declaring the scalar inside the ``TeamVectorRange`` lambda is what makes the
difference visible. Each iteration then gets its own, which is what a working
value wants, and the statement after the loop no longer compiles, which is
what a value read afterwards deserves. Deciding which scalars may be declared
there is this module's job.

The rule this module applies is the one an OpenMP ``private`` clause applies,
decided here rather than left to the reader:

* A scalar the loop writes before anything in the loop reads it, and which
  nothing reads after the loop without writing it first, is **loop-private**:
  it is a working value with no life beyond the iteration, and is declared
  inside the lambda so that each member owns one.
* A scalar the loop writes that fails either half of that is **shared**, and
  the region is refused. It is refused rather than generated with the
  declaration left outside because the two ways of continuing are both wrong:
  the value read afterwards belongs to the last iteration, which after a
  spread no member holds, and moving the declaration in would delete the
  error the C++ compiler would otherwise raise.
* A scalar the loop only reads is neither, and is left where it is. Every
  member computed it redundantly at team level, so every member's copy holds
  the same value.

The second half of the rule is liveness and not "used nowhere else". A
kernel's counters and its one or two general-purpose temporaries are reused
all through a body -- `df` drives a serial loop at team level and then a
nested loop inside a spread one; `t1` holds a difference at team level and a
sum inside a level -- and every one of those uses writes before it reads.
Refusing those loops would cost a region for a name, which is why the
declaration a spread loop takes inside its lambda *shadows* the region-scope
one where the body needs both. The region-scope declaration is dropped only
when nothing outside the spread loops mentions the name at all.

Nothing here decides which loops are spread; that is the region's
:py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.parallel_loops`, chosen
by the transformation. This module is asked only what the choice implies for
the scalars, and says so before a line of C++ is generated.
"""

from psyclone.psyir.nodes import (
    ArrayReference, Assignment, IfBlock, Loop, Reference, StructureReference,
    WhileLoop)


def _is_scalar_target(node) -> bool:
    """Return whether an assignment writes the whole of a scalar.

    An element of an array and a component of a structure are writes to
    something the region holds elsewhere -- a View, or scratch -- and never to
    a scalar this module can move.

    :param node: the left-hand side of an assignment.
    :type node: :py:class:`psyclone.psyir.nodes.Node`

    :returns: whether the assignment's target is a whole scalar.
    """
    return (isinstance(node, Reference)
            and not isinstance(node, (ArrayReference, StructureReference)))


def _reads(node) -> set:
    """Return the name of every symbol an expression reads.

    An array's own name is included along with the symbols in its subscripts.
    It costs nothing: an array is never a candidate to be made private, and
    excluding it would need this function to know which names are arrays.

    :param node: the expression to read.
    :type node: :py:class:`psyclone.psyir.nodes.Node`

    :returns: the names read.
    """
    return {reference.symbol.name for reference in node.walk(Reference)}


def _writes(statements) -> set:
    """Return the name of every scalar the statements assign, anywhere.

    Every assignment and every loop counter, whatever it is nested inside and
    whether or not it runs: this is what the loop *may* write, and so what has
    to be private if the loop is to be spread.

    :param statements: the statements to walk.
    :type statements: List[:py:class:`psyclone.psyir.nodes.Node`]

    :returns: the names written.
    """
    names = set()
    for statement in statements:
        for assignment in statement.walk(Assignment):
            if _is_scalar_target(assignment.lhs):
                names.add(assignment.lhs.symbol.name)
        for loop in statement.walk(Loop):
            names.add(loop.variable.name)
    return names


def _flow(statements) -> tuple:
    """Return what the statements definitely write, and what they read first.

    Walked in order, carrying the set of names already written. A name read
    while it is not in that set is *upward exposed*: its value comes from
    before these statements, so an iteration of a loop containing them depends
    on what an earlier iteration left behind.

    The two constructs that make this more than a scan are here:

    * a loop's body may run no times, so what it writes is not definite, but
      what it reads first is still exposed. Its counter, on the other hand, is
      written by the initialisation, which always runs;
    * an ``if`` writes a name definitely only when every branch writes it, so
      one without an ``else`` contributes no definite write at all.

    Anything else -- a ``while``, a call -- contributes its references as
    reads and no writes, which is the conservative answer in both directions:
    a name it really writes is then merely not made private.

    :param statements: the statements to walk, in order.
    :type statements: List[:py:class:`psyclone.psyir.nodes.Node`]

    :returns: the names definitely written, and the names read before this
        code writes them.
    :rtype: Tuple[set, set]
    """
    written, exposed = set(), set()
    for statement in statements:
        if isinstance(statement, Assignment):
            exposed |= _reads(statement.rhs) - written
            if _is_scalar_target(statement.lhs):
                written.add(statement.lhs.symbol.name)
            else:
                # The subscripts of an element write are read, the array is
                # not a scalar, and neither makes the target written.
                exposed |= _reads(statement.lhs) - written - {
                    statement.lhs.symbol.name}
        elif isinstance(statement, Loop):
            exposed |= _reads(statement.start_expr) - written
            exposed |= _reads(statement.stop_expr) - written
            exposed |= _reads(statement.step_expr) - written
            written.add(statement.variable.name)
            exposed |= _flow(statement.loop_body.children)[1] - written
        elif isinstance(statement, IfBlock):
            exposed |= _reads(statement.condition) - written
            if_written, if_exposed = _flow(statement.if_body.children)
            else_written, else_exposed = (
                (set(), set()) if statement.else_body is None
                else _flow(statement.else_body.children))
            exposed |= (if_exposed | else_exposed) - written
            written |= if_written & else_written
        else:
            exposed |= _reads(statement) - written
    return written, exposed


def _live_after(loop, schedule) -> set:
    """Return the names read after a loop before anything writes them.

    Everything that runs once the loop is over, walked outwards from the loop
    to the region's body: the statements after it, then the statements after
    whatever contains it, and so on. A name upward exposed in any of that is
    live at the loop's exit, and the value it is live with is the one the
    loop's last iteration left in it -- which, once the iterations are shared
    out, no member holds.

    A serial loop on the way out repeats, so the statements *before* the loop
    in its body run after it as well and are counted too.

    :param loop: the loop being spread over the team.
    :type loop: :py:class:`psyclone.psyir.nodes.Loop`
    :param schedule: the region's body, where the walk stops.
    :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`

    :returns: the names live at the loop's exit.
    """
    names, node = set(), loop
    while True:
        siblings = list(node.parent.children)
        index = [
            position for position, sibling in enumerate(siblings)
            if sibling is node][0]
        names |= _flow(siblings[index + 1:])[1]
        if node.parent is schedule:
            return names
        owner = node.parent.parent
        if isinstance(owner, (Loop, WhileLoop)):
            names |= _flow(siblings)[1]
            if isinstance(owner, WhileLoop):
                names |= _reads(owner.condition)
        node = owner


def _inside(node, bodies) -> bool:
    """Return whether a node is anywhere within one of the given bodies.

    By identity and through every ancestor, so that a reference deep inside a
    nest counts as being in the loop that encloses the nest.

    :param node: the node to place.
    :type node: :py:class:`psyclone.psyir.nodes.Node`
    :param bodies: the loop bodies to place it in.
    :type bodies: List[:py:class:`psyclone.psyir.nodes.Schedule`]

    :returns: whether the node lies within any of them.
    """
    while node is not None:
        if any(node is body for body in bodies):
            return True
        node = node.parent
    return False


def _names_outside(region) -> set:
    """Return every name the region uses outside its spread loops.

    A spread loop's own bounds are outside it: they are evaluated once, at
    team level, before the lambda is entered. A name read there and written in
    the body is therefore shared however the declaration is placed.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the names used outside the spread loops.
    """
    bodies = [loop.loop_body for loop in region.parallel_loops]
    names = set()
    for reference in region.schedule.walk(Reference):
        if not _inside(reference, bodies):
            names.add(reference.symbol.name)
    for loop in region.schedule.walk(Loop):
        if not _inside(loop, bodies):
            names.add(loop.variable.name)
    return names


def _refuse(region, loop, name, reason):
    """Return the error a scalar that cannot be made private is refused with.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param loop: the loop the region asked to spread over the team.
    :type loop: :py:class:`psyclone.psyir.nodes.Loop`
    :param name: the scalar that cannot be the member's own.
    :type name: str
    :param reason: what disqualifies it, as a clause.
    :type reason: str

    :returns: the error to raise.
    :rtype: ValueError
    """
    return ValueError(
        f"{region.name}: the loop over '{loop.variable.name}' cannot be "
        f"spread over the team, because it writes '{name}', which {reason}. "
        f"Spreading the loop divides its iterations between the members of "
        f"the team, and each member would need its own '{name}'; a scalar "
        f"this one cannot be given to.")


def team_private_scalars(region) -> tuple:
    """Return the scalars each spread loop declares inside its lambda.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: for each loop in
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.parallel_loops`,
        keyed by ``id``, the symbols to declare at the top of its lambda, in
        the order the symbol table holds them; and the names the region must
        no longer declare at its own scope, being those the private
        declarations replace rather than shadow.
    :rtype: Tuple[Dict[int,
        Tuple[:py:class:`psyclone.psyir.symbols.DataSymbol`]], Set[str]]

    :raises ValueError: if a spread loop writes a scalar that cannot be the
        member's own.
    """
    if not region.parallel_loops:
        return {}, set()
    scratch_names = {item.name for item in region.scratch}
    candidates = {
        symbol.name: symbol
        for symbol in region.schedule.symbol_table.automatic_datasymbols
        if not symbol.is_array and symbol.name not in scratch_names}
    outside = _names_outside(region)

    per_loop = {}
    for loop in region.parallel_loops:
        body = loop.loop_body.children
        exposed = _flow(body)[1]
        live = _live_after(loop, region.schedule)
        # Only what this loop writes. A name it merely reads belongs to
        # whatever wrote it -- the team body, or a loop this one is nested
        # inside -- and declaring it here would shadow that with an
        # uninitialised object of the same name. The loop's own counter is
        # the lambda's parameter: the members hold it separately already and
        # declaring it again would not compile.
        for name in sorted(_writes(body) - {loop.variable.name}):
            if name not in candidates:
                raise _refuse(
                    region, loop, name,
                    "is not a local of the kernel but part of its interface")
            if name in exposed:
                raise _refuse(
                    region, loop, name,
                    "an iteration reads before it writes")
            if name in live:
                raise _refuse(
                    region, loop, name,
                    "the region reads after the loop without writing it "
                    "first")
        per_loop[id(loop)] = _writes(body) - {loop.variable.name}

    ordered = region.schedule.symbol_table.automatic_datasymbols
    return {
        key: tuple(symbol for symbol in ordered if symbol.name in names)
        for key, names in per_loop.items()
    }, _replaced(region, per_loop, outside)


def _replaced(region, per_loop, outside) -> set:
    """Return the names the region no longer declares at its own scope.

    A name is replaced rather than shadowed when the private declarations
    account for every use of it: nothing outside the spread loops names it,
    and every spread loop that names it declares its own. A loop that only
    reads the name is not one of those, so a name written in one spread loop
    and read in another keeps the declaration they both refer to.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param per_loop: the private names of each spread loop, keyed by the
        loop's ``id``.
    :type per_loop: Dict[int, set]
    :param outside: the names the region uses outside its spread loops.
    :type outside: set

    :returns: the names to drop from the region's own declarations.
    """
    names = set()
    for name in set().union(*per_loop.values()) - outside:
        if all(name in per_loop[id(loop)]
               for loop in region.parallel_loops
               if name in _reads(loop.loop_body) | _writes(
                   loop.loop_body.children)):
            names.add(name)
    return names
