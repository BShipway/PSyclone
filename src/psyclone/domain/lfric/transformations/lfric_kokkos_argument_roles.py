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

"""Record what each actual of a kernel call is, for the region's Views.

Split from
:py:mod:`~psyclone.domain.lfric.transformations.lfric_kokkos_argument_mixin`
when that module reached pylint's thousand-line limit. The class is the one
thing there that is not a method of the mixin: it is an
:py:class:`~psyclone.domain.lfric.ArgOrdering` that
:py:meth:`LFRicKokkosArgumentMixin._argument_lists` builds the call's
argument list with, and it answers one question the mixin then reads off it
-- which role each stretch of that list carries.
"""

from psyclone.domain.lfric import KernCallArgList


class LFRicKokkosArgumentRoles(KernCallArgList):
    """A ``KernCallArgList`` that remembers what each actual is.

    Every array a region takes has to be placed somewhere the device can
    reach it, and where that is depends on what the array *is*: LFRic
    allocates field data in a space a device shares, and allocates a dofmap,
    a basis table, a quadrature weight, a map or an operator's local stencil
    in one it does not. The generated C++ cannot tell them apart -- they all
    arrive as pointers -- so the answer has to be taken here, where the
    kernel's metadata still says what each argument means, and carried down
    on the View as its
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosView.role`.

    ``ArgOrdering`` already visits each argument through a callback named for
    its kind, so a handful of callbacks are the whole question. Each records
    the stretch of the argument list its call added -- one entry for a field,
    one per component for a field vector, the count and the stencil for an
    operator -- against the role that stretch carries.

    Three kinds are named here rather than left to be read off the access the
    kernel declares. A *field* is named because its storage is in a space the
    device shares. An *LMA operator* is named for the same reason and carries
    the same role: ``lfric_core`` claims a field's ``data`` and an operator's
    ``local_stencil`` from one registry, on one default and behind one switch,
    so an operator's stencil is in the space a field's data is in and a region
    reads and writes it where it lies. Read off the access alone it would look
    like a dofmap -- read-only, so immutable, so cacheable -- and an apply
    would go on reading the copy the first call took while a later assembly
    wrote new values to the storage behind it. That is not a compile error and
    not a crash; it is a solver that stops converging, which is what the
    whole-model gate saw before the role was taken from the metadata rather
    than from the access. A *columnwise* operator is the exception: its matrix
    and its dofmaps are ordinary allocatables, nothing may assume they are
    device-reachable, and it keeps ``readwrite``.

    A *basis* or *differential basis* table and a rule's *quadrature weights*
    are named for the opposite reason. They are read-only for the call, and by
    the access alone they look exactly like a dofmap, but the PSy layer
    allocates them at the head of an invoke, fills them from the rule and
    deallocates them at its foot. The storage is therefore recycled between
    invokes: an address that held one space's table holds another's a moment
    later, so nothing keyed by the address stays good. They are ``transient``,
    which is the role that says read-only for the call and not cacheable
    across calls.

    :param kern: the kernel whose call is being built.
    :type kern: :py:class:`psyclone.domain.lfric.LFRicKern`

    """
    def __init__(self, kern):
        super().__init__(kern)
        #: The role of each position in the argument list a callback named,
        #: keyed by position. A position no callback here named is absent,
        #: and is placed by what the kernel reads or writes it as.
        self.roles = {}

    def _record(self, before, role):
        """Record everything the call just added under one role.

        :param int before: the length of the argument list before the call.
        :param str role: the role the added arguments carry.

        """
        self.roles.update(
            (position, role)
            for position in range(before, len(self._psyir_arglist)))

    def field(self, arg, var_accesses=None):
        """Add a field, and record where it landed.

        :param arg: the field to add.
        :type arg: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().field(arg, var_accesses)
        self._record(before, "field")

    def field_vector(self, argvect, var_accesses=None):
        """Add a field vector, and record where its components landed.

        :param argvect: the field vector to add.
        :type argvect: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().field_vector(argvect, var_accesses)
        self._record(before, "field")

    def operator(self, arg, var_accesses=None):
        """Add an LMA operator, and record where its stencil landed.

        The ``field`` role, because an LMA operator's local stencil is in the
        space a field's data is in: one registry claims both, on one default.
        A region therefore works on the caller's storage and nothing is
        copied around the call. The count the call also adds is a scalar,
        which carries no View and so no role.

        :param arg: the operator to add.
        :type arg: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().operator(arg, var_accesses)
        self._record(before, "field")

    def cma_operator(self, arg, var_accesses=None):
        """Add a columnwise operator, and record it as staged per call.

        Not the ``field`` role an LMA operator takes.
        ``columnwise_operator_mod`` allocates the banded matrix and its
        dofmaps the ordinary way, so none of them is device-reachable and
        each has to be staged. The transformation refuses a CMA kernel
        outright today; this keeps the answer right for the day it does not.

        :param arg: the operator to add.
        :type arg: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().cma_operator(arg, var_accesses)
        self._record(before, "readwrite")

    def basis(self, function_space, var_accesses=None):
        """Add a basis table, and record it as living only for the invoke.

        :param function_space: the space the table is for.
        :type function_space: \
            :py:class:`psyclone.domain.lfric.FunctionSpace`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().basis(function_space, var_accesses)
        self._record(before, "transient")

    def diff_basis(self, function_space, var_accesses=None):
        """Add a differential basis table, and record it the same way.

        :param function_space: the space the table is for.
        :type function_space: \
            :py:class:`psyclone.domain.lfric.FunctionSpace`
        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().diff_basis(function_space, var_accesses)
        self._record(before, "transient")

    def quad_rule(self, var_accesses=None):
        """Add the quadrature rule, and record its weights the same way.

        :param var_accesses: optional store for variable accesses.
        :type var_accesses: Optional[
            :py:class:`psyclone.core.VariablesAccessMap`]

        """
        before = len(self._psyir_arglist)
        super().quad_rule(var_accesses)
        self._record(before, "transient")


__all__ = ["LFRicKokkosArgumentRoles"]
