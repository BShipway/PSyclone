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

"""Give every name an inlined callee reads one origin the call site can reach.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``classmethod`` or a ``staticmethod``. It carries the import rules that
:py:class:`~psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin.\
LFRicKokkosInlineMixin` used to, because that module had passed pylint's
thousand-line limit, and these rules are one subject of their own: not
*whether* a call is inlined, but whether each name the callee's body reads
still means, at the call site, what it meant in the callee's module.

:py:class:`~psyclone.psyir.transformations.InlineTrans` merges the callee's
symbol table into the call site's and refuses where the two disagree about a
name. Every rule here settles such a disagreement the way Fortran already
has -- the name is an import of one module in both scopes -- and settles it
only towards what a module on the search path proves, so a name no module
settles is still refused, in ``InlineTrans``'s own words.
"""

from psyclone.psyir.nodes import Call, Container, Reference, Routine
from psyclone.psyir.symbols import (
    ContainerSymbol, DataSymbol, ImportInterface, Symbol, SymbolError)


class LFRicKokkosImportMixin:
    """Settle the names an inlined callee and its call site share.

    Each method is asked by the inlining loop of
    :py:class:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_bound_mixin.LFRicKokkosBoundMixin` or by the sibling absorption
    of :py:class:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin`, just before a call is
    inlined, and rewrites only copies: the call site's own copy of the file
    and the callee brought into it. The module the ``ModuleManager`` caches is
    left as its file declares it. The one rule that has to touch that module
    -- :py:meth:`_carry_module_names`, since the move reads the callee from
    it -- puts it back as it found it the moment the move is over.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    @classmethod
    def _carry_module_names(cls, call):
        """Give ``call``'s callee its own declaration of its module's names.

        :py:class:`~psyclone.domain.common.transformations.\
KernelModuleInlineTrans` refuses a routine that reads anything declared
        beside it in its own module, because moving the routine would leave
        the name behind: it "accesses data from its outer scope". LFRic's
        Held--Suarez helpers read ``KF``, ``KA`` and five more ``parameter``
        values of ``held_suarez_forcings_mod``, its panel transforms read
        ``PANEL_ROT_MATRIX`` of ``coord_transform_mod``, and each is refused
        on that ground alone. Two kinds of name can be made to travel, and
        each is given the declaration Fortran would accept for it anywhere:

        * a name its module makes **public** is imported from that module,
          as a ``use <module>, only : <name>`` of the routine's own. Every
          scope that may name it can import it, the PSy layer included, so
          it then reaches the region the way any imported name does: a
          constant by its value and a variable as a by-value argument
          (:py:class:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_constants_mixin.LFRicKokkosConstantsMixin`);
        * a **private** ``parameter`` is declared again, as a ``parameter``
          of the routine's own with the same type and value. Shadowing a
          module's constant with an equal local one changes nothing the
          routine computes, and a constant is a value the region carries
          rather than a name it has to import. Where its value names another
          constant of the module -- ``KA = KF/40.0_r_def`` -- that one is
          carried first, by the same two rules.

        A private *variable* is neither: nothing outside its module may name
        it, so nothing can carry it, and the move is left to refuse it in its
        own words. That is LFRic's ``sci_chi_transform_mod`` state as the file
        declares it, which the ``lfric_core`` edit this capability was scoped
        with makes ``public, protected``.

        **The module rewritten is the cached one, and it is put back.**
        ``KernelModuleInlineTrans`` reads the callee through
        :py:meth:`~psyclone.psyir.nodes.Call.get_callees`, which answers with
        the Container the ``ModuleManager`` parsed, so the declarations have
        to be added there to be seen at all. The move takes a copy, and the
        copy keeps them; the caller hands what this returns to
        :py:meth:`_restore_module_names` as soon as the move is over, whether
        it succeeded or not, and the module then reads as its file does.

        It is asked only about a callee in another Container: one already in
        the call's Container is a routine of the kernel's own module, which
        is not moved, and whose names the constants mixin reads where they
        stand.

        :param call: the call whose callee is about to be moved.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: what was added, one entry per name, for
            :py:meth:`_restore_module_names` to undo.
        :rtype: List[Tuple[:py:class:`psyclone.psyir.nodes.Routine`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            Optional[:py:class:`psyclone.psyir.symbols.ContainerSymbol`]]]
        """
        try:
            callees = call.get_callees()
        except Exception:                        # pylint: disable=W0703
            # A callee PSyclone cannot resolve has nothing to prepare, and
            # the caller reports the name it could not reach.
            return []
        carried = []
        for callee in callees:
            container = callee.ancestor(Container)
            for signature in callee.reference_accesses().all_signatures:
                cls._carry_module_name(callee, container, signature.var_name,
                                       carried)
        return carried

    @classmethod
    def _carry_module_name(cls, routine, container, name, carried):
        """Give ``routine`` its own declaration of one name of its module.

        The rules are :py:meth:`_carry_module_names`'s. A name the routine
        already declares, or one ``container`` does not declare itself, is
        left alone, and so is anything but data: a sibling procedure is
        absorbed rather than carried.

        :param routine: the callee to give the declaration to.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param container: the callee's own module.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param str name: the name the routine reads.
        :param carried: what has been added so far, appended to.
        :type carried: List[Tuple[:py:class:`psyclone.psyir.nodes.Routine`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            Optional[:py:class:`psyclone.psyir.symbols.ContainerSymbol`]]]
        """
        table = routine.symbol_table
        if name in table:
            return
        symbol = container.symbol_table.lookup(
            name, scope_limit=container, otherwise=None)
        if (not isinstance(symbol, DataSymbol) or symbol.is_import
                or symbol.is_unresolved):
            return
        added = None
        if symbol.visibility == Symbol.Visibility.PUBLIC:
            if container.name in table:
                source = table.lookup(container.name)
            else:
                source = added = ContainerSymbol(container.name)
                table.add(source)
            local = symbol.copy()
            local.interface = ImportInterface(source)
        elif symbol.is_constant:
            for reference in symbol.initial_value.walk(Reference):
                cls._carry_module_name(routine, container,
                                       reference.symbol.name, carried)
            local = symbol.copy()
            for reference in local.initial_value.walk(Reference):
                reference.symbol = table.lookup(reference.symbol.name)
        else:
            return
        table.add(local)
        for reference in routine.walk(Reference):
            if reference.symbol is symbol:
                reference.symbol = local
        carried.append((routine, symbol, local, added))

    @staticmethod
    def _restore_module_names(carried):
        """Undo :py:meth:`_carry_module_names`, last name first.

        Each routine's references are pointed back at its module's symbol,
        the declaration it was given is removed, and a ``use`` of its own
        module added for the purpose goes with the last name it imported.
        Last first, because a constant carried for another's value was
        carried before it.

        :param carried: what :py:meth:`_carry_module_names` returned.
        :type carried: List[Tuple[:py:class:`psyclone.psyir.nodes.Routine`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.symbols.DataSymbol`,
            Optional[:py:class:`psyclone.psyir.symbols.ContainerSymbol`]]]
        """
        for routine, symbol, local, added in reversed(carried):
            for reference in routine.walk(Reference):
                if reference.symbol is local:
                    reference.symbol = symbol
            table = routine.symbol_table
            table.remove(local)
            if added is not None:
                table.remove(added)

    @classmethod
    def _localise_imports(cls, routine):
        """Move into ``routine``'s own table the imports it reads from above.

        :py:class:`~psyclone.psyir.transformations.InlineTrans` copies the
        routine it is inlining, and a
        :py:class:`~psyclone.psyir.nodes.Routine` copied on its own is
        detached from the Container whose ``use`` statements named half of
        what it reads. Every such name arrives at the call site *unresolved*,
        and the second call to the same routine is then refused for a clash
        between two unresolved symbols of that name -- a refusal about the
        copy rather than about the code, and one that would let a routine be
        absorbed once and not twice.

        Giving the routine its own import of each name it reads is what the
        copy then carries: ``panel_neighbour``, which reads ``W``, ``S``,
        ``E`` and ``N`` from its module's ``use reference_element_mod``, is
        rewritten to read them from a ``use`` of its own -- the same Fortran,
        and an import the call site's own import of the name agrees with.

        :param routine: the routine about to be inlined into its own module.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        """
        table = routine.symbol_table
        for signature in routine.reference_accesses().all_signatures:
            name = signature.var_name
            if name in table:
                continue
            symbol = table.lookup(name, otherwise=None)
            if symbol is None or not symbol.is_import:
                continue
            cls._localise_import(routine, symbol)

    @staticmethod
    def _localise_import(routine, symbol):
        """Give ``routine`` its own import of a name it reads from above.

        A copy of the symbol is added rather than the symbol itself, because
        the table it is read from is the one every other routine of that
        scope reads the name through. The references in ``routine`` are
        re-pointed at the copy.

        :param routine: the routine to give the import to.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the imported symbol of an enclosing scope.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`
        """
        table = routine.symbol_table
        source = symbol.interface.container_symbol
        if source.name in table:
            local_source = table.lookup(source.name)
        else:
            local_source = ContainerSymbol(source.name)
            table.add(local_source)
        local = symbol.copy()
        local.interface = ImportInterface(
            local_source, orig_name=symbol.interface.orig_name)
        table.add(local)
        for reference in routine.walk(Reference):
            if reference.symbol is symbol:
                reference.symbol = local

    @classmethod
    def _agree_on_imports(cls, call):
        """Give the two scopes of ``call`` one origin for each shared name.

        ``InlineTrans`` merges the callee's table into the call site's, and
        refuses a name the two scopes hold differently:

            A symbol named 's' is present in both tables but is unresolved in
            one. That scope does not contain a direct wildcard import from the
            module 'reference_element_mod' from which it is imported in the
            other scope.

        The message names a wildcard because that is the import PSyclone
        would accept as the missing origin, not because one was written: any
        name one scope holds as an import and the other holds unresolved is
        refused this way. That is LFRic's ``W``, ``S``, ``E`` and ``N``, the
        face indices of ``reference_element_mod``: the FFSL flux kernels name
        them in a ``use ..., only`` of their own and their helpers -- siblings
        of the same module -- read them through the module's own ``use``, so
        copied away from it the helper's names arrive unresolved.

        The two are the same name of the same module, and PSyclone can say
        so rather than this being asserted: the resolved side names the
        module, and where the scope holds the name itself
        :py:meth:`~psyclone.psyir.symbols.SymbolTable.resolve_imports` reads
        that module through the ``ModuleManager``'s search path -- the one
        the build and the survey already set with ``-d``. Where the scope
        holds nothing of the name and reads it from the Container it sits in,
        that declaration is already in the tree and is copied down into the
        routine instead. Either way both sides are imports of one name from
        one module, which the merge accepts.

        Only a name the two scopes share is resolved, and only towards the
        module the other scope proves it comes from: resolving every wildcard
        import wholesale would pull the whole of each module into the table
        and answer a question nothing asked. What is written is a
        ``use <module>, only : <name>``, the name being the whole of what the
        region needs. Where PSyclone cannot resolve it -- a module the search
        path does not reach, or one that does not publish the name -- nothing
        is changed and the refusal stands as it did, which is the reader's
        evidence that the module was not readable rather than that the
        rewrite was not tried.

        The tables rewritten are the call site's and the callee's *as the call
        site sees them*, which is what
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._local_callees` answers: by
        then each is in a copy, either the one
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._module_inline` brought into
        the caller's Container or the one
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._rooted_copy` took of the
        file. Neither is the tree the frontend parsed, so the
        module the ``ModuleManager`` caches is left as its file declares it
        and a later capture of another kernel meets it unchanged.

        A third disagreement is between an import and a routine already
        brought in. The horizontal FFSL kernels call a module-local helper
        and a sibling module's helper in one body, and both call
        ``fourth_order_horizontal_edge`` from
        ``subgrid_horizontal_support_mod``: inlining the first brings that
        routine into the kernel's Container,
        and the second then arrives importing the same name, which the merge
        answers by renaming the Container's routine through the wrong table
        and raising ``ValueError`` (the survey of 2026-09-13). The callee's
        calls are aimed at the routine the Container holds and the import
        is taken out of its table, which is what the merge would have done
        had it been able to: the two are one routine of one module.

        :param call: the call about to be inlined.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        site = call.ancestor(Routine)
        if site is None:
            return
        container = site.ancestor(Container)
        cls._resolve_from_container(site, container)
        for callee in cls._local_callees(call):
            cls._resolve_from_container(callee, container)
            cls._resolve_shared_names(site, callee.symbol_table)
            cls._resolve_shared_names(callee, site.symbol_table)
            cls._prefer_local_routines(callee, container)

    @classmethod
    def _resolve_from_container(cls, routine, container):
        """Give ``routine``'s unresolved names the Container's import of them.

        A kernel and its module-local helper both read ``S`` through their
        module's ``use reference_element_mod, only : ..., S``, and each
        routine's own table holds the name unresolved; the merge refuses a
        name "present but unresolved in both tables" (``hori_dep_dist_ffsl``,
        2026-09-13). The Container the two sit in names the module, so the
        name is imported from it in the routine's own table, which is the
        ``use ..., only`` the copy the inliner takes will need anyway.

        :param routine: the routine whose unresolved names are settled.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param container: the Container the routine sits in.
        :type container: Optional[:py:class:`psyclone.psyir.nodes.Container`]
        """
        if container is None:
            return
        table = routine.symbol_table
        for symbol in list(table.symbols):
            if not symbol.is_unresolved:
                continue
            outer = container.symbol_table.lookup(
                symbol.name, scope_limit=container, otherwise=None)
            if outer is None or not outer.is_import:
                continue
            cls._import_by_name(
                table, symbol, outer.interface.container_symbol.name)

    @staticmethod
    def _prefer_local_routines(callee, container):
        """Aim ``callee``'s calls at routines ``container`` already holds.

        Each name ``callee`` imports that ``container`` holds as a routine
        of its own -- one an earlier inlining brought in -- has its calls
        re-aimed at that routine and its import removed. An import the
        callee uses in any other way is left, and the merge says what it
        makes of it.

        :param callee: the routine about to be inlined, as the call site
            sees it.
        :type callee: :py:class:`psyclone.psyir.nodes.Routine`
        :param container: the Container the call is made from.
        :type container: Optional[:py:class:`psyclone.psyir.nodes.Container`]
        """
        if container is None:
            return
        local_routines = {routine.name.lower()
                          for routine in container.walk(Routine)}
        table = callee.symbol_table
        for imported in list(table.imported_symbols):
            if imported.name.lower() not in local_routines:
                continue
            local = container.symbol_table.lookup(
                imported.name, scope_limit=container, otherwise=None)
            if local is None or local.is_import:
                continue
            for made in callee.walk(Call):
                if (made.routine is not None
                        and made.routine.symbol is imported):
                    made.routine.symbol = local
            try:
                table.remove(imported)
            except (NotImplementedError, ValueError, KeyError):
                # A use the merge will have to judge for itself.
                continue

    @classmethod
    def _resolve_shared_names(cls, routine, other):
        """Settle each name ``routine`` reads that ``other`` imports.

        A name of ``other``'s own table is one the merge will compare, and
        there are two ways ``routine`` can hold the same name without the
        merge agreeing. It is *unresolved* in ``routine``'s own table -- what
        a wildcard ``use`` leaves -- and :py:meth:`_import_by_name` gives it
        the other scope's module as its origin; or it is not in ``routine``'s
        own table at all, read through a ``use`` of the Container the routine
        sits in, and the copy ``InlineTrans`` takes is detached from that
        Container, so :py:meth:`_localise_import` gives the routine its own
        ``use`` of it first.

        Both are narrow: only a name ``routine`` reads, only a name the other
        scope has of its own, and only towards the module it proves.

        :param routine: the routine whose names are to be settled.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param other: the table this routine's is about to be merged with.
        :type other: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        table = routine.symbol_table
        for signature in routine.reference_accesses().all_signatures:
            name = signature.var_name
            if name not in other:
                continue
            twin = other.lookup(name)
            if not twin.is_import:
                continue
            module = twin.interface.container_symbol.name
            if name in table:
                symbol = table.lookup(name)
                if symbol.is_unresolved:
                    cls._import_by_name(table, symbol, module)
                continue
            symbol = table.lookup(name, otherwise=None)
            if (symbol is not None and symbol.is_import and
                    symbol.interface.container_symbol.name == module):
                cls._localise_import(routine, symbol)

    @staticmethod
    def _import_by_name(table, symbol, module):
        """Make ``symbol`` an import of ``module``'s declaration of the name.

        The module is read by ``resolve_imports``, which needs a
        ``ContainerSymbol`` to read it through and imports a name it was not
        already asked for only under a wildcard. So the Container is given a
        wildcard for the length of the call and narrowed again afterwards:
        what the table ends with is an import of the one name, which is the
        ``use ..., only`` the region needs and not the whole module.

        A module that could not be read or does not publish the name leaves
        nothing behind: ``resolve_imports`` reports both by raising, and any
        ``ContainerSymbol`` added for the attempt is taken out again.

        :param table: the table holding the unresolved symbol.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param symbol: the unresolved symbol to give an origin.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`
        :param str module: the module the other scope imports the name from.
        """
        source = table.lookup(module, otherwise=None) \
            if module in table else None
        added = source is None
        if added:
            source = ContainerSymbol(module)
            table.add(source)
        elif not isinstance(source, ContainerSymbol):
            return
        wildcard = source.wildcard_import
        source.wildcard_import = True
        try:
            table.resolve_imports(container_symbols=[source],
                                  symbol_target=symbol)
        # A module that cannot be read raises KeyError, having found the
        # target in none of the containers it searched; one that publishes the
        # name in a way the table cannot take raises SymbolError. Both are a
        # refusal and not a failure: see the docstring above.
        except (KeyError, SymbolError):
            if added:
                table.remove(source)
                return
        source.wildcard_import = wildcard

    @classmethod
    def _read_declarations(cls, call):
        """Read the declaration of each imported name the two scopes use.

        A name imported by a ``use ..., only`` that PSyclone never had to
        know the type of is a bare
        :py:class:`~psyclone.psyir.symbols.Symbol`: where it comes from is
        recorded and nothing else, which is enough to write it out again and
        not enough to reason about. A pass needing the *type* or the *value*
        stops on it, and lowering an array section to a loop is one:

            The supplied node should be a Reference to a DataSymbol but found
            'eps_r_tran: Symbol<Import(container='constants_mod')>'.

        ``eps_r_tran`` is a ``real(kind=r_tran), parameter`` of LFRic's
        ``constants_mod`` and a bound of a section an FFSL vertical helper
        takes. That declaration is on the search path the capture already
        passes with ``-d``, so it is read rather than guessed:
        :py:meth:`~psyclone.psyir.symbols.SymbolTable.resolve_imports` asked
        for the one name specialises the symbol in place, and the section
        lowering and the constants mixin then see the module's own
        :py:class:`~psyclone.psyir.symbols.DataSymbol`, type and value both.

        Only a name a body reads is asked for, and only one already imported
        from a named module: this reads a declaration the code depends on and
        goes looking for no other. A module the search path does not reach
        leaves the symbol as it was, for the later pass to refuse as it did.

        :param call: the call about to be inlined.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        site = call.ancestor(Routine)
        if site is None:
            return
        for routine in [site] + cls._local_callees(call):
            for signature in routine.reference_accesses().all_signatures:
                symbol = routine.symbol_table.lookup(signature.var_name,
                                                     otherwise=None)
                # pylint: disable-next=unidiomatic-typecheck
                if symbol is None or type(symbol) is not Symbol:
                    continue
                if not symbol.is_import:
                    continue
                table = symbol.find_symbol_table(routine)
                if table is not None:
                    cls._read_declaration(table, symbol)

    @staticmethod
    def _read_declaration(table, symbol):
        """Specialise ``symbol`` to the declaration its module gives it.

        :param table: the table holding the symbol.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param symbol: the bare imported symbol to read the declaration of.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`
        """
        try:
            table.resolve_imports(
                container_symbols=[symbol.interface.container_symbol],
                symbol_target=symbol)
        # A module that cannot be read raises KeyError and one that publishes
        # the name in a way the table cannot take raises SymbolError; either
        # way the symbol is left as it was: see the docstring above.
        except (KeyError, SymbolError):
            pass


__all__ = ["LFRicKokkosImportMixin"]
