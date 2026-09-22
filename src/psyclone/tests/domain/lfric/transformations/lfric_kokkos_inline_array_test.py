# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for a subscripted imported name the frontend read as a call."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin import (
    LFRicKokkosInlineMixin)
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Call, IntrinsicCall, Literal,
    Reference)
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType
from psyclone.psyir.transformations import TransformationError


# LFRic's `coord_transform_mod` in miniature: a module whose only export is a
# `parameter` array. `PANEL_ROT_MATRIX(3,3,6)` is the one this exists for;
# one dimension is the same phenomenon and reads more clearly.
_WEIGHT_MODULE = """
module weight_index_mod
  use constants_mod, only : i_def
  implicit none
  private
  integer(kind=i_def), public, parameter :: weight(3) = &
      (/ 2_i_def, 3_i_def, 5_i_def /)
end module weight_index_mod
"""


# The helper that reads an element of it, naming the array in an `only` list
# and nothing else. Nothing in this file says whether `weight(j)` subscripts
# an array or calls a function, so the frontend leaves a `Call` behind.
_WEIGHTED_HELPER = """
module weight_sweep_mod
  use constants_mod, only : i_def, r_def
  use weight_index_mod, only : weight
  implicit none
  private
  public :: sweep_column
contains
  subroutine sweep_column(levels, source, result)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: source
    real(kind=r_def), dimension(levels), intent(inout) :: result
    integer(kind=i_def) :: j
    do j = 1, levels
      result(j) = source(j) * real(weight(1 + mod(j, 3)), r_def)
    end do
  end subroutine sweep_column
end module weight_sweep_mod
"""


# The kernel that calls it. The name arrives in the kernel body only once the
# helper has been inlined, which is why the rewrite is run before each pass of
# the inlining rather than once at the start.
_WEIGHTED_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use weight_sweep_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# The same shape with the module removed from the search path. Nothing can be
# read, so nothing says the name is data and it stays the call it was parsed
# as -- and is refused for being one, as before.
_UNREADABLE_KERNEL = _WEIGHTED_KERNEL.replace(
    "weight_sweep_mod", "missing_sweep_mod")


def _weight_invoke(tmp_path, kernel_source):
    """Build an invoke whose kernel reads an element of a module's array.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str kernel_source: the kernel module the algorithm calls.

    :returns: as :py:func:`lfric_kokkos_sources._invoke` does.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM, kernel_source,
                   extra={"weight_index_mod": _WEIGHT_MODULE,
                          "weight_sweep_mod": _WEIGHTED_HELPER})


def test_a_subscripted_module_array_is_not_a_call(
        tmp_path, clear_module_manager_instance):
    """An element of an imported `parameter` array is read as one.

    `weight(j)` is an array access and fparser2 cannot tell: a name brought
    in by a `use ..., only` has no type to say whether the parentheses
    subscript or pass arguments, and the frontend settles that towards a
    `Call`. Being wrong about it costs the kernel -- resolving the callee
    reaches the module's datum of that name and raises out of `specialise` --
    so the declaration is read and the node rebuilt as the access it was.

    The check is the capture: the array reaches the region as the constant it
    is, and the region subscripts it.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _weight_invoke(tmp_path, _WEIGHTED_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert schedule.walk(ArrayReference)
    assert "sweep_column" not in cpp
    assert "weight" in cpp
    assert "use weight_index_mod, only : weight" in str(psy.gen).lower()


def test_a_name_no_module_settles_is_left_the_call_it_was(
        tmp_path, clear_module_manager_instance):
    """A name nothing can be read about is still refused as a call.

    The rewrite turns on a declaration read back from the module, not on a
    guess about what parentheses mean. Where the module cannot be read the
    node stays a `Call` and the kernel is declined, which is the behaviour
    this rewrite is careful not to widen.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _weight_invoke(tmp_path, _UNREADABLE_KERNEL)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().apply(loop)
    assert "sweep_column" in str(err.value)


def test_rebuild_data_accesses_leaves_every_other_call_alone(
        tmp_path, clear_module_manager_instance, monkeypatch):
    """The conditions that make a node an array access rather than a call.

    Each is asked of the tree rather than of the declaration, and each rules
    out a call rather than merely making one unlikely: a statement cannot be
    an array access, `f()` is a call whatever `f` is, a subscript carries no
    keyword, and a node with no routine symbol says nothing either way. A
    node failing any of them is left as it is, whatever its module says.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, _, kernel = _weight_invoke(tmp_path, _WEIGHTED_KERNEL)
    schedule = LFRicKokkosTrans._schedule(kernel)
    weight = DataSymbol("weight",
                        ArrayType(ScalarType.integer_type(), [3]))
    schedule.symbol_table.add(weight)
    target = schedule.symbol_table.lookup("swept")

    def _index():
        """:returns: a fresh subscript to build a node with."""
        return Literal("2", ScalarType.integer_type())

    def _rebuilt(arguments):
        """:returns: whether `weight(arguments)` is rebuilt as an access."""
        call = Call.create(Reference(weight), arguments)
        schedule.addchild(
            Assignment.create(ArrayReference.create(
                target, [Literal("1", ScalarType.integer_type())]), call))
        LFRicKokkosInlineMixin._rebuild_data_accesses(schedule)
        rebuilt = isinstance(schedule.children[-1].rhs, ArrayReference)
        schedule.children[-1].detach()
        return rebuilt

    assert _rebuilt([_index()])
    # `weight()` is a call whatever the module says `weight` is.
    assert not _rebuilt([])
    # An array subscript carries no keyword.
    assert not _rebuilt([("index", _index())])
    # A node with no routine symbol names nothing to read a declaration of.
    monkeypatch.setattr(type(schedule.walk(Call)[0]), "routine",
                        property(lambda _self: None))
    assert not _rebuilt([_index()])
    monkeypatch.undo()

    # 'call sweep_column(...)' is a statement, and stays one.
    calls = [call for call in schedule.walk(Call)
             if not isinstance(call, IntrinsicCall)]
    assert len(calls) == 1
    LFRicKokkosInlineMixin._rebuild_data_accesses(schedule)
    assert schedule.walk(Call)[0] is calls[0]
