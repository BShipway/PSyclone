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


"""Tests for LFRicKokkosElementMixin: an element where an array of one is
declared."""

# The transformation's helpers are private by design; the tests read them.
# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import (
    ArrayReference, Call, IntrinsicCall, Reference)
from psyclone.psyir.transformations import TransformationError


# A helper that takes an array and is called with one element of one, which
# is what Fortran's sequence association allows and PSyIR's rank-matching
# refuses. The extent is a dummy so that only the call decides it.
_EDGE_MODULE = """
module edge_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  public :: edge
contains
  subroutine edge(value, n)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), intent(inout) :: value(n)
    integer(kind=i_def) :: j
    do j = 1, n
      value(j) = value(j) * 2.0_r_def
    end do
  end subroutine edge
end module edge_mod
"""


# The horizontal FFSL shape: a level at a time, each level's value handed to
# a helper that declares an array the call sizes to one.
_ELEMENT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use edge_mod, only : edge").replace(
    "    do k = 1, nlayers\n"
    "      field_out(map_w3(1) + k - 1) = swept(k)\n"
    "    end do\n",
    "    do k = 1, nlayers\n"
    "      call edge(swept(k), 1)\n"
    "      field_out(map_w3(1) + k - 1) = swept(k)\n"
    "    end do\n")


# The same call with an extent the call site does not fix to one: the helper
# reads as many elements as the kernel has levels, which sequence association
# permits and a one-element section would not say.
_COLUMN_EXTENT_KERNEL = _ELEMENT_KERNEL.replace(
    "call edge(swept(k), 1)", "call edge(swept(k), nlayers)")


# An extent of two: the helper reads the element and its successor, which is
# what a one-element section would silently drop.
_PAIR_EXTENT_KERNEL = _ELEMENT_KERNEL.replace(
    "call edge(swept(k), 1)", "call edge(swept(k), 2)")


# A helper taking an assumed-shape array, whose declaration states no bound
# for this or any call to read.
_ASSUMED_MODULE = _EDGE_MODULE.replace("value(n)", "value(:)")


# A helper whose extent is arithmetic rather than a size, and a call that
# makes it one element without saying so in a literal.
_ARITHMETIC_MODULE = _EDGE_MODULE.replace("value(n)", "value(n + 1)").replace(
    "do j = 1, n", "do j = 1, n + 1")
_ARITHMETIC_KERNEL = _ELEMENT_KERNEL.replace(
    "call edge(swept(k), 1)", "call edge(swept(k), 0)")


# An element of a rank-2 array: two subscripts, which sequence association
# reads as the element and everything after it in storage order.
_RANK_TWO_KERNEL = _ELEMENT_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers,2) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    swept(nlayers,1) = partial(nlayers)").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k,1) = swept(k + 1,1) - partial(k)").replace(
    "call edge(swept(k), 1)", "call edge(swept(k,1), 1)").replace(
    "field_out(map_w3(1) + k - 1) = swept(k)",
    "field_out(map_w3(1) + k - 1) = swept(k,1)")


@pytest.fixture(name="element_target")
# pylint: disable-next=unused-argument
def element_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing one element where an array of one is taken."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ELEMENT_KERNEL,
        extra={"edge_mod": _EDGE_MODULE})


@pytest.fixture(name="column_extent_target")
# pylint: disable-next=unused-argument
def column_extent_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper's extent the call does not fix to one."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _COLUMN_EXTENT_KERNEL,
        extra={"edge_mod": _EDGE_MODULE})


@pytest.fixture(name="pair_extent_target")
# pylint: disable-next=unused-argument
def pair_extent_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper reads two elements from the one passed."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _PAIR_EXTENT_KERNEL,
        extra={"edge_mod": _EDGE_MODULE})


@pytest.fixture(name="assumed_target")
# pylint: disable-next=unused-argument
def assumed_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper takes an assumed-shape array."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ELEMENT_KERNEL,
        extra={"edge_mod": _ASSUMED_MODULE})


@pytest.fixture(name="arithmetic_target")
# pylint: disable-next=unused-argument
def arithmetic_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper's extent is arithmetic over a dummy."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARITHMETIC_KERNEL,
        extra={"edge_mod": _ARITHMETIC_MODULE})


@pytest.fixture(name="rank_two_target")
# pylint: disable-next=unused-argument
def rank_two_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing an element of a rank-2 array."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RANK_TWO_KERNEL,
        extra={"edge_mod": _EDGE_MODULE})


def _no_calls_left(kernel):
    """Whether the kernel schedule holds no non-intrinsic call."""
    schedule = LFRicKokkosTrans._schedule(kernel)
    return not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]


def test_an_element_reaching_an_array_of_one_is_inlined(element_target):
    """The actual becomes the one-element section and the call inlines.

    Without the rewriting PSyIR matches the call against no routine at all
    and the region is refused for a call site that is legal Fortran.
    """
    _, loop, kernel = element_target

    LFRicKokkosTrans().validate(loop)
    cpp = LFRicKokkosTrans().apply(loop)

    assert _no_calls_left(kernel)
    assert "edge" not in cpp
    # The helper's body ran against the one element it was given: its own
    # loop runs over one iteration and its subscript is offset from ``k``,
    # which is the section's lower bound.
    assert "TeamVectorRange(team, 1, 1 + 1)" in cpp
    element = "swept((((j - 1) + k) - 1))"
    assert f"{element} = ({element} * 2.0)" in cpp


def test_the_rewriting_reaches_only_the_argument_it_covers(element_target):
    """The extent actual, a literal of its own, is left as a literal."""
    _, _, kernel = element_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    call = [each for each in schedule.walk(Call)
            if not isinstance(each, IntrinsicCall)][0]
    LFRicKokkosTrans()._module_inline(call)
    LFRicKokkosTrans()._section_element_actuals(call)
    assert call.arguments[0].debug_string() == "swept(k:k)"
    assert call.arguments[1].debug_string() == "1"


def test_an_extent_the_call_does_not_fix_is_left_alone(column_extent_target):
    """A helper reading a whole column from one element is refused, not cut.

    The rewriting would say the helper reads one element where the Fortran
    says it reads the column, so the actual is left exactly as written and
    PSyclone refuses the call in its own words with the call named.
    """
    _, loop, _ = column_extent_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'edge'" in str(err.value)
    assert "swept(k)" in str(err.value)


def test_an_extent_of_more_than_one_is_left_alone(pair_extent_target):
    """Two elements are two, and a section of one would drop the second."""
    _, loop, _ = pair_extent_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'edge'" in str(err.value)


def test_a_named_argument_stops_the_rewriting(element_target, monkeypatch):
    """A named actual is not at the position its dummy is, so none is read."""
    _, _, kernel = element_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    call = [each for each in schedule.walk(Call)
            if not isinstance(each, IntrinsicCall)][0]
    LFRicKokkosTrans()._module_inline(call)
    monkeypatch.setattr(type(call), "argument_names",
                        property(lambda self: [None, "n"]))
    LFRicKokkosTrans()._section_element_actuals(call)
    assert isinstance(call.arguments[0], ArrayReference)
    assert call.arguments[0].debug_string() == "swept(k)"


def test_a_name_more_than_one_routine_answers_to_is_left_alone(
        element_target, monkeypatch):
    """Which specific of an interface a call resolves to is not guessed."""
    _, _, kernel = element_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    call = [each for each in schedule.walk(Call)
            if not isinstance(each, IntrinsicCall)][0]
    LFRicKokkosTrans()._module_inline(call)
    routines = LFRicKokkosTrans._local_callees(call)
    monkeypatch.setattr(LFRicKokkosTrans, "_local_callees",
                        classmethod(lambda cls, node: routines * 2))
    LFRicKokkosTrans()._section_element_actuals(call)
    assert call.arguments[0].debug_string() == "swept(k)"


def test_what_is_not_an_element_is_recognised_as_such(element_target):
    """The element test names each thing it is not, on real PSyIR."""
    _, _, kernel = element_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    call = [each for each in schedule.walk(Call)
            if not isinstance(each, IntrinsicCall)][0]
    # A literal is not a reference at all, and a whole array is not one
    # element of one.
    assert not LFRicKokkosTrans._is_element(call.arguments[1])
    assert LFRicKokkosTrans._is_element(call.arguments[0])


def _prepared_call(kernel):
    """Return the kernel's one call, with its callee brought into scope."""
    schedule = LFRicKokkosTrans._schedule(kernel)
    call = [each for each in schedule.walk(Call)
            if not isinstance(each, IntrinsicCall)][0]
    LFRicKokkosTrans()._module_inline(call)
    return call


def test_an_assumed_shape_dummy_states_no_extent_to_read(assumed_target):
    """``value(:)`` declares no bound, so no call makes it one element."""
    _, _, kernel = assumed_target
    call = _prepared_call(kernel)
    formals = LFRicKokkosTrans._local_callees(call)[0]\
        .symbol_table.argument_list
    positions = {id(formal): index for index, formal in enumerate(formals)}
    assert not LFRicKokkosTrans._holds_one_element(
        formals[0], call, positions)


def test_a_scalar_dummy_is_not_an_array_of_one(element_target):
    """A dummy that is not an array at all holds no element to read."""
    _, _, kernel = element_target
    call = _prepared_call(kernel)
    formals = LFRicKokkosTrans._local_callees(call)[0]\
        .symbol_table.argument_list
    positions = {id(formal): index for index, formal in enumerate(formals)}
    assert not LFRicKokkosTrans._holds_one_element(
        formals[1], call, positions)


def test_an_extent_written_as_arithmetic_is_not_folded(arithmetic_target):
    """``value(n + 1)`` with ``n`` zero is one element and is still refused.

    Reading the bound is not evaluating it: folding would need a constant
    evaluator, and a kernel writing a single level writes the size itself.
    """
    _, loop, _ = arithmetic_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'edge'" in str(err.value)


def test_an_element_of_a_rank_two_array_is_left_alone(rank_two_target):
    """Two subscripts are sequence association this does not model."""
    _, loop, _ = rank_two_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'edge'" in str(err.value)


def test_a_bound_naming_something_other_than_a_dummy_is_kept(element_target):
    """A bound the call does not supply is returned as the callee wrote it."""
    _, _, kernel = element_target
    call = _prepared_call(kernel)
    caller = LFRicKokkosTrans._schedule(kernel)
    bound = Reference(caller.symbol_table.lookup("nlayers"))
    assert LFRicKokkosTrans._with_actuals(bound, call, {}) == bound
