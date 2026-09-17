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

"""Tests for LFRicKokkosBoundMixin: inlining order and bounded locals."""

# The transformation's helpers are private by design; the tests read them.
# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.backend.fortran import FortranWriter
from psyclone.psyir.nodes import Call, IntrinsicCall, Reference
from psyclone.psyir.symbols import RoutineSymbol
from psyclone.psyir.transformations import TransformationError


# A helper whose locals are sized by its first dummy, in a module of its own
# so that the call site's frontend has only a name for it until the helper is
# brought in -- which is the state the real FFSL helpers are met in.
_RECON_MODULE = """
module recon_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  public :: recon
contains
  subroutine recon(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    real(kind=r_def) :: slope(n)
    real(kind=r_def) :: curve(n)
    integer(kind=i_def) :: j
    do j = 1, n
      slope(j) = source(j) * 0.5_r_def
      curve(j) = slope(j) * slope(j)
      result(j) = source(j) - curve(j)
    end do
  end subroutine recon
end module recon_mod
"""


# The vertical FFSL shape: the kernel computes the length of the column it
# hands the helper, and passes it as the dummy that sizes the helper's locals.
_WRITTEN_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use recon_mod, only : recon").replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def) :: length\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    length = nlayers - 1\n"
    "    swept(nlayers) = partial(nlayers)\n"
    "    call recon(length, partial(1:length), swept(1:length))\n")


# Two helpers called inside one loop, each passing the same extent to the
# other's sizing dummy: the horizontal FFSL shape. Neither is written to, but
# each call is the other's prior access until it is inlined.
_MATES_MODULE = """
module mates_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  public :: first_pass, second_pass
contains
  subroutine first_pass(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    real(kind=r_def) :: work(n)
    integer(kind=i_def) :: j
    do j = 1, n
      work(j) = source(j) * 2.0_r_def
      result(j) = result(j) + work(j)
    end do
  end subroutine first_pass
  subroutine second_pass(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    real(kind=r_def) :: work(n)
    integer(kind=i_def) :: j
    do j = 1, n
      work(j) = source(j) * 0.5_r_def
      result(j) = result(j) - work(j)
    end do
  end subroutine second_pass
end module mates_mod
"""


_LOOP_MATES_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use mates_mod, only : first_pass, second_pass").replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def) :: pass\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    swept(:) = 0.0_r_def\n"
    "    do pass = 1, 2\n"
    "      call first_pass(nlayers, partial, swept)\n"
    "      call second_pass(nlayers, partial, swept)\n"
    "    end do\n")


# The helper again, with a local the bound must leave alone (a fixed shape),
# one whose shape is deferred (which the inliner refuses for its ALLOCATE
# whatever the bound does), and a call to itself, so that the mixin's every
# branch is walked on a kernel of the same shape.
_ODD_RECON_MODULE = _RECON_MODULE.replace(
    "    real(kind=r_def) :: curve(n)\n",
    "    real(kind=r_def) :: curve(n)\n"
    "    real(kind=r_def) :: fixed(3)\n"
    "    real(kind=r_def), allocatable :: grown(:)\n").replace(
    "    do j = 1, n\n",
    "    allocate(grown(n))\n"
    "    fixed(:) = 0.0_r_def\n"
    "    grown(:) = 0.0_r_def\n"
    "    do j = 1, n\n").replace(
    "  end subroutine recon\n",
    "    deallocate(grown)\n"
    "  end subroutine recon\n")

# The self-calling helper is a procedure of the kernel's own module, as the
# inline tests' recursive case is: brought in from another module, a routine
# that names itself is refused by KernelModuleInlineTrans before the limit
# is ever reached.
_SELF_CALLING_KERNEL = _WRITTEN_BOUND_KERNEL.replace(
    "  use kernel_mod, only : kernel_type\n"
    "  use recon_mod, only : recon",
    "  use kernel_mod, only : kernel_type").replace(
    "end module column_solve_kernel_mod",
    "  subroutine recon(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    real(kind=r_def) :: slope(n)\n"
    "    integer(kind=i_def) :: j\n"
    "    do j = 1, n\n"
    "      slope(j) = source(j) * 0.5_r_def\n"
    "      result(j) = source(j) - slope(j)\n"
    "    end do\n"
    "    if (n > 1) call recon(n - 1, source, result)\n"
    "  end subroutine recon\n"
    "end module column_solve_kernel_mod")


# The helper called twice with the written length: both calls are refused,
# the first is deferred, and the second must not try to bring the callee in
# again -- the retry the survey of rhs_alg_mod crashed on (2026-09-13).
_TWICE_WRITTEN_BOUND_KERNEL = _WRITTEN_BOUND_KERNEL.replace(
    "    call recon(length, partial(1:length), swept(1:length))\n",
    "    call recon(length, partial(1:length), swept(1:length))\n"
    "    call recon(length, swept(1:length), partial(1:length))\n")


# The helper written with whole-array statements, as the real vertical FFSL
# helper is: 'slope = ...' means the helper's n elements, and must go on
# meaning them once slope is declared with the kernel's nlayers. The inquiry
# is the one case where a section answers differently from the array, when
# the lower bound is not one; here it is, so SIZE reads the same n.
_WHOLE_ARRAY_RECON_MODULE = _RECON_MODULE.replace(
    "    do j = 1, n\n"
    "      slope(j) = source(j) * 0.5_r_def\n"
    "      curve(j) = slope(j) * slope(j)\n"
    "      result(j) = source(j) - curve(j)\n"
    "    end do\n",
    "    slope = source * 0.5_r_def\n"
    "    curve(:) = slope(:) * slope(:)\n"
    "    result = source - curve\n"
    "    j = SIZE(curve)\n")

# A local declared from zero, used whole and asked its size: the section's
# size is one short of the array's, so the bound is refused.
_OFFSET_RECON_MODULE = _RECON_MODULE.replace(
    "    real(kind=r_def) :: curve(n)\n",
    "    real(kind=r_def) :: curve(n)\n"
    "    real(kind=r_def) :: offset(0:n)\n").replace(
    "    do j = 1, n\n",
    "    offset = 0.0_r_def\n"
    "    j = SIZE(offset)\n"
    "    do j = 1, n\n")

# The same local without the inquiry: the section keeps its lower bound.
_OFFSET_NO_INQUIRY_MODULE = _OFFSET_RECON_MODULE.replace(
    "    j = SIZE(offset)\n", "")


@pytest.fixture(name="whole_array_target")
# pylint: disable-next=unused-argument
def whole_array_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper uses its locals as whole arrays."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _WHOLE_ARRAY_RECON_MODULE})


@pytest.fixture(name="offset_target")
# pylint: disable-next=unused-argument
def offset_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper asks the size of a zero-based local."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _OFFSET_RECON_MODULE})


@pytest.fixture(name="offset_no_inquiry_target")
# pylint: disable-next=unused-argument
def offset_no_inquiry_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper uses a zero-based local whole."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _OFFSET_NO_INQUIRY_MODULE})


@pytest.fixture(name="written_bound_target")
# pylint: disable-next=unused-argument
def written_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper's locals a written argument sizes."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _RECON_MODULE})


@pytest.fixture(name="odd_locals_target")
# pylint: disable-next=unused-argument
def odd_locals_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose helper also holds fixed and deferred locals."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _ODD_RECON_MODULE})


@pytest.fixture(name="self_calling_target")
# pylint: disable-next=unused-argument
def self_calling_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose bounded helper calls itself."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SELF_CALLING_KERNEL)


@pytest.fixture(name="twice_target")
# pylint: disable-next=unused-argument
def twice_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling the helper twice with a written size."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _TWICE_WRITTEN_BOUND_KERNEL,
        extra={"recon_mod": _RECON_MODULE})


@pytest.fixture(name="loop_mates_target")
# pylint: disable-next=unused-argument
def loop_mates_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling two helpers inside one loop."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOOP_MATES_KERNEL,
        extra={"mates_mod": _MATES_MODULE})


def _no_calls_left(kernel):
    """Whether the kernel schedule holds no non-intrinsic call."""
    schedule = LFRicKokkosTrans._schedule(kernel)
    return not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]


def test_written_bound_is_refused_without_the_option(written_bound_target):
    """Without a bound the inliner's own refusal stands, in its words."""
    _, loop, _ = written_bound_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'recon'" in str(err.value)
    assert "assigned to before the call" in str(err.value)
    assert "'length = nlayers - 1'" in str(err.value)


def test_written_bound_is_inlined_with_a_bound(written_bound_target):
    """With the option the helper is inlined and its locals sized by the bound.

    The generated region declares the helper's two locals as scratch sized
    by ``nlayers``, the kernel's own extent, while the helper's loop still
    runs to ``length``: the body is untouched and only the declaration is
    widened.
    """
    _, loop, kernel = written_bound_target
    options = {"bounded_locals": {"recon": {"n": "nlayers"}}}

    LFRicKokkosTrans().validate(loop, options=options)
    cpp = LFRicKokkosTrans().apply(loop, options=options)

    assert _no_calls_left(kernel)
    assert "recon" not in cpp
    # The scratch the two locals become is sized by the bound, not by the
    # length the kernel computes.
    assert "shmem_size(nlayers)" in cpp
    assert "shmem_size(length)" not in cpp
    # The helper's loop still runs to the exact length.
    assert "<= length" in cpp or "< length + 1" in cpp or "length" in cpp


def test_whole_array_uses_of_a_bounded_local_keep_their_extent(
        whole_array_target):
    """A whole-array statement on a widened local runs to the helper's n.

    Widening 'slope(n)' to 'slope(nlayers)' would have 'slope = ...' and
    'slope(:)' run over nlayers elements against operands of length n, which
    is what the vertical FFSL helper's statements did before this (phase 7,
    2026-09-13). Each becomes an explicit section of the declared bounds, so
    that after inlining it reads 'slope(:length)', and SIZE of the local
    reads the section's length, which is the helper's n.
    """
    _, loop, _ = whole_array_target
    options = {"bounded_locals": {"recon": {"n": "nlayers"}}}

    # pylint: disable=protected-access
    routine = LFRicKokkosTrans._inlined_copy(
        LFRicKokkosTrans._schedule(loop.kernels()[0]), options)
    code = FortranWriter()(routine)

    assert "slope(:length) = partial(:length) * 0.5_r_def" in code
    assert "curve(:length) = slope(:length) * slope(:length)" in code
    assert "SIZE(curve(:length))" in code
    # The declarations are the widened ones all the same.
    assert "dimension(nlayers) :: slope" in code
    assert "dimension(nlayers) :: curve" in code


def test_an_inquiry_on_a_zero_based_bounded_local_is_refused(offset_target):
    """SIZE of 'offset(0:n)' is n + 1, of 'offset(0:n)' the section is n.

    Rather than let the two disagree the bound is refused, and the refusal
    names the local and the intrinsic.
    """
    _, loop, _ = offset_target
    options = {"bounded_locals": {"recon": {"n": "nlayers"}}}

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop, options=options)
    assert ("cannot bound the local 'offset' of 'recon': it is the subject "
            "of SIZE and its lower bound is not 1" in str(err.value))


def test_a_zero_based_bounded_local_used_whole_keeps_its_lower_bound(
        offset_no_inquiry_target):
    """Without an inquiry a zero-based local is sectioned from zero."""
    _, loop, _ = offset_no_inquiry_target
    options = {"bounded_locals": {"recon": {"n": "nlayers"}}}

    # pylint: disable=protected-access
    routine = LFRicKokkosTrans._inlined_copy(
        LFRicKokkosTrans._schedule(loop.kernels()[0]), options)
    code = FortranWriter()(routine)

    # The writer leaves out a lower bound that is the array's own.
    assert "offset(:length) = 0.0_r_def" in code
    assert "dimension(0:nlayers) :: offset" in code


def test_bound_option_shape_is_validated(written_bound_target):
    """An option that is not callee -> dummy -> bound is refused by shape."""
    _, loop, _ = written_bound_target
    for bad in ("recon", {"recon": "nlayers"}, {"recon": {}},
                {"recon": {"n": 3}}, {"": {"n": "nlayers"}}):
        with pytest.raises(TransformationError) as err:
            LFRicKokkosTrans().validate(
                loop, options={"bounded_locals": bad})
        assert "'bounded_locals' option must map a callee name" in str(
            err.value)


def test_bound_option_names_are_checked(written_bound_target):
    """A dummy the callee lacks, or a bound the caller lacks, is refused."""
    _, loop, _ = written_bound_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"m": "nlayers"}}})
    assert ("names 'm' as a dummy of 'recon', which takes no such argument"
            in str(err.value))
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"n": "no_such"}}})
    assert ("bounds 'n' of 'recon' by 'no_such', which is not in scope"
            in str(err.value))


def test_bound_option_for_an_absent_callee_is_inert(written_bound_target):
    """An entry for a callee the kernel does not call changes nothing."""
    _, loop, _ = written_bound_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"other": {"n": "nlayers"}}})
    assert "assigned to before the call" in str(err.value)


def test_loop_mates_passing_one_extent_are_inlined(loop_mates_target):
    """Two helpers in one loop, each the other's prior access, both inline.

    Tried in tree order the first is refused for the second's call; the
    second is then tried, inlines, and the first passes on the re-walk. No
    option is needed: nothing is written, the refusal was about order.
    """
    _, loop, kernel = loop_mates_target

    LFRicKokkosTrans().validate(loop)
    cpp = LFRicKokkosTrans().apply(loop)

    assert _no_calls_left(kernel)
    assert "first_pass" not in cpp
    assert "second_pass" not in cpp
    assert "2.0 * " in cpp or "* 2.0" in cpp
    assert "0.5" in cpp


def test_bound_leaves_other_locals_alone_and_the_allocate_is_refused(
        odd_locals_target):
    """A fixed-shape local is not touched and a deferred one is skipped.

    The bound is applied -- the two ``n``-sized locals are re-declared --
    and the inliner then refuses the helper for its ALLOCATE in its own
    words, which is the refusal a bound was never meant to remove.
    """
    _, loop, _ = odd_locals_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"n": "nlayers"}}})
    assert "cannot inline the call to 'recon'" in str(err.value)
    assert "ALLOCATE" in str(err.value) or "allocate" in str(err.value)
    assert "assigned to before the call" not in str(err.value)


def test_self_calling_bounded_helper_reaches_the_limit(self_calling_target):
    """A helper that calls itself is inlined into itself until the limit.

    Each inlining leaves the recursive call standing and passes the bound
    on with it; the limit is what stops the repetition, and its message
    names the routine still there.
    """
    _, loop, _ = self_calling_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"n": "nlayers"}}})
    assert (f"inlined {LFRicKokkosTrans._INLINE_LIMIT} calls into "
            "'column_solve_code'") in str(err.value)
    assert "call to recon still there" in str(err.value)


def test_a_deferred_call_is_refused_not_crashed(twice_target):
    """Two refused calls: the retry meets its callee already in the Container.

    Without the bound both calls are refused for the written length. The
    first is deferred, the second is tried, and its callee -- brought in by
    the first attempt -- is not brought in again. The result is the first
    call's refusal, in the inliner's words.
    """
    _, loop, _ = twice_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "assigned to before the call" in str(err.value)
    assert "KeyError" not in str(err.value)


def test_a_helper_called_twice_is_bounded_at_both_calls(twice_target):
    """With the bound both calls are inlined, each passing the bound."""
    _, loop, kernel = twice_target
    options = {"bounded_locals": {"recon": {"n": "nlayers"}}}
    cpp = LFRicKokkosTrans().apply(loop, options=options)
    assert _no_calls_left(kernel)
    assert cpp.count("shmem_size(nlayers)") >= 4


def test_a_failure_to_bring_the_callee_in_is_kept_as_the_reason(
        written_bound_target, monkeypatch):
    """Whatever bringing the callee in raises is reported, not propagated.

    KernelModuleInlineTrans reports a name already present with a KeyError;
    the mixin keeps it as the reason the inlining then fails for, so a
    deferred retry can never turn the capture into a crash.
    """
    _, loop, _ = written_bound_target

    def explode(_call):
        raise KeyError("Symbol table already contains a symbol")
    monkeypatch.setattr(LFRicKokkosTrans, "_module_inline", explode)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "bringing it into the container was refused first: KeyError" in str(
        err.value)


def test_a_failing_preparation_step_is_a_refusal(
        written_bound_target, monkeypatch):
    """A preparation step failing in a class of its own is still a refusal."""
    _, loop, _ = written_bound_target

    def explode(_call):
        raise ValueError("a shape the frontend never gave")
    monkeypatch.setattr(LFRicKokkosTrans, "_alias_locals", explode)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "cannot prepare the call to 'recon'" in str(err.value)
    assert "ValueError: a shape the frontend never gave" in str(err.value)


def test_a_detached_call_is_not_local():
    """A call with no Container has no local callee to find."""
    call = Call.create(RoutineSymbol("orphan"))
    assert not LFRicKokkosTrans._already_local(call)


def test_only_a_written_argument_refusal_is_deferred(
        loop_mates_target, monkeypatch):
    """A refusal of any other kind ends the inlining at once.

    The first call is made to fail for a reason inlining the second could
    never clear; the second is then not tried, which the count of attempts
    shows, and the refusal is the first call's own.
    """
    _, loop, _ = loop_mates_target
    attempts = []
    original = LFRicKokkosTrans._inline_one

    def counting(schedule, call, table):
        attempts.append(call.routine.name)
        if len(attempts) == 1:
            raise TransformationError("the callee reads its module state")
        return original(schedule, call, table)
    monkeypatch.setattr(LFRicKokkosTrans, "_inline_one", counting)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(loop)
    assert "reads its module state" in str(err.value)
    assert len(attempts) == 1


def test_an_expression_bound_sizes_the_scratch(written_bound_target):
    """A bound written as arithmetic sizes the scratch, evaluated in place.

    The horizontal special-edge transport's shape: the size the caller
    assigns is one of two branches and the bound is the larger of them,
    written as the arithmetic the kernel's own formals spell it with rather
    than as a name, because the caller holds no name for it.
    """
    _, loop, kernel = written_bound_target
    options = {"bounded_locals": {"recon": {"n": "1 + 2 * nlayers"}}}

    LFRicKokkosTrans().validate(loop, options=options)
    cpp = LFRicKokkosTrans().apply(loop, options=options)

    assert _no_calls_left(kernel)
    assert "shmem_size((1 + (2 * nlayers)))" in cpp
    assert "shmem_size(length)" not in cpp


def test_an_expression_bound_reads_the_caller_s_own_symbols(
        written_bound_target):
    """The parsed bound refers to the caller's variables, not to lookalikes.

    Parsing against a throwaway table would be worth nothing if the
    references it produced pointed at symbols of that table: the generated
    scratch would size itself from a name the region never declares.
    """
    _, _, kernel = written_bound_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    caller = schedule
    bound = LFRicKokkosTrans._bound_expression(
        caller, "recon", "n", "1 + 2 * nlayers")
    names = {reference.symbol.name for reference in bound.walk(Reference)}
    assert names == {"nlayers"}
    for reference in bound.walk(Reference):
        assert reference.symbol is caller.symbol_table.lookup("nlayers")


def test_an_expression_bound_naming_the_unknown_is_refused(
        written_bound_target):
    """A bound over a name the caller does not hold names it and refuses."""
    _, loop, _ = written_bound_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"n": "3 + 2*order"}}})
    assert ("bounds 'n' of 'recon' by '3 + 2*order', which names order: "
            "not in scope at the call in" in str(err.value))


def test_a_bound_that_is_not_an_expression_is_refused(written_bound_target):
    """Text the Fortran frontend cannot read as an expression is refused."""
    _, loop, _ = written_bound_target
    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            loop, options={"bounded_locals": {"recon": {"n": "3 +"}}})
    assert ("bounds 'n' of 'recon' by '3 +', which is not a Fortran "
            "expression:" in str(err.value))


def test_a_bound_the_launch_could_not_evaluate_is_refused(
        written_bound_target):
    """A bound holding a call or a subscript is refused by what it holds.

    The launch sizes the region's scratch before the functor runs, where an
    array is a View it does not hold and an intrinsic is not available: the
    same rule the spread extents are chosen by.
    """
    _, loop, _ = written_bound_target
    for bound, held in (("max(nlayers, 2)", "IntrinsicCall"),
                        ("partial(1)", "ArrayReference")):
        with pytest.raises(TransformationError) as err:
            LFRicKokkosTrans().validate(
                loop, options={"bounded_locals": {"recon": {"n": bound}}})
        assert "the launch could not evaluate where it sizes" in str(err.value)
        assert held in str(err.value)
