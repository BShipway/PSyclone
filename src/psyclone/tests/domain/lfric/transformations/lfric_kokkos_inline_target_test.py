# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosInlineMixin: inlining a called routine."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _HDIV_SECTION_KERNEL, _KERNEL, _LOCAL_KERNEL)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall, Loop
from psyclone.psyir.transformations import (
    InlineTrans, TransformationError)


# The same kernel passing a column of the local to a routine, which is how
# convert_hdiv_native_code reaches native_jacobian. The section is outside an
# assignment and so beyond the lowering, but the call is beyond the capture
# altogether, and that is the reason worth reporting.
_HDIV_CALL_KERNEL = _HDIV_SECTION_KERNEL.replace(
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)",
    "    call native_jacobian(vector(:,1))\n"
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)")


# A kernel calling a function from a module PSyclone has not read. Without the
# module the frontend cannot tell `helper(k)` from an array reference, so the
# symbol is a plain Symbol until something resolves it -- which is why the
# refusal has to be the one about calls rather than one about module data.
_CALLED_ROUTINE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_mod, only : helper").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - helper(k)")


# A second module holding the sweep, and holding state of its own that the
# sweep reads. This is the shape coordinate_jacobian has -- a GungHo helper in
# a module the kernel `use`s, reading a rotation matrix that module keeps --
# and it is the shape inlining cannot reach: the body could be moved but the
# datum it reads could not, since the kernel's module neither declares it nor
# has any name for it.
_EXTERNAL_STATE_MODULE = """
module helper_state_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  real(kind=r_def) :: relaxation = 0.5_r_def
  public :: sweep_column
contains
  subroutine sweep_column(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - relaxation * source(j)
    end do
  end subroutine sweep_column
end module helper_state_mod
"""


# The kernel that calls it. Written the same way as _MODULE_PROCEDURE_KERNEL
# so that the only difference between the two is where the callee lives.
_EXTERNAL_CALLEE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_state_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# A configuration module holding an array, and the kernel that reads it. The
# kernel's own file says nothing about what ``blend_weights`` is, so the
# frontend leaves ``blend_weights(1)`` as a Call: an indexed name in an
# expression is a function reference or an array element, and only the other
# module's source settles which. This is the shape the limited-area kernels
# have, and it is the shape that reaches a TypeError rather than a refusal
# when PSyclone is asked for the callee.
_ARRAY_LIKE_MODULE = """
module weights_config_mod
  use constants_mod, only : r_def
  implicit none
  private
  real(kind=r_def), public :: blend_weights(3) = &
      (/ 1.0_r_def, 2.0_r_def, 3.0_r_def /)
end module weights_config_mod
"""


_ARRAY_LIKE_CALL_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use weights_config_mod, only : blend_weights").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) * blend_weights(1)")


# The sweep of _LOCAL_KERNEL taken out into a module procedure of the kernel's
# own module. This is the shape inlining can reach: the callee is a procedure
# of the very Container the call site is in, so no import has to be followed
# and no second module's source has to be read. Its formals are an extent, a
# read column and a written one, which is what a GungHo helper looks like.
_MODULE_PROCEDURE_KERNEL = _LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same module procedure reached through a chain: the kernel calls one
# helper and that helper calls a second. Inlining one call exposes the next,
# so the rewrite has to be repeated until none is left rather than run once.
_CHAINED_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    call seed_column(n, source, result)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n").replace(
    "  end subroutine sweep_column\n",
    "  end subroutine sweep_column\n"
    "  subroutine seed_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine seed_column\n")


# A callee that calls itself. Inlining it once leaves a call to it behind, so
# a rewrite run to a fixed point would never reach one; the depth limit is
# what turns that into a refusal naming the routine.
_RECURSIVE_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    result(n) = source(n)\n"
    "    j = n - 1\n"
    "    call sweep_column(j, source, result)\n")


# A kernel handing a column of a rank-2 local to a module procedure, which is
# the shape convert_hdiv_native_code reaches native_jacobian with. The actual
# is a contiguous whole-dimension section, so before inlining it is a section
# outside an assignment; afterwards there is no argument at all, because the
# formal's every use has become a subscript of the local itself.
_SECTION_ACTUAL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def), dimension(nlayers, 2) :: column\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    column(:,1) = partial(:)\n"
    "    call sweep_column(nlayers, column(:,1), swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The target kernel subscripting one field by a constant its *module* imports,
# on both sides of an assignment. That is the shape LFRic's inter-grid
# prolongation has -- 'fine_field(map_fine(SWB, ...))', with SWB an
# 'integer, parameter' from reference_element_mod -- and nothing about it is
# inter-grid: what matters is that the name is resolved through the kernel
# module's symbol table rather than the subroutine's, and that the loop reads
# and writes one array so that the dependence analysis has two subscripts to
# compare symbolically. 'n_moist' is planet_config_mod's integer parameter.
_MODULE_INDEX_KERNEL = _KERNEL.replace(
    "  use planet_config_mod, only : recip_epsilon",
    "  use planet_config_mod, only : n_moist, recip_epsilon").replace(
    "        moist_dyn_gas(map_wtheta(df) + k) = &\n"
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "        moist_dyn_gas(map_wtheta(n_moist) + k) = &\n"
    "            moist_dyn_gas(map_wtheta(n_moist) + k) + &\n"
    "            recip_epsilon * mr_v_at_dof")


# ---------------------------------------------------------------------------
# Task E7: a TARGET dummy is inlinable.
# ---------------------------------------------------------------------------


def test_target_dummy_is_inlined(target_dummy_target):
    """A helper taking a TARGET column is inlined as its partial type.

    ``target`` is an assertion about aliasing -- a pointer may be aimed at
    the actual -- and binding the callee's formal to that actual neither
    creates a pointer nor invalidates one. The PSyIR does not model the
    attribute, so the declaration arrives as an ``UnsupportedFortranType``
    carrying the type it could parse; replacing the formal's type with that
    partial type before inlining is what lets ``InlineTrans`` proceed,
    without ``permit_unsupported_type_args`` being passed and so without
    anything else unsupported being let through with it.
    """
    _, loop, kernel = target_dummy_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    # The callee's statements, reading the caller's array through the
    # formals it was called with.
    assert "swept((nlayers - 1)) = partial((nlayers - 1));" in cpp


def test_pointer_dummy_is_still_refused(pointer_dummy_target):
    """A helper taking a POINTER column is refused as before.

    Only ``target`` is dropped. A pointer dummy is a name for storage the
    call site aims elsewhere, which binding a formal to an actual does not
    reproduce, so the declaration stays unsupported and the refusal is the
    one ``InlineTrans`` writes.
    """
    _, loop, _ = pointer_dummy_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert ("Symbol 'source' which is an Argument of UnsupportedType"
            in message)
    assert "POINTER" in message


def test_target_dummy_of_a_shared_helper_is_inlined_twice(
        shared_target_helper):
    """Two kernels calling one helper are each captured.

    The rewrite is made on the copy of the callee the capture works on, so
    the second kernel meets the module as its own file declares it rather
    than as the first capture left it.
    """
    psy, _, _ = shared_target_helper
    loops = psy.invokes.invoke_list[0].schedule.walk(Loop)
    kernel_loops = [loop for loop in loops if loop.kernels()]

    regions = [LFRicKokkosTrans().apply(loop) for loop in kernel_loops]

    assert len(regions) == 2
    for cpp in regions:
        assert "sweep_column" not in cpp
        assert "swept((nlayers - 1)) = partial((nlayers - 1));" in cpp


# ---------------------------------------------------------------------------
# Task E11: an inliner error is a refusal.
# ---------------------------------------------------------------------------


def test_an_unexpected_inliner_error_is_a_refusal(
        monkeypatch, target_dummy_target):
    """Whatever ``InlineTrans`` raises, ``validate`` refuses with it.

    ``validate`` answers one question -- can this loop be captured -- and a
    caller asking it of every loop in a model reads a
    :py:class:`~psyclone.psyir.transformations.TransformationError` as no and
    anything else as a crash. The inliner reaches machinery that raises on
    its own account, so the class of what it raises is not a list this mixin
    can keep: a symbolic comparison the SymPy writer cannot render is a
    ``VisitorError``, and a name that resolves to a datum a ``TypeError``.
    Every one of them is a kernel that is not captured, which is a refusal.
    ``ValueError`` stands for all of them here, and the message names its
    class so that the reader is not left to guess what the text came from.
    """
    _, loop, _ = target_dummy_target

    def _raise(self, node, options=None):
        # pylint: disable=unused-argument
        raise ValueError("nothing sensible to say about this call")

    monkeypatch.setattr(InlineTrans, "apply", _raise)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "ValueError: nothing sensible to say about this call" in message
