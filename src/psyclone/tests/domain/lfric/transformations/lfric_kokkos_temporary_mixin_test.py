# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for an array expression passed to a callee the capture inlines."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import Assignment, Call, IntrinsicCall, Routine


# LFRic's `jacobian_abr2XYZ` in miniature: an array-valued function whose
# dummy is read element by element, which is what an expression actual cannot
# be substituted into.
_SHIFT_MODULE = """
module shift_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  public :: shifted
contains
  function shifted(levels, radius) result(column)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: radius
    real(kind=r_def), dimension(levels) :: column
    integer(kind=i_def) :: j
    do j = 1, levels
      column(j) = 2.0_r_def * radius(j)
    end do
  end function shifted
end module shift_mod
"""

# LFRic's `native_jacobian` in miniature: a public routine passing an
# expression to a private sibling, which the capture inlines into it in the
# module's own tree before the routine is brought into the kernel's.
_NATIVE_MODULE = _SHIFT_MODULE.replace("shift_mod", "native_mod").replace(
    "  public :: shifted\n", "  public :: native\n").replace(
    "contains\n", """contains
  subroutine native(levels, radius, column)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: radius
    real(kind=r_def), dimension(levels), intent(inout) :: column
    column = shifted(levels, radius + 1.0_r_def)
  end subroutine native
""")

# The same call in the positions the preparation has to tell apart.
_PLACES = """
module places_mod
  use unknown_mod, only : opaque, opaque_t
  implicit none
contains
  subroutine places(n, a, r, out, s)
    integer, intent(in) :: n
    real, intent(in) :: a, r(n)
    real, intent(inout) :: out(n)
    type(opaque_t), intent(in) :: s(n)
    out = shifted(n, r)
    out = shifted(n, a + 1.0)
    out = shifted(n, opaque + 1.0)
    do while (any(shifted(n, r + 1.0) > 0.0))
      out = 0.0
    end do
    out = shifted(n, r + 1.0) + shifted(n, 2.0 * r)
    out = shifted(n, cshift(s, 1))
  end subroutine places
  function shifted(levels, radius) result(column)
    integer, intent(in) :: levels
    real, intent(in) :: radius(levels)
    real :: column(levels)
    column = radius
  end function shifted
end module places_mod
"""


def _places():
    """Parse :py:data:`_PLACES` and return its calls to ``shifted``.

    :returns: the routine and its calls to ``shifted``, in source order.
    :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`,
        List[:py:class:`psyclone.psyir.nodes.Call`]]
    """
    psyir = FortranReader().psyir_from_source(_PLACES)
    routine = next(routine for routine in psyir.walk(Routine)
                   if routine.name == "places")
    calls = [call for call in routine.walk(Call)
             if not isinstance(call, IntrinsicCall)]
    return routine, calls


def test_an_array_expression_actual_is_captured(
        tmp_path, clear_module_manager_instance):
    """LFRic's `chi_3_df+radius` actual reaches the region through a local.

    The kernel passes `partial + 1.0_r_def` where the helper reads
    `radius(j)`. The expression is assigned to `shifted_actual` before the
    statement, the helper is inlined against that, and no call is left:
    the local is a column array like the kernel's own, and is placed in
    scratch beside them.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    kernel_source = _LOCAL_KERNEL.replace(
        "  use kernel_mod, only : kernel_type",
        "  use kernel_mod, only : kernel_type\n"
        "  use shift_mod, only : shifted").replace(
        "    swept(nlayers) = partial(nlayers)\n"
        "    do k = nlayers - 1, 1, -1\n"
        "      swept(k) = swept(k + 1) - partial(k)\n"
        "    end do\n",
        "    swept = shifted(nlayers, partial + 1.0_r_def)\n")
    _, loop, kernel = _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, kernel_source,
        extra={"shift_mod": _SHIFT_MODULE})

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert ("shifted_actual((idx - 1)) = (partial((idx - 1)) + 1.0);"
            in cpp)
    assert "(2.0 * shifted_actual((j - 1)))" in cpp


def test_an_array_expression_a_sibling_is_passed_is_captured(
        tmp_path, clear_module_manager_instance):
    """An expression passed between two routines of one module is hoisted.

    `native` passes `radius + 1.0_r_def` to its private sibling `shifted`,
    as `native_jacobian` passes `chi_3_df+radius` to `jacobian_abr2XYZ`.
    The sibling is inlined into `native` in the module's tree, which is
    not the kernel's body, so the expression is given its local there; the
    local then travels with `native` into the region.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    kernel_source = _LOCAL_KERNEL.replace(
        "  use kernel_mod, only : kernel_type",
        "  use kernel_mod, only : kernel_type\n"
        "  use native_mod, only : native").replace(
        "    swept(nlayers) = partial(nlayers)\n"
        "    do k = nlayers - 1, 1, -1\n"
        "      swept(k) = swept(k + 1) - partial(k)\n"
        "    end do\n",
        "    call native(nlayers, partial, swept)\n")
    _, loop, kernel = _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, kernel_source,
        extra={"native_mod": _NATIVE_MODULE})

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "shifted_actual" in cpp
    assert "(2.0 * shifted_actual((j - 1)))" in cpp


def test_only_an_array_expression_is_hoisted():
    """A variable, a scalar and an expression of unknown type are left.

    The first three calls pass `r`, `a + 1.0` and `opaque + 1.0`: a
    variable the inliner substitutes as it is, a scalar it substitutes as
    a value, and an expression whose type the PSyIR does not know. The last
    passes an array of a derived type the PSyIR knows only by name. Neither
    of those two has a declaration to give a local. None is rewritten.
    """
    routine, calls = _places()
    before = routine.debug_string()

    for call in calls[:3] + calls[6:]:
        LFRicKokkosTrans._hoist_array_expressions(call)

    assert routine.debug_string() == before


def test_a_while_condition_is_left_alone():
    """An expression in a `do while` condition is not moved before the loop.

    An assignment before the loop would be evaluated once, where Fortran
    evaluates the condition on every trip.
    """
    routine, calls = _places()
    before = routine.debug_string()

    LFRicKokkosTrans._hoist_array_expressions(calls[3])

    assert routine.debug_string() == before


@pytest.mark.parametrize("index, expression", [(4, "r + 1.0"),
                                               (5, "2.0 * r")])
def test_each_expression_gets_its_own_local(index, expression):
    """Two calls in one statement each get a local, before the statement.

    `new_symbol` numbers the second, and each assignment stands immediately
    before the statement the call was in, in the order they were made.
    """
    routine, calls = _places()
    statement = calls[4].ancestor(Assignment)

    LFRicKokkosTrans._hoist_array_expressions(calls[4])
    LFRicKokkosTrans._hoist_array_expressions(calls[5])

    name = {4: "shifted_actual", 5: "shifted_actual_1"}[index]
    hoisted = statement.parent.children[statement.position - 6 + index]
    assert hoisted.lhs.name == name
    assert hoisted.rhs.debug_string() == expression
    assert f"shifted(n, {name})" in statement.debug_string()
    local = routine.symbol_table.lookup(name)
    assert local.datatype.shape[0].upper.debug_string() == "n"


# The shape of LFRic's alphabetar2xyz, reached through a module of its own:
# a constructor inside an array expression, an elemental conversion of a
# section of a module parameter passed to MATMUL, and a MATMUL over the
# module's PROTECTED state, which is sci_chi_transform_mod's chi2xyz_rot_mat.
_ROTATE_MODULE = """
module rotate_mod
  use constants_mod, only : i_def, r_def, r_second
  implicit none
  private
  real(kind=r_second), public, parameter :: ROT(3,3,2) = reshape((/ &
      1.0_r_second, 0.0_r_second, 0.0_r_second, &
      0.0_r_second, 1.0_r_second, 0.0_r_second, &
      0.0_r_second, 0.0_r_second, 1.0_r_second, &
      0.0_r_second, 1.0_r_second, 0.0_r_second, &
      1.0_r_second, 0.0_r_second, 0.0_r_second, &
      0.0_r_second, 0.0_r_second, 1.0_r_second /), (/ 3, 3, 2 /))
  real(kind=r_def), public, protected :: turn(3,3)
  public :: rotated
contains
  subroutine rotated(alpha, panel, xyz)
    real(kind=r_def), intent(in) :: alpha
    integer(kind=i_def), intent(in) :: panel
    real(kind=r_def), dimension(3), intent(out) :: xyz
    real(kind=r_def), dimension(3) :: local
    local = alpha * (/ 1.0_r_def, tan(alpha), 2.0_r_def /)
    xyz = matmul(real(ROT(:,:,panel), r_def), local)
    xyz = matmul(turn, xyz)
  end subroutine rotated
end module rotate_mod
"""

_ROTATING_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use rotate_mod, only : rotated").replace(
    "    real(kind=r_def), dimension(nlayers) :: swept\n",
    "    real(kind=r_def), dimension(nlayers) :: swept\n"
    "    real(kind=r_def), dimension(3) :: xyz\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call rotated(partial(1), 2_i_def, xyz)\n"
    "    swept = xyz(1) + xyz(3)\n")

_HOISTS = """
module hoists_mod
  use unknown_mod, only : opaque
  implicit none
contains
  subroutine hoists(n, a, r, m, out)
    integer, intent(in) :: n
    real, intent(in) :: a, r(3), m(3,3)
    real, intent(inout) :: out(3)
    real :: s
    out = (/ a, a, a /)
    out = a * (/ r, a /)
    out = 2.0 * (/ opaque, a, a /)
    s = sum((/ a, a /))
    out(1) = a * sum((/ a, a /))
    do while (sum(matmul(2.0 * m, r)) > 0.0)
      out = 0.0
    end do
    out = matmul(transpose(m), r)
    out = matmul(m, r) + matmul(m, opaque + r)
    out = matmul(real(m, 8), 2.0 * r)
    s = real(a)
  end subroutine hoists
end module hoists_mod
"""


def _hoists():
    """Parse :py:data:`_HOISTS` and return its routine and statements.

    :returns: the routine and the statements of its body, in source order.
    :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`,
        List[:py:class:`psyclone.psyir.nodes.Statement`]]
    """
    psyir = FortranReader().psyir_from_source(_HOISTS)
    routine = next(routine for routine in psyir.walk(Routine)
                   if routine.name == "hoists")
    return routine, list(routine.children)


def test_alphabetar2xyz_is_captured(tmp_path, clear_module_manager_instance):
    """The three shapes `alphabetar2xyz` carries reach the region.

    The constructor becomes `constructor`, assigned element by element; the
    converted section becomes `matmul_operand`, a loop over the section;
    and the PROTECTED matrix the helper imports on its module's behalf is
    typed as the real array it is, and taken as a formal of the region.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, kernel = _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ROTATING_KERNEL,
        extra={"rotate_mod": _ROTATE_MODULE})

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    table = schedule.symbol_table
    assert table.lookup("constructor").datatype.shape[0].upper.value == "3"
    assert "constructor((2 - 1)) = Kokkos::tan(partial((1 - 1)));" in cpp
    assert ("local((idx - 1)) = (partial((1 - 1)) * "
            "constructor((idx - 1)));" in cpp)
    assert ("matmul_operand((idx_2 - 1), (idx_1 - 1)) = "
            "(double)rot((idx_2 - 1), (idx_1 - 1), (2 - 1));" in cpp)
    assert "const double *turn_data" in cpp


def test_a_constructor_is_hoisted_only_inside_an_array_expression():
    """A constructor is moved only where the lowering would meet it.

    Standing alone on the right it is what the backend writes already;
    holding an array element, or one of unknown type, it has no length or
    declaration to give a local; in a fold whose value is a scalar, or in
    an assignment to one element, it is not in an assignment the lowering
    rewrites. Only the second statement of all of those qualifies, and it
    holds an array element, so nothing is moved.
    """
    routine, _ = _hoists()
    before = routine.debug_string()

    LFRicKokkosTrans._hoist_constructors(routine)

    assert routine.debug_string() == before


def test_an_operand_is_hoisted_only_where_it_has_to_be():
    """An intrinsic operand is moved only when it is an array expression.

    A whole array stays, and so does another intrinsic of the tier, which
    the writer subscripts itself; an operand in a `do while` condition
    stays, since an assignment before the loop is evaluated once; an
    operand of unknown type stays for want of a declaration. `real(m, 8)`,
    which the PSyIR types as a scalar, and `2.0 * r` are each given a local
    before their statement, `real(m, 8)` of `m`'s shape.
    """
    routine, statements = _hoists()

    LFRicKokkosTrans._hoist_intrinsic_operands(routine)

    names = [statement.lhs.name for statement in routine.children
             if isinstance(statement, Assignment)
             and statement.lhs.name.startswith("matmul_operand")]
    assert names == ["matmul_operand", "matmul_operand_1"]
    local = routine.symbol_table.lookup("matmul_operand")
    assert len(local.datatype.shape) == 2
    assert statements[-2].debug_string() == (
        "out = MATMUL(matmul_operand, matmul_operand_1)\n")
    assert "opaque + r" in statements[-3].debug_string()
    assert "2.0 * m" in statements[5].debug_string()
    assert "TRANSPOSE(m)" in statements[6].debug_string()


def test_a_scalar_conversion_keeps_its_type():
    """An elemental call on scalars is typed as the PSyIR types it."""
    _, statements = _hoists()
    conversion = statements[-1].rhs

    datatype = LFRicKokkosTrans._elemental_datatype(conversion)

    assert datatype is not None
    assert datatype == conversion.datatype
