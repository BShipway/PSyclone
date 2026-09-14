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

"""Tests of LFRicKokkosTrans capturing an LFRic built-in: the synthesised
kernel schedule, the region it generates, and the built-ins it refuses."""

# The fixtures are used by injection, and the mixin under test is made of
# private helpers.
# pylint: disable=unused-argument,protected-access

from types import SimpleNamespace

import pytest

from lfric_kokkos_sources import _KERNEL, _invoke
from psyclone.domain.lfric import lfric_builtins
from psyclone.domain.lfric.lfric_builtins import LFRicSetvalCKern
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.domain.lfric.transformations import lfric_kokkos_builtin_mixin
from psyclone.psyir.nodes import Return
from psyclone.psyir.symbols import ContainerSymbol, ScalarType, SymbolTable
from psyclone.psyir.transformations import TransformationError


# A field incremented by another: two field arguments, no scalar.
_INC_ALGORITHM = """
program kokkos_builtin_inc_test
  use field_mod, only : field_type
  implicit none
  type(field_type) :: f1, f2
  call invoke(inc_X_plus_Y(f1, f2))
end program kokkos_builtin_inc_test
"""


# Two scalars, one a variable and one a literal, so both spellings of a scalar
# actual reach the synthesised body.
_AXPBY_ALGORITHM = """
program kokkos_builtin_axpby_test
  use constants_mod, only : r_def
  use field_mod, only : field_type
  implicit none
  type(field_type) :: f1, f2, f3
  real(kind=r_def) :: a
  call invoke(aX_plus_bY(f3, a, f1, 0.5_r_def, f2))
end program kokkos_builtin_axpby_test
"""


# Two scalars passed the same literal: each must claim one occurrence.
_SAME_LITERAL_ALGORITHM = _AXPBY_ALGORITHM.replace(
    "aX_plus_bY(f3, a, f1, 0.5_r_def, f2)",
    "aX_plus_bY(f3, 0.5_r_def, f1, 0.5_r_def, f2)")


# One scalar read twice by the body.
_TWICE_ALGORITHM = _AXPBY_ALGORITHM.replace(
    "aX_plus_bY(f3, a, f1, 0.5_r_def, f2)", "aX_plus_aY(f3, a, f1, f2)")


# A negated literal, which is an expression rather than a literal.
_NEGATED_ALGORITHM = _AXPBY_ALGORITHM.replace(
    "aX_plus_bY(f3, a, f1, 0.5_r_def, f2)", "a_times_X(f3, -1.0_r_def, f1)")


# A precision conversion between two kinds of different width.
_CONVERT_ALGORITHM = """
program kokkos_builtin_convert_test
  use field_mod, only : field_type
  use r_solver_field_mod, only : r_solver_field_type
  implicit none
  type(field_type) :: f1
  type(r_solver_field_type) :: f2
  call invoke(real_to_real_X(f1, f2))
end program kokkos_builtin_convert_test
"""


# An integer-valued built-in.
_INTEGER_ALGORITHM = """
program kokkos_builtin_integer_test
  use constants_mod, only : i_def
  use integer_field_mod, only : integer_field_type
  implicit none
  type(integer_field_type) :: f1
  call invoke(int_setval_c(f1, 3_i_def))
end program kokkos_builtin_integer_test
"""


# A reduction.
_REDUCTION_ALGORITHM = """
program kokkos_builtin_sum_test
  use constants_mod, only : r_def
  use field_mod, only : field_type
  implicit none
  type(field_type) :: f1
  real(kind=r_def) :: total
  call invoke(sum_X(total, f1))
end program kokkos_builtin_sum_test
"""


def _strip(code):
    """Return ``code`` without its comment lines, for body assertions."""
    return "\n".join(line for line in code.splitlines()
                     if not line.lstrip().startswith("//"))


def test_builtin_is_captured_as_a_dof_region(
        tmp_path, clear_module_manager_instance):
    """A built-in is captured as the dof kernel it is.

    The region is the dof launch over the loop's own count, its body is the
    built-in's assignment over positional formals, each field crosses as a
    rank-1 View at the field role, and the PSy layer passes the fields whole
    where the loop passed one dof of each -- exactly what a coded dof kernel
    gets. The name carries the built-in and the kind of its arguments.
    """
    psy, loop, kernel = _invoke(tmp_path, "inc", _INC_ALGORITHM, _KERNEL)
    assert kernel.name == "inc_x_plus_y"

    code = LFRicKokkosTrans().apply(loop)

    assert 'extern "C" void builtin_inc_x_plus_y_r_def_kokkos(' in code
    assert "double *arg1_data,\n    const double *arg2_data,\n" in code
    assert "Kokkos::RangePolicy<>(0, ndofs)" in code
    assert "arg1(df) = (arg1(df) + arg2(df));" in _strip(code)
    assert "arg1_data, lfric_kokkos::Role::field, ndofs)" in code
    assert "arg2_data, lfric_kokkos::Role::field, ndofs)" in code
    # The comment the lowering attaches to the assignment does not reach
    # the region.
    assert "Built-in" not in code

    fortran = str(psy.gen)
    assert ("call builtin_inc_x_plus_y_r_def_kokkos(f1_data, f2_data, "
            "loop0_stop)" in fortran)
    assert "Built-in" not in fortran
    assert "loop0_stop = f1_proxy%vspace%get_last_dof_owned()" in fortran


def test_builtin_scalars_by_variable_and_by_literal(
        tmp_path, clear_module_manager_instance):
    """A scalar actual is a formal whether the algorithm passed a name or a
    value.

    The lowering writes the algorithm's own expression into the body, so a
    variable is found by its name and a literal by its value, and both
    become the positional formal the PSy layer passes the actual for.
    """
    psy, loop, _ = _invoke(tmp_path, "axpby", _AXPBY_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    assert ("double *arg1_data,\n    const double arg2,\n"
            "    const double *arg3_data,\n    const double arg4,\n"
            "    const double *arg5_data,\n" in code)
    assert ("arg1(df) = ((arg2 * arg3(df)) + (arg4 * arg5(df)));"
            in _strip(code))
    assert "0.5" not in code

    fortran = str(psy.gen)
    assert ("call builtin_ax_plus_by_r_def_kokkos(f3_data, a, f1_data, "
            "0.5_r_def, f2_data, loop0_stop)" in fortran)


def test_builtin_two_scalars_passed_one_literal_each_take_a_formal(
        tmp_path, clear_module_manager_instance):
    """Two scalar arguments given one literal are still two formals.

    The body is written over the formals rather than matched against the
    actuals, so what the algorithm passed cannot change the text: a site
    passing the same literal twice generates the region a site passing two
    variables does, which the whole-model capture requires of the sites of
    one region symbol.
    """
    _, loop, _ = _invoke(tmp_path, "same", _SAME_LITERAL_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    assert ("arg1(df) = ((arg2 * arg3(df)) + (arg4 * arg5(df)));"
            in _strip(code))
    assert "0.5" not in code


def test_builtin_scalar_read_twice_and_negated_literal(
        tmp_path, clear_module_manager_instance):
    """A scalar the body reads twice reads the formal twice, and a negated
    literal never reaches the body at all."""
    _, loop, _ = _invoke(tmp_path, "twice", _TWICE_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    # PSyclone lowers aX_plus_aY as a * (X + Y).
    assert "arg1(df) = (arg2 * (arg3(df) + arg4(df)));" in _strip(code)
    assert "const double arg2,\n" in code
    assert "arg5" not in code

    _, loop, _ = _invoke(tmp_path, "negated", _NEGATED_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    assert "arg1(df) = (arg2 * arg3(df));" in _strip(code)
    assert "-1.0" not in _strip(code)


def test_builtin_of_two_kinds_names_both_and_casts(
        tmp_path, clear_module_manager_instance):
    """A precision conversion carries both kinds and generates the cast.

    The region is named for both kinds in argument order, because one
    built-in name covers every precision and the generated types differ;
    the r_solver field crosses as float, and the body is the cast the
    lowering wrote.
    """
    _, loop, _ = _invoke(tmp_path, "convert", _CONVERT_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    assert ('extern "C" void builtin_real_to_real_x_r_def_r_solver_kokkos('
            in code)
    assert "double *arg1_data,\n    const float *arg2_data,\n" in code
    assert "arg1(df) = (double)arg2(df);" in _strip(code)


def test_builtin_on_an_integer_field(
        tmp_path, clear_module_manager_instance):
    """An integer-valued built-in crosses at the integer kind."""
    _, loop, _ = _invoke(tmp_path, "integer", _INTEGER_ALGORITHM, _KERNEL)

    code = LFRicKokkosTrans().apply(loop)

    assert 'extern "C" void builtin_int_setval_c_i_def_kokkos(' in code
    assert "int *arg1_data,\n    const int arg2,\n" in code
    assert "arg1(df) = arg2;" in _strip(code)


def test_builtin_validate_leaves_the_invoke_as_found(
        tmp_path, clear_module_manager_instance):
    """Validation lowers a copy: the invoke still holds the built-in.

    The lowering that supplies the body replaces the built-in with its
    assignment, which must happen to a copy, because the PSy layer has to
    hold the built-in until apply() replaces the loop with the launch.
    """
    psy, loop, kernel = _invoke(tmp_path, "inc", _INC_ALGORITHM, _KERNEL)

    LFRicKokkosTrans().validate(loop)

    assert loop.kernels() == [kernel]
    fortran = str(psy.gen)
    assert "! Built-in: inc_X_plus_Y" in fortran
    assert "_kokkos(" not in fortran


def test_builtin_rule_passes_over_a_coded_kernel(target):
    """The built-in rule has nothing to say about a coded kernel."""
    _, loop, kernel = target
    assert not LFRicKokkosTrans._is_builtin(kernel)

    LFRicKokkosTrans._validate_builtin(loop)


def test_builtin_reduction_is_refused(
        tmp_path, clear_module_manager_instance):
    """A reduction built-in is refused by name: the dof launch has no shape
    for an accumulation into a scalar."""
    _, loop, _ = _invoke(tmp_path, "sum", _REDUCTION_ALGORITHM, _KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the LFRic builtin 'sum_x': "
            "it is a reduction, and the dof launch has no shape for one."
            in str(error.value))


def test_builtin_of_another_api_is_refused(
        tmp_path, monkeypatch, clear_module_manager_instance):
    """A built-in that is not LFRic's has no lowering this mixin knows."""
    _, loop, _ = _invoke(tmp_path, "inc", _INC_ALGORITHM, _KERNEL)

    class _OtherBuiltIn:  # pylint: disable=too-few-public-methods
        """Stands for another API's built-in class."""

    monkeypatch.setattr(lfric_kokkos_builtin_mixin, "LFRicBuiltIn",
                        _OtherBuiltIn)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the builtin 'inc_x_plus_y': "
            "it is not an LFRic builtin." in str(error.value))


def test_builtin_lowering_that_is_not_an_assignment_is_refused(
        tmp_path, monkeypatch, clear_module_manager_instance):
    """A lowering of any other shape is refused rather than described.

    Every non-reduction built-in lowers to one assignment today; this is
    what a built-in added later with another shape meets.
    """
    _, loop, _ = _invoke(tmp_path, "inc", _INC_ALGORITHM, _KERNEL)

    def _lower(self):
        node = Return()
        self.replace_with(node)
        return node

    monkeypatch.setattr(lfric_builtins.LFRicIncXPlusYKern,
                        "lower_to_language_level", _lower)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the LFRic builtin "
            "'inc_x_plus_y': its lowering is not a single assignment but a "
            "Return." in str(error.value))


def test_builtin_kind_of_an_argument_naming_none_is_the_default():
    """An argument that names no kind is declared at the default precision,
    and creates no import."""
    table = SymbolTable()
    module = ContainerSymbol("constants_mod")
    kinds = {}

    kind = LFRicKokkosTrans._builtin_kind(
        table, kinds, module, SimpleNamespace(precision=None))

    assert kind is ScalarType.Precision.UNDEFINED
    assert not kinds
    assert not table.symbols


def test_builtin_kind_is_created_once_and_imported():
    """Two arguments of one kind share the symbol, imported from the kinds
    module."""
    table = SymbolTable()
    module = ContainerSymbol("constants_mod")
    table.add(module)
    kinds = {}

    first = LFRicKokkosTrans._builtin_kind(
        table, kinds, module, SimpleNamespace(precision="r_def"))
    second = LFRicKokkosTrans._builtin_kind(
        table, kinds, module, SimpleNamespace(precision="r_def"))

    assert first is second
    assert list(kinds) == ["r_def"]
    assert first.interface.container_symbol is module
    assert table.lookup("r_def") is first


@pytest.mark.parametrize("name, intrinsic", [
    ("real", ScalarType.Intrinsic.REAL),
    ("integer", ScalarType.Intrinsic.INTEGER),
    ("logical", ScalarType.Intrinsic.BOOLEAN)])
def test_builtin_psyir_intrinsic(name, intrinsic):
    """Each metadata intrinsic name maps to its PSyIR intrinsic."""
    assert LFRicKokkosTrans._psyir_intrinsic(name) is intrinsic


def test_builtin_schedule_name_lists_kinds_in_order():
    """The schedule is named for the built-in and its kinds, first use
    first, with the suffix _region_name strips."""
    kernel = SimpleNamespace(name="Real_To_Real_X")

    name = LFRicKokkosTrans._builtin_schedule_name(
        kernel, {"r_def": None, "r_solver": None})

    assert name == "builtin_real_to_real_x_r_def_r_solver_code"
    assert (LFRicKokkosTrans._region_name(SimpleNamespace(name=name))
            == "builtin_real_to_real_x_r_def_r_solver_kokkos")


def test_setval_c_class_is_a_builtin_and_a_schedule_is_synthesised_fresh(
        tmp_path, clear_module_manager_instance):
    """Each call synthesises the schedule again rather than caching it."""
    _, _, kernel = _invoke(tmp_path, "inc", _INC_ALGORITHM, _KERNEL)
    assert LFRicKokkosTrans._is_builtin(kernel)
    assert not LFRicKokkosTrans._is_builtin(LFRicSetvalCKern)

    first = LFRicKokkosTrans._builtin_schedule(kernel)
    second = LFRicKokkosTrans._builtin_schedule(kernel)

    assert first is not second
    assert first.name == second.name == "builtin_inc_x_plus_y_r_def_code"
    assert [symbol.name for symbol in first.symbol_table.argument_list] == [
        "arg1", "arg2"]
    assert first.root.__class__.__name__ == "FileContainer"
    assert first.parent.name == "builtin_inc_x_plus_y_r_def_mod"
