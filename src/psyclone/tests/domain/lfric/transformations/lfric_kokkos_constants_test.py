# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosConstantsMixin: declared module constants."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _LOCAL_ALGORITHM, _LOCAL_KERNEL, _PLANET_CONFIG, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import (
    ArrayConstructor, IntrinsicCall, Literal, Reference)
from psyclone.psyir.symbols import (
    ArrayType, ContainerSymbol, DataSymbol, ImportInterface, ScalarType,
    StaticInterface, Symbol, SymbolError, UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError


# The same kernel declaring its own constants beside the routine, the way
# poly1d_reconstruction and create_w2mask do. The module is `private`, so the
# PSy layer could not import either name even if it wanted to; the value is
# what reaches the region.
_STATIC_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: nfaces = 4\n"
    "  real(kind=r_def), parameter :: tol = 1.0e-9_r_def\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * nfaces + tol")


# The same, with the constant declared as an array. `face_order(4)` is
# gungho's shape: a parameter array of a kernel's own module, indexed by a
# value the loop computes, so there is no single element to substitute. The
# module is private, as a kernel module is, which is why the values go into
# the generated unit rather than across the ABI.
_ARRAY_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: face_order(4) = [1, 2, 3, 4]\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(face_order(k)) "
    "* face_order(1)")


# An array parameter whose values are not there to be read: `reshape` is a
# call, so there is no element list to fold and nothing to declare.
_RESHAPED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: point(2, 2) = &\n"
    "      reshape((/ 1, 2, 3, 4 /), (/ 2, 2 /))\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(point(1, 1))")


# A scalar parameter declared as an arithmetic over other parameters, one of
# them from another module. None of the names is data the region could read,
# but the value they state is one it can carry.
_FOLDED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : n_moist").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: n_extra = 2\n"
    "  real(kind=r_def), parameter :: weight = &\n"
    "      1.0_r_def / (n_moist + n_extra)\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - weight * partial(k)")


# A kernel spelling a kind in its body rather than only in its declarations.
# `real(x, r_def)` puts r_def into the tree as a Reference, which is not data
# the PSy layer passes by value: the cast consumes it.
_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) + real(k, r_def)")


# The same, casting to a kind no declaration in the body repeats. The width
# has to come from the cast argument or the region silently computes at the C
# writer's default instead of at the width the Fortran asked for.
_UNDECLARED_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_second").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) + real(k, r_second)")


# A kernel importing a logical constant. It was named for being off the ABI,
# which stage 5 made false: a logical scalar now crosses by conversion, and an
# imported constant crosses as an argument rather than a literal because its
# value is known only where the PSy layer runs. The name is kept so that the
# fixture's history is legible against the plans that refer to it.
_OFF_ABI_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : rehabilitate").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (rehabilitate) swept(k) = swept(k + 1) - partial(k)")


# A kernel importing a module constant declared with no kind. PROTECTED leaves
# the frontend with the declaration text rather than a typed symbol, so this
# reaches the ABI by the declaration reader rather than by the symbol's own
# type -- the second of the two routes an unkinded declaration can arrive by,
# and the one that would otherwise still refuse it.
_DEFAULT_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : quenching").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (quenching) swept(k) = swept(k + 1) - partial(k)")


# A kernel importing a module datum whose kind the ABI does not carry.
# 'unmapped_width' is an r_quad real, 16 bytes and so off a C ABI carrying 4-
# and 8-byte ones, and it is declared with no attributes so that PSyIR models
# it rather
# than leaving the text to be re-read. Stage 5 needs this because the case used
# to be carried by an l_def logical constant, which is now admitted: the
# refusal is unchanged, so it keeps a witness.
_UNMAPPED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : unmapped_width").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * unmapped_width")


# A kernel importing a constant from a module PSyclone cannot read. Its kind
# is stated only there, so there is no width to put on the ABI and no honest
# guess to make -- the largest single cause left in the survey's residue.
_UNREADABLE_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use unreadable_constants_mod, only : eps").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + eps")


# An array parameter of a type the generated unit has no declaration for. A
# logical crosses the ABI by conversion, which is per value, so an array of
# them has no C type -- and unlike the cases above the values are perfectly
# readable, which is why this refusal is separate from being unable to fold.
_LOGICAL_ARRAY_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  logical(kind=l_def), parameter :: sweep(4) = &\n"
    "      [.true., .false., .true., .false.]\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (sweep(k)) swept(k) = swept(k + 1) - partial(k)")


@pytest.fixture(name="default_constant_target")
# pylint: disable-next=unused-argument
def default_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an unkinded module constant."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _DEFAULT_CONSTANT_KERNEL)


@pytest.fixture(name="static_constant_target")
# pylint: disable-next=unused-argument
def static_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares its own constants."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _STATIC_CONSTANT_KERNEL)


@pytest.fixture(name="array_constant_target")
# pylint: disable-next=unused-argument
def array_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares an array parameter."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARRAY_CONSTANT_KERNEL)


@pytest.fixture(name="cast_kind_target")
# pylint: disable-next=unused-argument
def cast_kind_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel body names a kind in a cast."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CAST_KIND_KERNEL)


@pytest.fixture(name="undeclared_cast_kind_target")
# pylint: disable-next=unused-argument
def undeclared_cast_kind_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke casting to a kind no declaration in the body uses."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNDECLARED_CAST_KIND_KERNEL)


@pytest.fixture(name="off_abi_constant_target")
# pylint: disable-next=unused-argument
def off_abi_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke importing a constant of a kind off the C ABI."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _OFF_ABI_CONSTANT_KERNEL)


@pytest.fixture(name="unmapped_constant_target")
# pylint: disable-next=unused-argument
def unmapped_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke importing a datum of a kind the ABI does not map."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNMAPPED_CONSTANT_KERNEL)


@pytest.fixture(name="unreadable_constant_target")
# pylint: disable-next=unused-argument
def unreadable_constant_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke importing a constant from an unreadable module."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNREADABLE_CONSTANT_KERNEL)


@pytest.fixture(name="logical_array_constant_target")
# pylint: disable-next=unused-argument
def logical_array_constant_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a logical array parameter."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _LOGICAL_ARRAY_CONSTANT_KERNEL)


@pytest.fixture(name="reshaped_constant_target")
# pylint: disable-next=unused-argument
def reshaped_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a reshaped parameter array."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RESHAPED_CONSTANT_KERNEL)


@pytest.fixture(name="folded_constant_target")
# pylint: disable-next=unused-argument
def folded_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a parameter as an expression."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _FOLDED_CONSTANT_KERNEL,
        extra={"planet_config_mod": _PLANET_CONFIG})


def test_lfric_kokkos_trans_accepts_an_unkinded_module_constant(
        default_constant_target):
    """The declaration reader admits an unkinded constant on the same terms.

    A module constant carrying PROTECTED leaves the frontend with the original
    text rather than a typed symbol, so its kind is recovered by reading the
    declaration. That reader has to answer the unkinded case the same way the
    typed path does, or the contract would depend on which attributes a
    constant happens to carry: 'quenching' is a plain logical, and it crosses
    by the same conversion 'rehabilitate' does.
    """
    psy, loop, _ = default_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool quenching" in cpp
    assert "if (quenching)" in cpp
    assert "LOGICAL(quenching, kind=c_bool)" in fortran


def test_lfric_kokkos_trans_places_a_local_sized_by_a_named_constant(
        named_constant_local_target):
    """``parameter :: nfaces = 4`` sizing a local is read from its value.

    This is the commonest kernel-local shape GungHo has, and before the
    extent was resolved it was refused: ``nfaces`` is not a kernel argument,
    so the launch was told it could not compute the size. It does not have
    to. The value is stated by the declaration standing beside the array, so
    it is substituted into the extent exactly as the same value is already
    substituted into the body wherever the kernel names it.

    The constant is a routine-local ``parameter``, which is the shape the
    kernels use, rather than one of the module's own; both reach the same
    resolution, and the module case is covered by
    ``test_lfric_kokkos_trans_folds_a_static_constant``.
    """
    psy, loop, kernel = named_constant_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("v_dot_n")

    assert LFRicKokkosTrans._extents(symbol) == ("4",)
    # Nothing is left for the launch to be asked for.
    assert LFRicKokkosTrans._extent_names(symbol) == set()

    cpp = LFRicKokkosTrans().apply(loop)

    assert "v_dot_n_scratch_t::shmem_size(4)" in cpp
    assert "v_dot_n_scratch_t v_dot_n(team.team_scratch(0), 4);" in cpp
    # The name is nowhere in the generated unit: not in the extent, not in
    # the body, and not across the interface.
    assert "nfaces" not in cpp
    assert "nfaces" not in str(psy.gen)


def test_lfric_kokkos_trans_writes_a_declared_constant_as_its_value(
        static_constant_target):
    """A module-level parameter reaches the region as its literal.

    It has to: the kernel module is ``private`` and publishes only its
    metadata and its ``_code`` routine, so ``use column_solve_kernel_mod,
    only: nfaces`` in the PSy layer would not compile. The value is stated in
    the declaration, so the region carries the value.
    """
    _, loop, _ = static_constant_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "partial((k - 1)) * 4" in cpp
    assert "+ 1.0e-9" in cpp
    assert "nfaces" not in cpp
    assert "tol" not in cpp


def test_lfric_kokkos_trans_does_not_import_a_declared_constant(
        static_constant_target):
    """The PSy layer gains no import for a constant written in as a value."""
    psy, loop, _ = static_constant_target

    LFRicKokkosTrans().apply(loop)

    generated = str(psy.gen)
    assert "nfaces" not in generated
    assert "tol" not in generated


def test_lfric_kokkos_trans_accepts_an_array_parameter(array_constant_target):
    """An array parameter is declared in the unit, not passed to it.

    Its values are in the Fortran and its module is private, so there is
    nothing for the PSy layer to import and no argument worth adding: the
    generated unit declares it itself, among the body's locals rather than at
    file scope so that a device compiler can read it. It is indexed as C
    indexes an array, and with the same origin removed that a View subscript
    has.
    """
    psy, loop, _ = array_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int face_order[4] = {1, 2, 3, 4};" in cpp
    assert "static const" not in cpp
    assert cpp.count("const int face_order") == 1
    assert "partial((face_order[(k - 1)] - 1))" in cpp
    assert "face_order[(1 - 1)]" in cpp
    assert "face_order" not in fortran


def test_lfric_kokkos_trans_refuses_a_reshaped_array_parameter(
        reshaped_constant_target):
    """An array parameter built by a call states no values to declare.

    ``reshape`` is evaluated by the compiler, not by PSyIR, so there is no
    element list to write into the generated unit -- and a rank above one
    would not be indexed in C by the subscripts the body writes in any case.
    """
    _, loop, _ = reshaped_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'point': a module-level constant is written "
            "into the region as its value, and this one was not declared "
            "with a literal value" in str(error.value))


def test_lfric_kokkos_trans_accepts_a_folded_parameter_expression(
        folded_constant_target):
    """A parameter declared as an expression is carried as that expression.

    ``weight = 1.0_r_def / (n_moist + n_extra)`` names two other
    parameters, one of them another module's. None of the three names is
    anything the region could read, but between them they state a value it
    can carry, so each is replaced by what it was declared as and the
    arithmetic is left for the C++ compiler to do exactly as the Fortran
    compiler would have.
    """
    psy, loop, _ = folded_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen)

    assert "(1.0 / (3 + 2))" in cpp
    assert "weight" not in cpp
    assert "n_moist" not in generated


def test_lfric_kokkos_trans_refuses_a_logical_array_parameter(
        logical_array_constant_target):
    """An array parameter with no C type is refused, values or no values.

    ``[.true., .false., .true., .false.]`` states its elements perfectly
    well. What it has no answer for is the type of the declaration they
    would go into: a logical is on the ABI by conversion, which is per
    value, and an array of them is no more declarable in the generated body
    than it is passable.
    """
    _, loop, _ = logical_array_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot carry 'sweep': only" in str(error.value))
    assert "arrays have a place in the generated translation unit" in str(
        error.value)


def test_lfric_kokkos_trans_reads_no_value_it_cannot_resolve(monkeypatch):
    """A constant whose module cannot be read contributes no value.

    ``_constant_value`` answers ``None`` rather than raising, because the
    caller that needs the symbol on the ABI is the one that reports the
    missing module by name -- and it reports it once, rather than every
    reference to it reporting it again.
    """
    symbol = DataSymbol(
        "eps", ScalarType(ScalarType.Intrinsic.REAL,
                          ScalarType.Precision.UNDEFINED),
        interface=ImportInterface(ContainerSymbol("nowhere_mod")))
    monkeypatch.setattr(
        DataSymbol, "resolve_type",
        lambda self: (_ for _ in ()).throw(SymbolError("no such module")))

    assert LFRicKokkosTrans._constant_value(symbol) is None


@pytest.mark.parametrize("declaration", [
    "REAL(KIND = r_def), DIMENSION(3), PUBLIC :: coefficients",
    "TYPE(field_type), PUBLIC :: state",
])
def test_lfric_kokkos_trans_reads_no_c_type_from_a_shape_or_a_type(
        declaration):
    """Only a scalar of an intrinsic type is read out of a declaration.

    The declaration text is the last resort for a symbol PSyIR could not
    model, and it is read for a width to pass a *value* at. A declaration
    carrying a shape is not one value, and one naming a derived type is not
    a width, so both are answered ``None`` rather than parsed further.
    """
    symbol = DataSymbol(
        declaration.split("::")[1].strip(),
        UnsupportedFortranType(declaration))

    assert LFRicKokkosTrans._declared_c_type(symbol) is None


def _integer_literal(value):
    """Return one integer literal for the folding tests.

    :param int value: the value the literal states.

    :returns: the literal.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    return Literal(str(value), ScalarType(
        ScalarType.Intrinsic.INTEGER, ScalarType.Precision.UNDEFINED))


def test_lfric_kokkos_trans_folds_a_reference_to_a_constant():
    """Folding a bare reference replaces the whole expression.

    Everything else is replaced inside the copy being folded, but an
    expression that *is* a reference has no parent to be replaced in, so the
    value becomes the result rather than being written into it.
    """
    symbol = DataSymbol(
        "nfaces", ScalarType(ScalarType.Intrinsic.INTEGER,
                             ScalarType.Precision.UNDEFINED),
        is_constant=True, initial_value=_integer_literal(4),
        interface=StaticInterface())

    folded = LFRicKokkosTrans._fold(Reference(symbol))

    assert isinstance(folded, Literal)
    assert folded.value == "4"


def test_lfric_kokkos_trans_folds_nothing_that_is_not_arithmetic():
    """An expression holding a node the region could not carry is not folded.

    A subscript-free substitution can put arithmetic over names and literals
    into the body and have it mean the same thing. It cannot do that for a
    call: what the compiler would have evaluated is not there to evaluate.
    """
    expression = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ABS, [_integer_literal(4)])

    assert LFRicKokkosTrans._fold(expression) is None


def test_lfric_kokkos_trans_folds_nothing_that_names_a_variable():
    """A name with no declared value stops the fold rather than surviving it.

    The region has no ``count`` to read, so an expression naming one states
    no value however much arithmetic surrounds it.
    """
    symbol = DataSymbol(
        "count", ScalarType(ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.UNDEFINED),
        interface=StaticInterface())

    assert LFRicKokkosTrans._fold(Reference(symbol)) is None


def test_lfric_kokkos_trans_reads_no_array_from_an_unresolvable_import():
    """A constant whose module cannot be read contributes no value.

    ``_constant_value`` answers ``None`` rather than raising, because the
    caller that needs the symbol on the ABI is the one that reports the
    missing module by name.
    """
    symbol = DataSymbol(
        "eps", ScalarType(ScalarType.Intrinsic.REAL,
                          ScalarType.Precision.UNDEFINED),
        interface=ImportInterface(ContainerSymbol("nowhere_mod")))

    assert LFRicKokkosTrans._constant_value(symbol) is None


def _array_parameter(initial_value):
    """Return one ``parameter`` array symbol with the value given.

    :param initial_value: the initialiser the declaration carries.
    :type initial_value: :py:class:`psyclone.psyir.nodes.DataNode`

    :returns: the symbol.
    :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
    """
    integer = ScalarType(
        ScalarType.Intrinsic.INTEGER, ScalarType.Precision.UNDEFINED)
    return DataSymbol(
        "face_order", ArrayType(integer, [4]), is_constant=True,
        initial_value=initial_value, interface=StaticInterface())


def _absolute_four():
    """Return ``ABS(4)``, a value the region cannot carry.

    :returns: the call.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    return IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ABS, [_integer_literal(4)])


@pytest.mark.parametrize("symbol", [
    Symbol("face_order"),
    _array_parameter(_absolute_four()),
    _array_parameter(ArrayConstructor.create([_absolute_four()])),
])
def test_lfric_kokkos_trans_declares_no_array_it_cannot_read(symbol):
    """Only an array whose every element states a value is declared.

    A symbol PSyIR never specialised has no declaration to read at all; one
    initialised by a call rather than by a constructor has no element list
    to read; and one whose constructor holds a call has an element that
    states no value. None of the three has values to write into the
    generated unit, and each is answered ``None`` rather than
    half-declared.
    """
    assert LFRicKokkosTrans._constant_array(symbol) is None


def test_lfric_kokkos_trans_casts_at_the_kind_the_body_names(cast_kind_target):
    """A kind spelled in a cast is a type name, not data to pass by value.

    ``real(k, r_def)`` puts ``r_def`` into the tree as a Reference like any
    other. The writer consumes it as the cast's target and never emits it, so
    there is nothing for the PSy layer to import or pass.
    """
    _, loop, _ = cast_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(double)k" in cpp
    assert "r_def" not in cpp


def test_lfric_kokkos_trans_casts_at_a_kind_no_declaration_repeats(
        undeclared_cast_kind_target):
    """A cast's own kind sets its width even when nothing else names it.

    ``r_second`` is 8 bytes and appears only as this cast's target. Reading
    the width from declarations alone would leave the backend with no entry
    for it, and the C writer's default would silently make it ``float``.
    """
    _, loop, _ = undeclared_cast_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(double)k" in cpp
    assert "(float)k" not in cpp


def test_lfric_kokkos_trans_accepts_a_protected_logical_constant(
        off_abi_constant_target):
    """An imported logical constant crosses by the same conversion.

    ``rehabilitate`` is declared as gungho's configuration modules declare
    every namelist logical: ``logical(l_def), public, protected``, its kind
    stated positionally and its value assigned by the namelist reader rather
    than by a ``parameter``. PSyIR models none of that, so the type is read
    back out of the declaration text.

    It reaches the region as an argument rather than a literal, because its
    value is only known where the PSy layer runs, so the same wrapping applies
    to it as to a formal -- which is the point of doing the wrapping over
    ``region.arguments`` rather than over the formals alone.
    """
    psy, loop, _ = off_abi_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool rehabilitate" in cpp
    assert "if (rehabilitate)" in cpp
    assert "use planet_config_mod, only : rehabilitate" in fortran
    assert "LOGICAL(rehabilitate, kind=c_bool)" in fortran


def test_lfric_kokkos_trans_refuses_a_constant_of_a_kind_off_the_abi(
        unmapped_constant_target):
    """A constant that resolves can still have no place on the interface.

    This case used to be carried by ``rehabilitate``, an ``l_def`` logical,
    which stage 5 admits. The refusal itself did not change, so it keeps a
    witness of a kind that is still off the ABI: ``unmapped_width`` is an
    ``r_quad`` real, 16 bytes where the ABI carries 4 and 8. The
    message names the kinds the ABI does carry, which now includes the
    logical clause and the unkinded one.
    """
    _, loop, _ = unmapped_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot pass 'unmapped_width' from 'planet_config_mod' by value"
            in str(error.value))
    assert "4-byte integer" in str(error.value)
    assert "logical of any kind" in str(error.value)
    assert "integer or logical declared with no kind" in str(error.value)


def test_lfric_kokkos_trans_names_the_module_it_could_not_read(
        unreadable_constant_target):
    """An unreadable container is named rather than guessed around.

    The kind of an imported constant is stated only in its own module, so a
    module PSyclone cannot read leaves no width for the ABI. Choosing one
    would put a silently wrong type on the interface, so the refusal says
    which module is missing and leaves the search path to the caller.
    """
    _, loop, _ = unreadable_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot type 'eps' without the source of "
            "'unreadable_constants_mod'" in str(error.value))
    assert "module search path" in str(error.value)
