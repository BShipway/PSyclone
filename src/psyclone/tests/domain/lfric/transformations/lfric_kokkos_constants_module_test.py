# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the module variables a captured body may read."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _ALLOCATABLE_MODULE_VARIABLE_KERNEL, _LOCAL_ALGORITHM, _LOCAL_KERNEL,
    _MODULE_VARIABLE_KERNEL, _PLANET_CONFIG, _invoke)

from psyclone.configuration import Config
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.parse import ModuleManager
from psyclone.psyir.symbols import ContainerSymbol, ImportInterface, Symbol
from psyclone.psyir.transformations import TransformationError


# The module the kernel above calls into, written out only where a test needs
# PSyclone to have read it. Until it does, `helper` is a plain Symbol; once it
# has, `resolve_type` specialises it to a RoutineSymbol, which is the whole
# difference the refusal turns on.
_HELPER_MODULE = """
module helper_mod
  use constants_mod, only : r_def
  implicit none
contains
  function helper(i) result(scaled)
    integer, intent(in) :: i
    real(kind=r_def) :: scaled
    scaled = real(i, r_def)
  end function helper
end module helper_mod
"""


# A kernel module declaring a variable rather than a constant beside the
# routine. It looks like the constant case at the point the walk meets it --
# module scope, a literal beside the name -- and is not one: without
# `parameter` the value is an initialisation the module may overwrite, so
# writing it into the region would capture a state rather than a constant.
_STATIC_VARIABLE_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  real(kind=r_def) :: cached_tol = 1.0e-9_r_def\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + cached_tol")


# The same module variable, assigned to. The region is handed the value the
# module holds when the launch is made and has no share in the module's
# storage, so the assignment would be lost rather than carried back.
_ASSIGNED_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)",
    "    profile_size = nlayers\n    swept(nlayers) = partial(nlayers)")


# A module array whose extent is named rather than stated. The region sizes
# its own View, and it can only size it from what it can evaluate: `n_profile`
# is not one of its arguments.
_SIZED_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "  private\n",
    "  private\n"
    "  integer(kind=i_def), public :: n_profile = 100\n").replace(
    "  real(kind=r_def), public :: profile_heights(100)",
    "  real(kind=r_def), public :: profile_heights(n_profile)")


# A variable declared inside the routine with an initialiser, which is
# rtheta_bd_kernel_mod's `upwind`. Fortran gives it the SAVE attribute, so it
# is a static of the routine rather than of the module: nothing outside the
# routine declares it, and the PSy layer has no name to pass.
_LOCAL_STATIC_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def) :: visits = 0\n").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + visits")


# A module array of a type that has no place on the ABI. A logical crosses by
# conversion, which is per value, so a logical array has nowhere to go -- the
# same refusal a logical array argument meets, reached by a module variable.
_LOGICAL_MODULE_VARIABLE_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  logical(kind=l_def), public :: profile_active(100)\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (profile_active(k)) swept(k) = swept(k + 1) - partial(k)")


# A name reaching the kernel through a wildcard `use`. The import states no
# name, so PSyIR has no container to resolve the symbol in and no declaration
# to type it from: it is not an import the PSy layer could repeat, and not
# anything either module declares as far as the tree can tell.
_WILDCARD_IMPORT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * recip_epsilon")


@pytest.fixture(name="static_variable_target")
# pylint: disable-next=unused-argument
def static_variable_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares a module variable."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _STATIC_VARIABLE_KERNEL)


@pytest.fixture(name="module_variable_target")
# pylint: disable-next=unused-argument
def module_variable_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads public state of its own module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="assigned_module_variable_target")
# pylint: disable-next=unused-argument
def assigned_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns to state of its own module."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _ASSIGNED_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="allocatable_module_variable_target")
# pylint: disable-next=unused-argument
def allocatable_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an allocatable module array."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _ALLOCATABLE_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="sized_module_variable_target")
# pylint: disable-next=unused-argument
def sized_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a module array of named extent."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _SIZED_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="local_static_target")
# pylint: disable-next=unused-argument
def local_static_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a routine-local static."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOCAL_STATIC_KERNEL)


@pytest.fixture(name="logical_module_variable_target")
# pylint: disable-next=unused-argument
def logical_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a module logical array."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _LOGICAL_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="wildcard_import_target")
# pylint: disable-next=unused-argument
def wildcard_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a name from a wildcard import."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WILDCARD_IMPORT_KERNEL,
        extra={"planet_config_mod": _PLANET_CONFIG})


def test_lfric_kokkos_trans_accepts_a_module_scalar_variable(
        module_variable_target):
    """A public module scalar becomes a formal the PSy layer passes.

    ``profile_size`` is neither a constant nor an argument, so the region
    can neither carry its value nor find it already in the call. It is a
    name the PSy layer can ``use``, though, so the region takes it by value
    and the launch reads it where it is made: what the module holds at
    region entry is what the region sees.
    """
    psy, loop, _ = module_variable_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int profile_size) {" in cpp
    assert "integer(c_int), value :: profile_size" in fortran
    assert ("use column_solve_kernel_mod, only : profile_heights, "
            "profile_size" in fortran)
    assert "loop0_stop, profile_heights, profile_size)" in fortran


def test_lfric_kokkos_trans_accepts_a_module_array_variable(
        module_variable_target):
    """A public module array of literal extents becomes a read-only View.

    An array is state rather than a value, so it crosses by reference as
    every read-only array formal does -- sized from the extents the
    declaration states, and indexed with the origin those extents give.
    """
    psy, loop, _ = module_variable_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const double *profile_heights_data" in cpp
    assert (("auto profile_heights = lfric_kokkos::stage<\n"
             "      Kokkos::View<const double*, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      profile_heights_data, lfric_kokkos::Role::readonly, "
             "100);") in cpp)
    assert "profile_heights((profile_size - 1))" in cpp
    assert ("real(c_double), dimension(*), intent(in) :: profile_heights"
            in fortran)


def test_lfric_kokkos_trans_refuses_an_assigned_module_variable(
        assigned_module_variable_target):
    """A module variable the body writes is refused rather than dropped.

    The region is given the value the module holds when the launch is made
    and has no share in the module's storage, so an assignment inside it
    would be lost. Losing it silently would be a wrong answer rather than an
    unsupported kernel.
    """
    _, loop, _ = assigned_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_size' from 'column_solve_kernel_mod': "
            "the body assigns to it" in str(error.value))


def test_lfric_kokkos_trans_rejects_an_allocatable_module_variable(
        allocatable_module_variable_target):
    """An allocatable module array states no shape to give the region.

    Its extents are set at run time by whatever allocated it, so there is
    nothing for the region to size a View from and nothing the generated
    interface could declare. A deferred shape is a shape with no bounds, so
    it is refused by the reading of the bounds rather than by a check of its
    own.
    """
    _, loop, _ = allocatable_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'profile_heights' to be declared with explicit bounds"
            in str(error.value))


def test_lfric_kokkos_trans_rejects_a_module_array_of_named_extent(
        sized_module_variable_target):
    """A module array sized by a name is refused, not sized by guess.

    The region builds its own View over the pointer it is handed, so the
    extent has to be something it can evaluate. ``n_profile`` is a module
    variable rather than one of the region's arguments, and reading it as
    anything would be inventing a size.
    """
    _, loop, _ = sized_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_heights': the region would have to "
            "size it from n_profile" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_routine_local_static(
        local_static_target):
    """A local with an initialiser is a static of the routine, not the module.

    ``integer(i_def) :: visits = 0`` is rtheta_bd_kernel_mod's ``upwind``
    shape: Fortran gives it the SAVE attribute, so it keeps its value from
    one call to the next and nothing outside the routine declares it. The
    PSy layer has no name to pass and the region has nowhere to keep it.
    """
    _, loop, _ = local_static_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'visits': it is declared in the routine with an "
            "initialiser, which Fortran gives the SAVE attribute"
            in str(error.value))


def test_lfric_kokkos_trans_refuses_a_module_logical_array(
        logical_module_variable_target):
    """A module array off the ABI is refused as an argument would be.

    A logical crosses by conversion, which is per value; an array crosses by
    reference, and a ``View<bool*>`` over ``logical(l_def)`` storage would
    reinterpret its elements rather than convert them. Coming from a module
    rather than from the argument list changes none of that.
    """
    _, loop, _ = logical_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_active' from 'column_solve_kernel_mod'"
            in str(error.value))
    assert "arrays of them have a place on the generated C ABI" in str(
        error.value)


def test_lfric_kokkos_trans_refuses_a_wildcard_imported_name(
        wildcard_import_target):
    """A name from a wildcard ``use`` is not an import the region can repeat.

    ``use planet_config_mod`` states no names, so the symbol carries no
    container to resolve a kind in and no declaration to read a shape from.
    It is not the kernel's own module's either, so there is nothing to
    describe and the refusal says so.
    """
    _, loop, _ = wildcard_import_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'recip_epsilon': it is neither a kernel argument "
            "nor imported from a module" in str(error.value))


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_refuses_a_resolved_routine_as_a_constant(
        tmp_path, clear_module_manager_instance):
    """A symbol that resolves to a routine is refused as one.

    ``_constants`` skips a call's own routine reference, but a name used
    another way -- as a procedure argument, say -- reaches
    ``_describe_constant``, and only becomes known to be a routine when
    ``resolve_type`` specialises it. Reading the module is what makes the
    difference, so this test provides one.
    """
    (tmp_path / "helper_mod.f90").write_text(_HELPER_MODULE)
    Config.get().include_paths = [str(tmp_path)]
    ModuleManager.get().add_search_path(str(tmp_path))
    symbol = Symbol(
        "helper", interface=ImportInterface(ContainerSymbol("helper_mod")))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._describe_constant(symbol)

    assert ("cannot capture 'helper' from 'helper_mod': it is a routine "
            "rather than data" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_private_module_variable(
        static_variable_target):
    """A module variable the module keeps to itself cannot be passed.

    ``real(kind=r_def) :: cached_tol = 1.0e-9_r_def`` is state rather than a
    constant however it was initialised, so the region has to be given its
    value rather than write the initialisation in. A kernel module is
    ``private`` by default, though, and this one does not name ``cached_tol``
    in its ``public`` list: there is no name for the PSy layer to import.
    """
    _, loop, _ = static_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'cached_tol' from 'column_solve_kernel_mod': the "
            "module declares it private" in str(error.value))
