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

"""Tests for the run-time choice of a region's team-scratch level.

Level-0 team scratch is shared memory on a GPU, and Kokkos 4.7 caps it at
the per-block default, 48 KiB on an H100. A region's scratch is column
arrays, so its size grows with ``nlayers``: every region fits at 30 levels,
and at 85 the horizontal FFSL flux regions and ``operator_tri_solve`` do not
(psy-ir-aidev, Phase 8 task G1). The launch therefore asks at run time
whether its request fits a team of one warp in level 0, and moves it to
level 1 -- global memory -- when it does not, generating the launch once
for each level so that each copy names its level as a constant. These tests
assert the text that makes that choice; that it makes the right one on a
card is the model runs' evidence, in psy-ir-aidev's
``validation/device/column-scratch-2026-09-26/``.
"""

from dataclasses import replace

from psyclone.psyir.backend.kokkos import KokkosAlias, KokkosWriter
from psyclone.psyir.backend.kokkos_launch import (
    SCRATCH_PROBE_TYPE, scratch_level_dispatch, scratch_placement,
    scratch_policy, scratch_probe_definition)
from psyclone.psyir.nodes import Loop
from psyclone.tests.psyir.backend.kokkos_member_local_test import (
    _member_only_region)
from psyclone.tests.psyir.backend.kokkos_test import (
    _alias_region, _region, _scratch_region)


def _hierarchical(region):
    """Return the region with its first loop spread over the team.

    :param region: a region with at least one loop.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the same region, taking the hierarchical launch.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    return replace(region, parallel_loops=(region.schedule.walk(Loop)[0],))


def test_scratch_probe_is_defined_only_where_scratch_is():
    """A region with no local array names no probe and defines none."""
    assert not scratch_probe_definition(_region())
    assert SCRATCH_PROBE_TYPE not in KokkosWriter()(_region())

    definition = scratch_probe_definition(_scratch_region())
    assert f"struct {SCRATCH_PROBE_TYPE} {{" in definition
    assert ("KOKKOS_INLINE_FUNCTION void operator()(\n"
            "      const Kokkos::TeamPolicy<>::member_type &) const {}"
            ) in definition


def test_scratch_probe_is_defined_outside_the_region():
    """The probe precedes the region's function, where nvcc accepts it.

    A lambda would have to be defined inside the function, and ``nvcc``
    refuses an extended lambda defined inside another's scope; and the
    region's own functor cannot be the one asked, because it captures the
    level being chosen.
    """
    code = KokkosWriter()(_scratch_region())
    assert code.index(f"struct {SCRATCH_PROBE_TYPE}") < code.index(
        'extern "C" void')


def test_placement_asks_the_static_bound_before_the_device():
    """The static bound short-circuits the probe; only a GPU asks either.

    ``scratch_size_max(0)`` assumes a team of 1024, so a request under it
    fits any team and the device is not asked. A per-thread request is
    asked for a warp's worth of threads, a per-team one for itself.
    """
    thread = scratch_placement("", "PerThread")
    team = scratch_placement("", "PerTeam")

    assert ("      scratch_bytes * TeamPolicy::vector_length_max() "
            "<= size_t(TeamPolicy::scratch_size_max(0))\n") in thread
    assert ("      scratch_bytes <= size_t(TeamPolicy::scratch_size_max(0))\n"
            ) in team
    for text, per in ((thread, "PerThread"), (team, "PerTeam")):
        assert (f".set_scratch_size(0, Kokkos::{per}(scratch_bytes))\n"
                f"                 .team_size_max({SCRATCH_PROBE_TYPE}(), "
                "Kokkos::ParallelForTag())\n"
                "             >= TeamPolicy::vector_length_max();\n"
                "  const int scratch_level = scratch_fits ? 0 : 1;\n") in text
        gpu = text.index(
            "#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)")
        host = text.index("#else\n  const int scratch_level = 0;\n#endif\n")
        assert gpu < text.index("scratch_fits") < host


def test_placement_splits_the_request_between_the_levels():
    """The chosen level gets the bytes and the other gets none."""
    text = scratch_placement("", "PerTeam")
    assert ("  const size_t scratch_bytes_0 = "
            "scratch_level == 0 ? scratch_bytes : 0;\n") in text
    assert ("  const size_t scratch_bytes_1 = "
            "(scratch_level == 1 ? scratch_bytes : 0);\n") in text
    assert "alias_bytes" not in text


def test_placement_keeps_alias_targets_in_level_one():
    """Alias targets are in level 1 whichever level the rest is given."""
    text = scratch_placement("x_new_scratch_t::shmem_size(nlayers)",
                             "PerThread")
    assert ("  const size_t alias_bytes = "
            "x_new_scratch_t::shmem_size(nlayers);\n") in text
    assert ("  const size_t scratch_bytes_1 = "
            "(scratch_level == 1 ? scratch_bytes : 0) + alias_bytes;\n"
            ) in text
    assert text.index("alias_bytes =") < text.index("scratch_bytes_1 =")


def test_scratch_policy_requests_both_levels():
    """Both levels are requested; a level given no bytes allocates nothing."""
    assert scratch_policy("PerTeam") == (
        ".set_scratch_size(0, Kokkos::PerTeam(scratch_bytes_0))\n"
        "          .set_scratch_size(1, Kokkos::PerTeam(scratch_bytes_1))")


def test_flat_launch_chooses_before_the_functor_captures():
    """The level is chosen before the functor that captures it is built.

    A lambda captures by value when it is created, so a choice made after
    ``body`` would not reach it; and the probe policy that sizes the team
    asks for the same two requests the launch does.
    """
    code = KokkosWriter()(_scratch_region())
    assert code.index("const int scratch_level = scratch_fits") < \
        code.index("auto body = KOKKOS_LAMBDA")
    # Each copy of the launch has a probe and a launch, both asking for the
    # same two requests.
    assert code.count(
        ".set_scratch_size(0, Kokkos::PerThread(scratch_bytes_0))") == 4
    assert code.count(
        ".set_scratch_size(1, Kokkos::PerThread(scratch_bytes_1))") == 4
    assert "x_new_scratch_t x_new(team.thread_scratch(0), " \
        "nlayers);" in code


def test_hierarchical_launch_chooses_per_team():
    """The hierarchical launch asks per team, and its team clamp sees it."""
    code = KokkosWriter()(_hierarchical(_scratch_region()))
    assert code.index("const int scratch_level = scratch_fits") < \
        code.index("auto body = KOKKOS_LAMBDA")
    assert ".set_scratch_size(0, Kokkos::PerTeam(scratch_bytes))" in code
    # In each copy, the team clamp's probe and the launch carry the same
    # requests.
    assert code.count(
        ".set_scratch_size(0, Kokkos::PerTeam(scratch_bytes_0))") == 4
    assert code.count(
        ".set_scratch_size(1, Kokkos::PerTeam(scratch_bytes_1))") == 4
    assert "x_new_scratch_t x_new(team.team_scratch(0), " \
        "nlayers);" in code


def test_every_array_an_alias_target_moves_nothing_at_run_time():
    """With every array in level 1 the run-time choice has nothing to move.

    The sum whose level is chosen is ``0``, which the static bound admits,
    so level 0 is asked for nothing and no construction names the chosen
    level.
    """
    code = KokkosWriter()(_hierarchical(_alias_region()))
    assert "const size_t scratch_bytes = 0;" in code
    assert "team_scratch(0)" not in code
    assert code.count("team.team_scratch(1)") == 2


def test_member_local_only_hierarchical_region_asks_for_no_scratch():
    """A region with only member-local arrays launches with no request.

    The probe is still defined, since the region has scratch, but nothing
    names it: there is no team scratch to place.
    """
    code = KokkosWriter()(_member_only_region())
    assert "scratch_level" not in code
    assert "set_scratch_size" not in code
    assert f"struct {SCRATCH_PROBE_TYPE}" in code


def test_member_local_only_flat_region_still_compiles_its_probe():
    """A flat launch names the probe even with nothing to place.

    Its request is ``0``, which the static bound admits, so the probe is
    never asked; but the name is compiled, so it has to be defined.
    """
    code = KokkosWriter()(replace(_member_only_region(), parallel_loops=()))
    assert "const size_t scratch_bytes = 0;" in code
    assert f"team_size_max({SCRATCH_PROBE_TYPE}()," in code
    assert f"struct {SCRATCH_PROBE_TYPE}" in code


def test_an_alias_target_and_a_chosen_array_are_placed_apart():
    """An alias target is in level 1 while its neighbour's level is chosen."""
    region = _alias_region(
        aliases=(KokkosAlias(name="chosen", targets=("y", "x_new")),))
    code = KokkosWriter()(region)
    assert "x_new_scratch_t x_new(team.thread_scratch(1), nlayers);" in code
    assert ("tri_plus_new_scratch_t tri_plus_new("
            "team.thread_scratch(0), nlayers);") in code


def test_dispatch_names_each_level_as_a_constant():
    """Each copy of the launch constructs over a literal level.

    ``nvcc`` emits shared-memory loads only where it can prove a pointer is
    into shared memory; with the level a run-time value it cannot.
    """
    launch = ("  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
              "    x_t x(team.thread_scratch(scratch_level), n);\n"
              "\n"
              "  };\n")
    code = scratch_level_dispatch(launch)
    assert "_scratch(scratch_level)" not in code
    assert code == (
        "  // One copy of the launch for each level, each naming its level\n"
        "  // as a constant: nvcc emits shared-memory loads only where it\n"
        "  // can prove the arrays are in level 0.\n"
        "#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)\n"
        "  if (scratch_level == 1) {\n"
        "    auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        "      x_t x(team.thread_scratch(1), n);\n"
        "\n"
        "    };\n"
        "  } else\n"
        "#endif\n"
        "  {\n"
        "    auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        "      x_t x(team.thread_scratch(0), n);\n"
        "\n"
        "    };\n"
        "  }\n")


def test_dispatch_leaves_a_launch_with_no_chosen_level_alone():
    """A launch whose every array has a fixed level is not duplicated."""
    launch = "    x_t x(team.team_scratch(1), n);\n"
    assert scratch_level_dispatch(launch) == launch


def test_dispatch_keeps_preprocessor_lines_in_column_one():
    """A directive in the launch is not indented with the rest of it."""
    launch = ("#ifdef X\n"
              "    x_t x(team.team_scratch(scratch_level), n);\n"
              "#endif\n")
    code = scratch_level_dispatch(launch)
    assert code.count("\n#ifdef X\n      x_t x(team.team_scratch(") == 2


def test_host_compiles_one_copy_of_every_team_launch():
    """Only the level-0 copy is outside the GPU guard, in all three shapes.

    A host backend never chooses level 1, so compiling that copy there would
    only double the region's build time.
    """
    for region in (_scratch_region(), _hierarchical(_scratch_region()),
                   replace(_hierarchical(_scratch_region()), team_size=4)):
        code = KokkosWriter()(region)
        guard = code.index("  if (scratch_level == 1) {\n")
        host = code.index("  } else\n#endif\n  {\n", guard)
        assert code.count("Kokkos::parallel_for(\"") == 2
        assert code.index("Kokkos::parallel_for(\"", guard) < host
        assert code.index("Kokkos::parallel_for(\"", host) > host
        assert "_scratch(1)" in code[guard:host]
        assert "_scratch(1)" not in code[host:]
        assert "_scratch(0)" in code[host:]
        # The guard opens after the choice the branch reads.
        assert code.rindex("#if defined(KOKKOS_ENABLE_CUDA)", 0, guard) > \
            code.index("const size_t scratch_bytes_1")


def test_an_all_alias_region_launches_once():
    """With no level chosen at run time there is one launch, as before."""
    code = KokkosWriter()(_hierarchical(_alias_region()))
    assert "if (scratch_level == 1)" not in code
    assert code.count("Kokkos::parallel_for(\"") == 1
