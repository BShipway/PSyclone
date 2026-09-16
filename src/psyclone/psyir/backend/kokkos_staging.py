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

"""Staging: how a region's array arguments reach the device it runs on.

A generated region is handed raw pointers by the Fortran that calls it, and
until now it wrapped each of them in an unmanaged ``Kokkos::View`` and ran.
That is correct exactly when the storage behind the pointer is reachable from
the execution space -- true on a CPU build, true on a GPU only for storage
allocated in a space the device can read. LFRic field data and an LMA
operator's local stencil are allocated in ``SharedSpace``; a dofmap, a basis
table, a quadrature weight, a stencil or colour map and a columnwise
operator's banded matrix are not.

This module holds the text of the header the generated regions use to close
that gap, ``lfric_kokkos_staging.hpp``, and the spellings the writer emits to
call into it. The header reads ``LFRIC_KOKKOS_STAGING`` once per process and
offers three modes:

``none``
    An unmanaged View over the caller's pointer -- what a region did before
    this existed, and the default when the variable is unset, so no build and
    no gate moves unless it is asked to.

``all``
    Every array is allocated in the execution space's memory space and copied
    in on entry; a written one is copied back after the fence. Correct on any
    device whatever space the caller allocated in, and slow enough that
    timings taken in it say nothing about the prototype.

``non-field``
    A ``field`` view is left unmanaged over the caller's pointer, because
    LFRic claims field data and an LMA operator's local stencil from
    ``SharedSpace``. A ``readonly`` array is copied
    once and cached by ``(pointer, bytes)``, because a dofmap or a map is
    allocated once and immutable for the run. A ``readwrite`` array, and a
    ``transient`` one whose storage the PSy layer allocates and frees around
    the invoke, is staged per call: neither may be keyed by an address.
    ``LFRIC_KOKKOS_STAGING_CACHE=0`` turns the cache off and copies on every
    call.

What the header did is reported to stderr at ``Kokkos::finalize``: a line of
totals, and then a line per role carrying that role's arrays, copies, bytes
copied each way and allocations made and released. The split is the
measurement any decision about staging rests on -- a total cannot say whether
a run's copying is a dofmap copied once or a basis table copied on every
call -- and ``psy-ir-aidev``'s ``bin/measure-timestep`` reads both lines into
one row per run.

The *role* -- which of those kinds an argument is -- is decided by the
LFRic transformation that knows what the argument means, never here and never
by the writer: a C++ writer sees a ``double *`` and cannot tell a field from a
basis table.

Nothing in this module imports PSyclone. That is deliberate: the shell gates
that compile the header load this file directly and print
:py:func:`header_text`, so no gate carries a pasted copy that can drift from
what the regions are generated against.

"""

#: The roles a View may carry, in the order the header declares them.
ROLES = ("field", "readonly", "readwrite", "transient")

#: The name the header is written under, beside the regions that include it.
HEADER_NAME = "lfric_kokkos_staging.hpp"

_HEADER_TEXT = r'''
// ---------------------------------------------------------------------
// Generated by PSyclone: psyclone.psyir.backend.kokkos_staging.
// Do not edit. Every generated Kokkos region includes this header, and a
// region compiled against a different copy of it is a silent mismatch.
// -----------------------------------------------------------------------
//
// Staging for generated LFRic Kokkos regions.
//
// A region is handed raw pointers. Whether the storage behind one is
// reachable from the execution space depends on where the caller allocated
// it, so each View is obtained through stage() and each written View is
// released through unstage() after the region's fence. The mode is read
// once, from LFRIC_KOKKOS_STAGING:
//
//   none       an unmanaged View over the caller's pointer (the default)
//   all        every array copied into and out of the execution space
//   non-field  fields and LMA operator stencils unmanaged (LFRic claims
//              both from SharedSpace), every other read-only array copied
//              once and cached, every other written array staged per call
//
// LFRIC_KOKKOS_STAGING_CACHE=0 disables the cache in non-field mode.
//
// LFRIC_KOKKOS_STAGING_PREFETCH=1 asks, on a CUDA build, for a field's pages
// to be moved to the card before the launch that reads them rather than
// faulted across one at a time by the launch itself; it is off unless asked
// for. LFRIC_KOKKOS_STAGING_PREFETCH_STRIDE=N asks for a pointer at most
// once in every N field stagings, for a field that is already resident.
//
// At Kokkos::finalize the header reports what it did on stderr: one line of
// totals, then one line per role (staging_role=field, readonly, readwrite,
// transient) carrying that role's arrays, copies, bytes each way and
// allocations. A total cannot say which kind of argument the copying was
// for, and the four roles are copied on entirely different schedules.
//
// stage() returns the View type it is given, so a launch body compiled
// against one mode is the text it is compiled against in every other: only
// the memory the View covers changes.

#ifndef LFRIC_KOKKOS_STAGING_HPP
#define LFRIC_KOKKOS_STAGING_HPP

#include <Kokkos_Core.hpp>

// After Kokkos, which is what defines KOKKOS_ENABLE_CUDA, so that this
// header says the same thing wherever a region includes it.
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace lfric_kokkos {

// What an argument is, as the LFRic transformation that built the region
// knows it. A C++ writer cannot tell these apart; the role is the answer
// carried down from where the question could be answered.
enum class Role {
  field,      // field data, and an LMA operator's local stencil: LFRic
              // claims both from SharedSpace through one registry, so the
              // caller's storage is what the region reads and writes
  readonly,   // dofmap, stencil or colour map, mesh property, module array:
              // read-only storage a long-lived LFRic object owns, so a copy
              // of it keyed by its address stays good for the run
  readwrite,  // a columnwise operator's banded matrix, caller-supplied
              // scratch: storage a region writes that is not in SharedSpace
  transient   // a basis or differential-basis table, quadrature weights:
              // read-only for the call and gone after the invoke that made
              // it, so its address says nothing about its contents and a
              // copy of it is taken afresh every call
};

// The roles in the order they are declared, so that a report reads in that
// order however it is gathered.
constexpr int role_count = 4;

inline const char *role_name(Role role) {
  switch (role) {
  case Role::field:
    return "field";
  case Role::readonly:
    return "readonly";
  case Role::readwrite:
    return "readwrite";
  default:
    return "transient";
  }
}

enum class Mode { none, all, non_field };

inline const char *mode_name(Mode setting) {
  switch (setting) {
  case Mode::all:
    return "all";
  case Mode::non_field:
    return "non-field";
  default:
    return "none";
  }
}

inline Mode read_mode() {
  const char *setting = std::getenv("LFRIC_KOKKOS_STAGING");
  if (setting == nullptr || setting[0] == '\0' ||
      std::strcmp(setting, "none") == 0) {
    return Mode::none;
  }
  if (std::strcmp(setting, "all") == 0) {
    // Once, on first use. This mode copies every array in and out on every
    // region call: it is here to be correct on a device whose spaces are
    // not shared, and a timing taken in it measures the copies.
    std::fprintf(stderr,
                 "lfric_kokkos: LFRIC_KOKKOS_STAGING=all stages every array "
                 "into and out of the execution space on every region call. "
                 "It is a correctness mode; timings taken in it mean "
                 "nothing.\n");
    return Mode::all;
  }
  if (std::strcmp(setting, "non-field") == 0) {
    return Mode::non_field;
  }
  std::fprintf(stderr,
               "lfric_kokkos: LFRIC_KOKKOS_STAGING=%s is not one of none, "
               "all, non-field\n",
               setting);
  Kokkos::abort("lfric_kokkos: unknown LFRIC_KOKKOS_STAGING setting");
  return Mode::none;
}

inline Mode mode() {
  static const Mode setting = read_mode();
  return setting;
}

inline bool caching() {
  static const bool setting = []() {
    const char *value = std::getenv("LFRIC_KOKKOS_STAGING_CACHE");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  return setting;
}

// The buffer pool, on unless LFRIC_KOKKOS_STAGING_POOL=0. A transient array
// is copied afresh on every region call, so its buffer is allocated and
// released on every call too: at C48 that is 136 allocations and 136
// releases a timestep, nine tenths of every allocation the staging header
// makes. The contents cannot be reused -- the address says nothing about
// them -- but the *storage* can, so a released buffer is kept and handed to
// the next argument of the same size instead of being given back to the
// driver. Nothing about what is copied changes; only who owns the bytes.
inline bool pooling() {
  static const bool setting = []() {
    const char *value = std::getenv("LFRIC_KOKKOS_STAGING_POOL");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  return setting;
}

// How much unused storage the pool may hold, in megabytes. A bound rather
// than none because the pool keys on an exact byte count: a run meeting many
// distinct sizes would otherwise keep a spare of each for ever. Past the
// bound a released buffer goes back to the driver as it did before.
inline std::size_t pool_limit() {
  static const std::size_t bytes = []() {
    const char *value = std::getenv("LFRIC_KOKKOS_STAGING_POOL_MB");
    const long megabytes = value == nullptr ? 256 : std::atol(value);
    return megabytes <= 0 ? std::size_t(0)
                          : std::size_t(megabytes) * 1024 * 1024;
  }();
  return bytes;
}

// PREFETCH ON STAGE. In non-field mode -- and in none mode, which reaches
// the same line -- a Role::field View is left over the caller's pointer,
// which on a CUDA build is managed memory whose pages live wherever they
// were last touched: a launch reading a field the host has just written
// faults it onto the card 64 KB at a time (at C48, 525 thousand GPU faults
// in ten steps and 5.5 s of the 12.3 s of kernel time, nearly all of it in
// the captured built-ins). Asking the driver for the whole range first turns
// those faults into one bulk copy, issued on the stream the launch that
// follows will run on, so the region's own fence orders it and nothing new
// is waited for.
//
// Off unless asked for: a prefetch of a field that is already resident costs
// the call and moves nothing, and which way that goes is a measurement.
inline bool prefetching() {
  static const bool setting = []() {
    const char *value = std::getenv("LFRIC_KOKKOS_STAGING_PREFETCH");
    return value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0;
  }();
  return setting;
}

// How rarely one pointer may be prefetched again, counted in field stagings
// of any pointer; zero, the default, prefetches on every staging. This is
// the honest form of "once a timestep": a header included by regions sees no
// step boundary -- every region ends in the same fence, and the Fortran that
// knows the step never calls in here -- so a count of stagings is its only
// clock. The field set is the same every step, so N stagings is a fixed
// fraction of a step, and field_stages on the report line divided by the
// step count is what to set N from.
inline std::size_t prefetch_stride() {
  static const std::size_t stride = []() {
    const char *value = std::getenv("LFRIC_KOKKOS_STAGING_PREFETCH_STRIDE");
    const long count = value == nullptr ? 0 : std::atol(value);
    return count <= 0 ? std::size_t(0) : std::size_t(count);
  }();
  return stride;
}

// A device allocation and the handle that owns it. The owner is a
// shared_ptr to a managed Kokkos::View: holding the View is what keeps the
// allocation alive, and dropping the last owner is what frees it.
struct State;

struct Block {
  std::shared_ptr<void> owner;
  void *data = nullptr;
  std::size_t bytes = 0;
  int references = 0;
  Role role = Role::readonly;
  // Which memory space's pool this block belongs to, as the one thing about
  // the space that survives the template it was allocated in: the release
  // happens in release() or at finalise, where the value type and the space
  // are no longer in hand.
  void (*recycle)(State &, Block &) = nullptr;
};

// An allocation nobody is using: the owner whose lifetime is the
// allocation's, and the address it covers. Held by exact byte count, so the
// next argument of that size takes it instead of asking the driver.
struct Spare {
  std::shared_ptr<void> owner;
  void *data = nullptr;
};

// Keyed by byte count alone, and not by element type: a Kokkos allocation is
// aligned to at least the alignment of any scalar in every space this header
// stages into -- CudaSpace goes through cudaMalloc, which gives 256 bytes;
// HostSpace aligns to 64 -- so a buffer taken for doubles serves ints and the
// other way about. What it may not do is serve a different *size*, which is
// why the key is exact rather than a best fit.
using Spares = std::map<std::size_t, std::vector<Spare>>;

// Host pointer and byte count. The byte count is in the key so that two
// arguments over the same base pointer with different lengths -- a field
// vector's components, say -- are not mistaken for one another.
using Key = std::pair<const void *, std::size_t>;

// What staging did, kept once per role. Totals say how much copying a run
// does; only the split by role says which kind of argument it is for, and
// the four kinds are copied on entirely different schedules -- a field never,
// a dofmap once for the run, a transient table on every call. Bytes are
// counted beside calls because a call count says nothing about the bus, and
// allocations beside bytes because an allocator call costs whatever its size.
struct Counters {
  std::size_t staged = 0;      // arrays staged for the duration of one call
  std::size_t cached = 0;      // arrays copied once and kept for the run
  std::size_t hits = 0;        // calls served from the cache
  std::size_t shared = 0;      // calls served by another argument's staging
  std::size_t copies_in = 0;
  std::size_t copies_out = 0;
  std::size_t bytes_in = 0;
  std::size_t bytes_out = 0;
  std::size_t allocations = 0;  // allocations made in the execution space
  std::size_t frees = 0;        // those allocations given back to the driver
  std::size_t reuses = 0;       // buffers taken from the pool instead
  std::size_t recycled = 0;     // buffers returned to the pool instead
};

struct State {
  std::mutex lock;
  std::map<Key, Block> cache;  // read-only non-field, kept for the run
  std::map<Key, Block> live;   // staged for this region call
  Counters role[role_count];
  // One pool per memory space met, registered here when it is first used so
  // that the finalize hook can empty every one of them before Kokkos goes.
  std::vector<Spares *> pools;
  std::size_t pool_bytes = 0;     // held unused in the pools
  std::size_t pool_peak = 0;      // the most ever held
  std::size_t pool_released = 0;  // pooled buffers given back at finalise

  // What the prefetch did. Counted here rather than in Counters because
  // only a field's storage is ever prefetched.
  std::size_t field_stages = 0;       // Role::field Views left unmanaged
  std::size_t prefetches = 0;         // ranges handed to the driver
  std::size_t prefetch_bytes = 0;     // their total size
  std::size_t prefetch_refused = 0;   // pointers the driver would not take
  std::size_t prefetch_skipped = 0;   // held back by the stride
  // Whether a pointer's storage is managed memory, asked once per pointer:
  // a run of fallen-back fields would otherwise pay a query per staging.
  std::map<const void *, bool> managed;
  // The staging at which each pointer was last prefetched, for the stride.
  std::map<const void *, std::size_t> prefetched_at;

  Counters &of(Role which) { return role[static_cast<int>(which)]; }

  // The totals are the sum of the roles rather than a second set of
  // counters: two tallies of one quantity can disagree, and one cannot.
  Counters total() const {
    Counters sum;
    for (int index = 0; index < role_count; ++index) {
      sum.staged += role[index].staged;
      sum.cached += role[index].cached;
      sum.hits += role[index].hits;
      sum.shared += role[index].shared;
      sum.copies_in += role[index].copies_in;
      sum.copies_out += role[index].copies_out;
      sum.bytes_in += role[index].bytes_in;
      sum.bytes_out += role[index].bytes_out;
      sum.allocations += role[index].allocations;
      sum.frees += role[index].frees;
      sum.reuses += role[index].reuses;
      sum.recycled += role[index].recycled;
    }
    return sum;
  }
};

inline void finish();

inline State &state() {
  // Deliberately never destroyed. The maps hold Kokkos Views, which must be
  // released before Kokkos::finalize() and not at static destruction time
  // after it; the finalize hook below empties them, and the empty object is
  // then free to outlive everything.
  static State *singleton = []() {
    State *created = new State();
    Kokkos::push_finalize_hook([]() { finish(); });
    return created;
  }();
  return *singleton;
}

#if defined(KOKKOS_ENABLE_CUDA)

// The stream the launch that follows will be issued on: ordering the copy
// before the launch on one stream is the whole of the synchronisation this
// needs. Only the Cuda execution space has a stream to name; any other
// default space takes the template below and gets the default stream.
inline cudaStream_t stream_of(const Kokkos::Cuda &space) {
  return space.cuda_stream();
}

template <typename Space>
inline cudaStream_t stream_of(const Space &) {
  return cudaStream_t(0);
}

// Is the storage behind this pointer managed memory? A host-only build has
// none, and on a device build a field whose SharedSpace claim was refused
// falls back to Fortran ALLOCATE (lfric_core's kokkos_memory.cpp), so a
// region may be handed a pointer no prefetch can take. Asked once per
// pointer and remembered. Called with the state's lock held.
inline bool managed_pointer(State &current, const void *pointer) {
  const auto found = current.managed.find(pointer);
  if (found != current.managed.end()) {
    return found->second;
  }
  cudaPointerAttributes attributes{};
  const bool answer =
      cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
      attributes.type == cudaMemoryTypeManaged;
  if (!answer) {
    // A host pointer is not an error for the next CUDA call to be blamed
    // for: some toolkits answer one, and a sticky error outlives the call.
    (void)cudaGetLastError();
  }
  current.managed.emplace(pointer, answer);
  return answer;
}

#endif  // KOKKOS_ENABLE_CUDA

// Move one field's range to the card, if the driver will take it. Every
// staging of a field either issues a prefetch, is refused or is held back
// by the stride, so those three sum to field_stages. Lock held.
inline void prefetch(State &current, const void *pointer, std::size_t bytes) {
  const std::size_t index = current.field_stages;
  current.field_stages += 1;
#if defined(KOKKOS_ENABLE_CUDA)
  if (pointer == nullptr || bytes == 0) {
    current.prefetch_refused += 1;
    return;
  }
  const std::size_t stride = prefetch_stride();
  if (stride > 0) {
    const auto seen = current.prefetched_at.find(pointer);
    if (seen != current.prefetched_at.end() && index - seen->second < stride) {
      current.prefetch_skipped += 1;
      return;
    }
  }
  int device = 0;
  if (!managed_pointer(current, pointer) ||
      cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    current.prefetch_refused += 1;
    return;
  }
  cudaMemLocation on_device{};
  on_device.type = cudaMemLocationTypeDevice;
  on_device.id = device;
  if (cudaMemPrefetchAsync(const_cast<void *>(pointer), bytes, on_device, 0,
                           stream_of(Kokkos::DefaultExecutionSpace())) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    current.prefetch_refused += 1;
    return;
  }
  current.prefetches += 1;
  current.prefetch_bytes += bytes;
  if (stride > 0) {
    current.prefetched_at[pointer] = index;
  }
#else
  // No card and no managed memory: asked for, nothing to do, and said.
  (void)pointer;
  (void)bytes;
  current.prefetch_refused += 1;
#endif
}

// The lock is the only cost a staging pays for this, and only when the knob
// is on: the caller tests prefetching() first.
inline void prefetch_field(const void *pointer, std::size_t bytes) {
  State &current = state();
  std::lock_guard<std::mutex> guard(current.lock);
  prefetch(current, pointer, bytes);
}

// The pool for one memory space. A function-local static, so a space that
// is never staged into never has one, and registered with the State on
// first use so that the finalize hook empties it while Kokkos is still up.
template <typename Space>
inline Spares &pool_of(State &current) {
  static Spares *spares = [&current]() {
    Spares *created = new Spares();
    current.pools.push_back(created);
    return created;
  }();
  return *spares;
}

// Put a block's allocation into its space's pool rather than give it back.
// Bound to the space at allocation time and reached through Block::recycle.
template <typename Space>
inline void recycle_into(State &current, Block &block) {
  Spares &spares = pool_of<Space>(current);
  spares[block.bytes].push_back(Spare{block.owner, block.data});
  current.pool_bytes += block.bytes;
  current.pool_peak = std::max(current.pool_peak, current.pool_bytes);
}

// Release one block, counting it against the role that took it. With the
// pool on, and room in it, "release" means putting the storage where the
// next argument of that size will find it; otherwise the driver gets it
// back. Every allocation this header makes leaves through here, so
// allocations plus reuses equals frees plus recycles once nothing is live.
inline void drop(State &current, Block &block, bool recycle = true) {
  if (recycle && pooling() && block.owner && block.recycle != nullptr &&
      current.pool_bytes + block.bytes <= pool_limit()) {
    block.recycle(current, block);
    current.of(block.role).recycled += 1;
  } else {
    current.of(block.role).frees += 1;
  }
  block.owner.reset();
  block.data = nullptr;
}

inline void drop_all(State &current, std::map<Key, Block> &blocks,
                     bool recycle = true) {
  for (auto &entry : blocks) {
    drop(current, entry.second, recycle);
  }
  blocks.clear();
}

// Give every pooled buffer back to the driver. Called from the finalize
// hook, before Kokkos::finalize, because a View released after it is an
// error Kokkos reports.
inline void empty_pools(State &current) {
  for (Spares *spares : current.pools) {
    for (auto &entry : *spares) {
      current.pool_released += entry.second.size();
    }
    spares->clear();
  }
  current.pool_bytes = 0;
}

inline void finish() {
  State &current = state();
  std::lock_guard<std::mutex> guard(current.lock);
  // Before the report, so that the frees it prints are all of them: what is
  // still held at finalise was allocated and is about to be released. Not
  // recycled -- there is nothing left to hand it to, and a buffer put in the
  // pool here would only have to come out again below.
  drop_all(current, current.cache, false);
  drop_all(current, current.live, false);
  empty_pools(current);
  const Counters sum = current.total();
  // Printed when staging did anything, and whenever the prefetch knob was
  // asked for, so that a run in none mode -- which stages nothing -- still
  // says what the knob resolved to: a knob that silently did nothing passes
  // every correctness gate and reads as a null lever.
  if (sum.staged + sum.cached > 0 || prefetching()) {
    std::fprintf(stderr,
                 "lfric_kokkos: staging=%s staged=%zu cached=%zu hits=%zu "
                 "shared=%zu copies_in=%zu copies_out=%zu "
                 "prefetch=%s prefetches=%zu prefetch_bytes=%zu "
                 "prefetch_refused=%zu prefetch_skipped=%zu "
                 "prefetch_stride=%zu field_stages=%zu\n",
                 mode_name(mode()), sum.staged, sum.cached, sum.hits,
                 sum.shared, sum.copies_in, sum.copies_out,
                 prefetching() ? "on" : "off", current.prefetches,
                 current.prefetch_bytes, current.prefetch_refused,
                 current.prefetch_skipped, prefetch_stride(),
                 current.field_stages);
    // One line per role, whether or not that role was met, so that a reader
    // and a parser both find the same four rows in every run.
    for (int index = 0; index < role_count; ++index) {
      const Counters &counted = current.role[index];
      std::fprintf(stderr,
                   "lfric_kokkos: staging_role=%s staged=%zu cached=%zu "
                   "hits=%zu shared=%zu copies_in=%zu copies_out=%zu "
                   "bytes_in=%zu bytes_out=%zu allocs=%zu frees=%zu "
                   "reuses=%zu recycled=%zu\n",
                   role_name(static_cast<Role>(index)), counted.staged,
                   counted.cached, counted.hits, counted.shared,
                   counted.copies_in, counted.copies_out, counted.bytes_in,
                   counted.bytes_out, counted.allocations, counted.frees,
                   counted.reuses, counted.recycled);
    }
    // What the pool did, in one line: how often storage was handed on
    // rather than allocated, and the most it held at once, which is what a
    // reader worried about device memory wants to see.
    std::fprintf(stderr,
                 "lfric_kokkos: staging_pool=%s reuses=%zu recycled=%zu "
                 "released=%zu peak_bytes=%zu\n",
                 pooling() ? "on" : "off", sum.reuses, sum.recycled,
                 current.pool_released, current.pool_peak);
  }
}

// Storage for one staged array: a spare of exactly the right size if the
// pool has one, and otherwise an allocation. The two are indistinguishable
// afterwards -- the same bytes in the same space, uninitialised either way,
// and about to be written by the copy-in that follows every call of this.
template <typename Value, typename Space>
inline Block allocate(State &current, std::size_t count, Role role) {
  Block block;
  block.bytes = count * sizeof(Value);
  block.references = 1;
  block.role = role;
  block.recycle = &recycle_into<Space>;
  if (pooling()) {
    Spares &spares = pool_of<Space>(current);
    auto found = spares.find(block.bytes);
    if (found != spares.end() && !found->second.empty()) {
      block.owner = found->second.back().owner;
      block.data = found->second.back().data;
      found->second.pop_back();
      current.pool_bytes -= block.bytes;
      current.of(role).reuses += 1;
      return block;
    }
  }
  using Owner = Kokkos::View<Value *, Kokkos::LayoutLeft, Space>;
  auto owner = std::make_shared<Owner>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lfric_kokkos_staging"),
      count);
  block.owner = owner;
  block.data = owner->data();
  current.of(role).allocations += 1;
  return block;
}

// Flat copies, because a LayoutLeft View with dynamic extents is contiguous
// and its elements are in the same order either side.
template <typename Value, typename Space>
inline void copy_in(void *device, const Value *host, std::size_t count) {
  Kokkos::View<Value *, Kokkos::LayoutLeft, Space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      target(static_cast<Value *>(device), count);
  Kokkos::View<const Value *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      source(host, count);
  Kokkos::deep_copy(target, source);
}

template <typename Value, typename Space>
inline void copy_out(Value *host, const void *device, std::size_t count) {
  Kokkos::View<const Value *, Kokkos::LayoutLeft, Space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      source(static_cast<const Value *>(device), count);
  Kokkos::View<Value *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      target(host, count);
  Kokkos::deep_copy(target, source);
}

// Obtain the View a region declares. The returned type is exactly the type
// asked for, in every mode, so the launch body never changes: what changes
// is whether the memory it covers is the caller's or this header's.
template <typename ViewType, typename... Extents>
inline ViewType stage(typename ViewType::value_type *pointer, Role role,
                      Extents... extents) {
  using Value =
      typename std::remove_const<typename ViewType::value_type>::type;
  if (mode() == Mode::none ||
      (mode() == Mode::non_field && role == Role::field)) {
    // The one path that leaves a field's storage where the caller put it,
    // and so the only one where the launch would pay a fault per page.
    // Nothing is issued for another role: a readonly or transient array in
    // none mode may be ordinary host memory, which no prefetch can take.
    if (role == Role::field && prefetching()) {
      const std::size_t size =
          (std::size_t(1) * ... * static_cast<std::size_t>(extents));
      prefetch_field(static_cast<const void *>(pointer),
                     size * sizeof(Value));
    }
    return ViewType(pointer, extents...);
  }
  using Space = typename ViewType::memory_space;
  using Pointer = typename ViewType::value_type *;
  const std::size_t count =
      (std::size_t(1) * ... * static_cast<std::size_t>(extents));
  const Key key(static_cast<const void *>(pointer), count * sizeof(Value));
  const bool cacheable =
      mode() == Mode::non_field && role == Role::readonly && caching();
  State &current = state();
  std::lock_guard<std::mutex> guard(current.lock);
  Counters &counted = current.of(role);
  if (cacheable) {
    auto found = current.cache.find(key);
    if (found == current.cache.end()) {
      Block block = allocate<Value, Space>(current, count, role);
      copy_in<Value, Space>(block.data, pointer, count);
      counted.cached += 1;
      counted.copies_in += 1;
      counted.bytes_in += block.bytes;
      found = current.cache.emplace(key, block).first;
    } else {
      counted.hits += 1;
    }
    return ViewType(static_cast<Pointer>(found->second.data), extents...);
  }
  // Not cached: one allocation per (pointer, bytes) per region call, shared
  // by every argument naming the same storage, so that two arguments
  // aliasing one array see each other's writes exactly as they do in none
  // mode.
  auto found = current.live.find(key);
  if (found != current.live.end()) {
    found->second.references += 1;
    counted.shared += 1;
    return ViewType(static_cast<Pointer>(found->second.data), extents...);
  }
  Block block = allocate<Value, Space>(current, count, role);
  copy_in<Value, Space>(block.data, pointer, count);
  counted.staged += 1;
  counted.copies_in += 1;
  counted.bytes_in += block.bytes;
  current.live.emplace(key, block);
  return ViewType(static_cast<Pointer>(block.data), extents...);
}

// Copy a written View back to the caller's storage. A no-op in none mode,
// and for a View this call did not stage -- a field in non-field mode, or
// an array whose last reference is still held by another argument.
template <typename ViewType>
inline void unstage(const ViewType &view,
                    typename ViewType::value_type *pointer) {
  if (mode() == Mode::none) {
    return;
  }
  using Value =
      typename std::remove_const<typename ViewType::value_type>::type;
  using Space = typename ViewType::memory_space;
  const std::size_t count = view.size();
  const Key key(static_cast<const void *>(pointer), count * sizeof(Value));
  State &current = state();
  std::lock_guard<std::mutex> guard(current.lock);
  auto found = current.live.find(key);
  if (found == current.live.end()) {
    return;
  }
  found->second.references -= 1;
  if (found->second.references > 0) {
    return;
  }
  copy_out<Value, Space>(const_cast<Value *>(pointer), found->second.data,
                         count);
  Counters &counted = current.of(found->second.role);
  counted.copies_out += 1;
  counted.bytes_out += found->second.bytes;
  drop(current, found->second);
  current.live.erase(found);
}

// End of a region call: drop what is still staged. Read-only arrays staged
// in all mode have no unstage() of their own -- there is nothing to copy
// back -- and without this they would accumulate one allocation per call.
// The cache is untouched.
inline void release() {
  if (mode() == Mode::none) {
    return;
  }
  State &current = state();
  std::lock_guard<std::mutex> guard(current.lock);
  drop_all(current, current.live);
}

}  // namespace lfric_kokkos

#endif  // LFRIC_KOKKOS_STAGING_HPP
'''


def header_text():
    '''The text of ``lfric_kokkos_staging.hpp``.

    The single source of it: the transformation that captures a model's
    regions writes this beside them, and the gates that compile it load this
    function rather than carrying a copy.

    :returns: the header, ending in a newline.
    :rtype: str

    '''
    return _HEADER_TEXT.lstrip("\n")


def include_line():
    '''The ``#include`` a generated region carries for the header.

    :returns: the line, without its newline.
    :rtype: str

    '''
    return f'#include "{HEADER_NAME}"'


def role_of(view):
    '''The staging role of a View, falling back to what its constness says.

    A description built by the LFRic transformations names the role, because
    that is where an argument's meaning is known. A description built by hand
    -- a test, or a writer used directly -- may not, and then the role is
    read from ``read_only``: a read-only array is ``readonly`` and a written
    one is ``readwrite``. Neither of those claims the argument is in a space
    the device shares, which is the claim ``field`` makes and the only one
    that would be unsafe to assume.

    :param view: the View to place.
    :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

    :returns: one of :py:data:`ROLES`.
    :rtype: str

    :raises ValueError: if the View names a role that is not one of
        :py:data:`ROLES`.

    '''
    role = getattr(view, "role", None)
    if role is None:
        return "readonly" if view.read_only else "readwrite"
    if role not in ROLES:
        raise ValueError(
            f"View '{view.name}' carries staging role '{role}', which is not "
            f"one of {list(ROLES)}.")
    return role


# Six arguments, and each is one field of the declaration being written: the
# View's type, the name the body subscripts, the pointer the storage
# arrives in, the role, the extents and the indentation. Gathering them
# into an object would be a second description of a View beside
# KokkosView, which is the thing the caller already holds.
def stage_declaration(view_type, name, data_name, role, extents, indent="  "):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    '''The declaration of one staged View.

    :param str view_type: the C++ View type, which is what ``stage`` returns
        in every mode and therefore what the launch body is compiled against.
    :param str name: the name the region's body subscripts.
    :param str data_name: the pointer argument the storage arrives in.
    :param str role: one of :py:data:`ROLES`.
    :param extents: one C++ integer expression per dimension.
    :type extents: Iterable[str]
    :param str indent: the indentation of the first line.

    :returns: the declaration, indented, without a trailing newline.
    :rtype: str

    '''
    arguments = ", ".join(
        (data_name, f"lfric_kokkos::Role::{role}", *extents))
    return (f"{indent}auto {name} = lfric_kokkos::stage<\n"
            f"{indent}    {view_type}>(\n"
            f"{indent}    {arguments});")


def unstage_statement(name, data_name, indent="  "):
    '''The write-back of one staged View, for after the region's fence.

    :param str name: the View.
    :param str data_name: the pointer its contents belong to.
    :param str indent: the indentation.

    :returns: the statement, ending in a newline.
    :rtype: str

    '''
    return f"{indent}lfric_kokkos::unstage({name}, {data_name});\n"


def release_statement(indent="  "):
    '''The end-of-region release of whatever is still staged.

    :param str indent: the indentation.

    :returns: the statement, ending in a newline.
    :rtype: str

    '''
    return f"{indent}lfric_kokkos::release();\n"


def view_declaration(view):
    '''The declaration of one of a region's array arguments.

    The C++ type is the type it was before staging existed -- element type,
    layout, rank and memory traits -- because that is what the launch body
    is compiled against and none of it may move. What staging decides is the
    memory the View covers, and that is decided at run time inside
    :py:func:`stage`.

    :param view: the View to declare over storage the caller owns.
    :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

    :returns: the declaration, indented for the region body, without a
        trailing newline.
    :rtype: str

    '''
    const = "const " if view.read_only else ""
    rank = "*" * len(view.extents)
    traits = "ReadOnly" if view.random_access else "Unmanaged"
    return stage_declaration(
        f"Kokkos::View<{const}{view.c_type}{rank}, "
        f"Kokkos::LayoutLeft, MemorySpace, {traits}>",
        view.name, view.data_name, role_of(view), view.extents)


def staging_epilogue(region) -> str:
    """Return the statements a region runs after its fence.

    After the fence, so that every write the launch made has landed: a
    written View is copied back to the caller's storage, and then whatever
    staging still holds for this call is dropped. Both are no-ops in the
    default mode, and a region with no Views at all emits neither rather
    than a call about nothing.

    :param region: the captured region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos_region.KokkosRegion`

    :returns: the epilogue statements, each already a full line, or the
        empty string for a region that describes no View.
    """
    # Imported here rather than at the top of the module: the shell gates
    # that compile the header load this file by path, with no PSyclone
    # package around it, and only the spelling helpers are wanted there.
    # pylint: disable=import-outside-toplevel
    from psyclone.psyir.backend.kokkos_region import KokkosView
    view_arguments = [argument for argument in region.arguments
                      if isinstance(argument, KokkosView)]
    if not view_arguments:
        return ""
    return "".join(
        unstage_statement(argument.name, argument.data_name)
        for argument in view_arguments
        if not argument.read_only) + release_statement()


__all__ = ["HEADER_NAME", "ROLES", "header_text", "include_line",
           "release_statement", "role_of", "stage_declaration",
           "staging_epilogue", "unstage_statement", "view_declaration"]
