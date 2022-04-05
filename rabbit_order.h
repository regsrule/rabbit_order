//
// ==================================================================
// ==                                                              ==
// ==      PLEASE READ 'license.txt' BEFORE USING THIS SOURCE      ==
// ==                                                              ==
// ==================================================================
//
// Copyright (C) 2015 Nippon Telegraph and Telephone Corporation
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//
#ifndef RABBIT_ORDER 
#define RABBIT_ORDER
#pragma once

#include <numa.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <parallel/algorithm>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/atomic.hpp>
#include <boost/optional/optional.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/count_if.hpp>
#include <boost/range/algorithm/remove_if.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/range/algorithm/unique.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>

//
// Output stream for fatal errors.
// Prints messages and exits the program with EXIT_FAILURE.
// "RO" stands for "Rabbit Order"
//
// Usage:
//     RO_DIE << "error message";  // same as std::cerr
//
#define RO_DIE rabbit_order::aux::die_t(__FILE__, __LINE__, __func__)

namespace rabbit_order {
namespace aux {

//==============================================================================
//
// UTILITIES
//
//==============================================================================

//------------------------------------------------------------------------------
//
// Helpers of `RO_DIE`
//
//------------------------------------------------------------------------------

struct die_t {
  die_t(const char* file, uint64_t line, const char* func) {
    std::cerr << file << ':' << line << '(' << func << ") [FATAL ERROR] ";
  }
  ~die_t() {
    std::cerr << std::endl;
    exit(EXIT_FAILURE);
  }
};

template<typename T>
die_t operator<<(die_t d, const T& x) {
  std::cerr << x;
  return d;
}

//------------------------------------------------------------------------------
//
// unique_c_ptr:
// std::unique_ptr that releases memory by `std::free()` (C-language style)
//
//------------------------------------------------------------------------------

struct free_functor {void operator()(void* p) const {std::free(p);}};
template<typename T> using unique_c_ptr = std::unique_ptr<T, free_functor>;

//
// Imitation of `std::make_unique<T[]>(size_t size)`
//
template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
unique_c_ptr<T> make_aligned_unique(const size_t n, const size_t align); 

//------------------------------------------------------------------------------
//
// NUMA-related definitions
//
//------------------------------------------------------------------------------

//
// Keeps the size of each allocated region because `numa_free` requires it
//
template<typename T>
struct block {
  size_t size;
  T      body[0];
};

//
// Allocated regions must be released by `free_body`
//
template<typename T>
T* alloc_interleaved(const size_t nelem); 

//
// Releases a region allocated by `alloc_*`
//
template<typename T>
void free_body(T* const body);

//
// unique_ptr for memory regions allocated by `alloc_*`
//
template<typename T>
struct free_body_functor {void operator()(T* p) const {free_body(p);}};
template<typename T>
using numa_unique_ptr = std::unique_ptr<
    T, free_body_functor<typename std::remove_extent<T>::type> >;

//
// Imitation of `std::make_unique<T[]>(size_t size)`
//
template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
numa_unique_ptr<T> make_unique_interleaved(const size_t n); 

//------------------------------------------------------------------------------
// Customized boost::atomic
// - Uses `accuire`, `release`, and `acq_rel` as a default memory_order
// - Has a default constructor and a copy constructor
// - Allows access to the internal value by reference
//------------------------------------------------------------------------------
template<typename T>
union atomix {
  // `std::atomic<T>` requires (and checks) that `T` is TriviallyCopyable, and
  // hence we resort to use `boost::atomic<T>` instead.
  boost::atomic<T> a;
  T                raw;

  atomix()                   : a(T())               {}
  atomix(const atomix<T>& x) : a(static_cast<T>(x)) {}
  atomix(T x)                : a(x)                 {}

  operator T() const {return a.load(boost::memory_order_acquire);}
  T fetch_add(T x) {return a.fetch_add(x, boost::memory_order_acq_rel);}
  T exchange(T x) {return a.exchange(x, boost::memory_order_acq_rel);}

  bool compare_exchange_weak(T& exp, T x)
      {return a.compare_exchange_weak(exp, x, boost::memory_order_acq_rel);}
  atomix& operator=(const atomix<T>& x)
      {a.store(x, boost::memory_order_release); return *this;}
  atomix& operator=(T x)
      {a.store(x, boost::memory_order_release); return *this;}

  // Direct access to the value in the boost::atomic
  const T* operator->() const {
    assert(sizeof(*this) == sizeof(T) && a.is_lock_free());

    // Accessing `a` via `raw` breaks the strict aliasing rule, but has better
    // portability and efficiency
    return &raw;
  }

  T* operator->() {
    return const_cast<T*>(static_cast<const atomix<T>*>(this)->operator->());
  }
};

//------------------------------------------------------------------------------
//
// Miscellaneous definitions
//
//------------------------------------------------------------------------------

//
// {a0, a1, a2, b0, c0, c1, ...}
// ==> {accum(accum(a0, a1), a2), b0, accum(c0, c1), ...}
//     where equal(a0, a1), equal(a0, a2), !equal(a0, b0), !equal(b0, c0),
//           and equal(c0, c1)
//
template<typename InputIt, typename OutputIt, typename Equal, typename Accum>
OutputIt uniq_accum(InputIt       first,
                    const InputIt last,
                    OutputIt      dest,
                    Equal         equal,
                    Accum         accum); 

double now_sec(); 

template<typename T>
std::vector<T> join(const std::vector<std::deque<T> >& xss);

//==============================================================================
//
// CONSTANTS, TYPES, AND FUNCTIONS TO HANDLE THEM
//
//==============================================================================

typedef uint32_t vint;  // Integer for vertex IDs

constexpr vint vmax = std::numeric_limits<vint>::max();  // Used as invalid ID

//
// (Vertex ID of the target) * (edge weight)
//
typedef std::pair<vint, float> edge;

//
// Part of the vertex attributes that is to be atomically modified
//
struct atom {
  atomix<float> str;    // Total weighted degree of the community members
  atomix<vint>  child;  // Last vertex that is merged to this vertex

  atom()                : str(0.0), child(vmax) {}
  atom(float s)         : str(s),   child(vmax) {}
  atom(float s, vint c) : str(s),   child(c)    {}
} __attribute__((aligned(8)));

//
// Vertex attributes
//
struct vertex {
  atomix<atom> a;
  atomix<vint> sibling;
  vint         united_child;

  vertex(float str) : a(atom(str)), sibling(vmax), united_child(vmax) {}
};

//
// Graph structure and a record of incremental aggregation steps
//
struct graph {
  numa_unique_ptr<atomix<vint>[]>     coms;     // Vertex ID -> community ID
  unique_c_ptr<vertex[]>              vs;       // Vertex ID -> attributes
  std::vector<std::vector<edge> >     es;       // Vertex ID -> edges
  double                              tot_wgt;  // Total weight of all edges
  boost::optional<std::vector<vint> > tops;     // Top-level vertices (result)

  // Counters for performance analysis
  atomix<size_t> n_reunite;    // # of calls of `unite()` after the lock
  atomix<size_t> n_fail_lock;  // # of merge rollbacks due to lock
  atomix<size_t> n_fail_cas;   // # of merge rollbacks due to CAS
  atomix<size_t> tot_nbrs;

  graph()                   = default;
  graph(graph&& x)          = default;
  graph& operator=(graph&&) = default;

  graph(std::vector<std::vector<edge> > _es)
      : coms(), vs(), es(std::move(_es)), tot_wgt(), tops(), n_reunite(),
        n_fail_lock(), n_fail_cas(), tot_nbrs() {
    const vint nvtx = static_cast<vint>(es.size());
    vs   = make_aligned_unique<vertex[]>(nvtx, sizeof(vertex));
    coms = make_unique_interleaved<atomix<vint>[]>(nvtx);

    double w = 0.0;
    #pragma omp parallel for reduction(+:w)
    for (vint v = 0; v < nvtx; ++v) {
      float s = 0.0f;
      for (auto& e : es[v]) s += e.second;
      ::new(&vs[v]) vertex(s);
      w += s;
      coms[v] = v;
    }
    tot_wgt = w;
  }

  vint n() const {return static_cast<vint>(es.size());}
};

//============================================================================
//
// CORE FUNCTIONS
//
//============================================================================

//
// Returns ID of a vertex representing the community that `v` belongs to.
//
vint trace_com(const vint v, graph* const g); 

//
// Aggregates the duplicate edges in `[it, last)`.
// This function is sequential because it is called from parallel regions.
//
template <typename InputIt, typename OutputIt>
OutputIt compact(InputIt it, const InputIt last, OutputIt result); 

//
// Aggregates the edges of `v` and the vertices merged to `v`, and writes them
// to `nbrs`.
//
void unite(const vint v, std::vector<edge>* const nbrs, graph* const g); 

//
// Returns a vertex that yields the best modularity improvement when it is
// merged with `v`.
//
vint find_best(const graph& g, const vint v, const double vstr);

//
// Merges `v` into one of the neighbors of `v`
//
// [Return value]
// `v`    : `v` is not merged since it has no neighbor that improves modularity
// `vmax` : `v` is not merged due to contention with another threads
// others : `v` is merged into a vertex represented by the return value
//
vint merge(const vint v, std::vector<edge>* const nbrs, graph* const g); 

//
// Sorts the vertices into ascending order of (unweighted) degree
//
std::unique_ptr<std::pair<vint, vint>[]> merge_order(const graph& g);

//
// Write `v` and the lineal descendants of `v` on the dendrogram to `it`
// (siblings of the children are not included)
//
template<typename OutputIt, typename G>
void descendants(const G& g, vint v, OutputIt it);

graph aggregate(std::vector<std::vector<edge> > adj);

std::unique_ptr<vint[]> compute_perm(const graph& g); 

//============================================================================
//
// DEBUGGING & TESTING UTILITIES
//
//============================================================================

//
// True if attributes of `v` is consistent as a top-level vertex
//
bool is_toplevel(const graph& g, const vint v); 

//
// True if attributes of `v` is consistent as a vertex that has been merged to
// another vertex
//
bool is_merged(const graph& g, const vint v);

//
// Checks result of incremental aggregation using `assert`
//
bool check_result(graph* const pg); 

}  // namespace rabbit_order::aux

using aux::vint;
using aux::now_sec;
using aux::edge;
using aux::trace_com;
using aux::aggregate;
using aux::compute_perm;

}  // namespace rabbit_order

#endif
