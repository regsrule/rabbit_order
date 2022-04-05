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
#include "rabbit_order.h"

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


//
// Imitation of `std::make_unique<T[]>(size_t size)`
//
template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
unique_c_ptr<T> make_aligned_unique(const size_t n, const size_t align) {
  typedef typename std::remove_extent<T>::type elem_t;
  const size_t z = sizeof(elem_t) * n;
  elem_t*      p;
  if (posix_memalign(reinterpret_cast<void**>(&p), align, z) != 0 ||
      p == nullptr) {
    RO_DIE << "posix_memalign(3) failed";
  }
  return std::unique_ptr<T, free_functor>(p, free_functor());
}

//------------------------------------------------------------------------------
//
// NUMA-related definitions
//
//------------------------------------------------------------------------------

//
// Allocated regions must be released by `free_body`
//
template<typename T>
T* alloc_interleaved(const size_t nelem) {
  const size_t    z = sizeof(block<T>) + nelem * sizeof(T);
  block<T>* const b = reinterpret_cast<block<T>*>(numa_alloc_interleaved(z));
  if (b == NULL)
    RO_DIE << "numa_alloc_interleaved(3) failed";
  b->size = z;
  return b->body;  // Returns a pointer to the body
}

//
// Releases a region allocated by `alloc_*`
//
template<typename T>
void free_body(T* const body) {
  if (body != nullptr) {
    // Compute an address that satisfies `b->body == body`
    block<T>* const b = reinterpret_cast<block<T>*>(
        reinterpret_cast<uint8_t*>(body) - offsetof(block<T>, body));
    assert(b->size > offsetof(block<T>, body));
    numa_free(b, b->size);
  }
}

//
// Imitation of `std::make_unique<T[]>(size_t size)`
//
template<typename T, typename = std::enable_if_t<std::is_array<T>::value> >
numa_unique_ptr<T> make_unique_interleaved(const size_t n) {
  return numa_unique_ptr<T>(
      alloc_interleaved<typename std::remove_extent<T>::type>(n));
}

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
                    Accum         accum) {
  if (first != last) {
    auto x = *first;
    while (++first != last) {
      if (equal(x, *first)) {
        x = accum(x, *first);
      } else {
        *dest++ = x;
        x       = *first;
      }
    }
    *dest++ = x;
  }
  return dest;
}

double now_sec() {
  return static_cast<std::chrono::duration<double> >(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

template<typename T>
std::vector<T> join(const std::vector<std::deque<T> >& xss) {
  size_t len = 0;
  for (auto& xs : xss) len += xs.size();

  std::vector<T> ys;
  ys.reserve(len);
  for (auto& xs : xss) boost::copy(xs, std::back_inserter(ys));
  return ys;
}

//============================================================================
//
// CORE FUNCTIONS
//
//============================================================================

//
// Returns ID of a vertex representing the community that `v` belongs to.
//
vint trace_com(const vint v, graph* const g) {
  vint com = v;
  for (;;) {
    const vint c = g->coms[com];
    if (c == com) break;
    com = c;
  }

  // Update community information of `v`
  //
  // [Why is `v != com` checked?]
  // If `v == com` (i.e., `coms[v] == v`), there are two possibilities:
  // (1) `v` has not yet been merged, or (2) `v` is top-level.
  // In case (1), we must not overwrite `coms[v]` because another thread may
  // concurrently merge `v` into its neighbor and store a value such that
  // `coms[v] != v`.
  //
  // [Why is `coms[v] != com` checked?]
  // It may cause unnecessary cache-block invalidation.
  //
  // [Possibility to store old values]
  // Because vertex `com` may be concurrently merged to another vertex,
  // `coms[com] == com` may be broken at `coms[v] = com`.
  // However, even if the old value is stored in `coms[v]`, the next call of
  // `trace_com` updates `coms[v]` into the latest value.
  //
  if (v != com && g->coms[v] != com)
    g->coms[v] = com;

  return com;
}

//
// Aggregates the duplicate edges in `[it, last)`.
// This function is sequential because it is called from parallel regions.
//
template <typename InputIt, typename OutputIt>
OutputIt compact(InputIt it, const InputIt last, OutputIt result) {
  if (it == last)
    return result;

  std::sort(it, last, [](auto& e0, auto& e1) {return e0.first < e1.first;});
  return uniq_accum(it, last, result,
      [](auto x, auto y) {return x.first == y.first;},
      [](auto x, auto y) {return edge {x.first, x.second + y.second};});
}

//
// Aggregates the edges of `v` and the vertices merged to `v`, and writes them
// to `nbrs`.
//
void unite(const vint v, std::vector<edge>* const nbrs, graph* const g) {
  ptrdiff_t icmb = 0;

  nbrs->clear();

  const auto push_edges = [v, nbrs, g, &icmb](const vint u) {
    const size_t     cap  = nbrs->capacity();
    constexpr size_t npre = 8;  // TODO: tuning parameter
    auto&            es   = g->es[u];

    for (size_t i = 0; i < es.size() && i < npre; ++i)
      __builtin_prefetch(&g->coms[es[i].first], 0, 3);
    for (size_t i = 0; i < es.size(); ++i) {
      if (i + npre < es.size())
        __builtin_prefetch(&g->coms[es[i + npre].first], 0, 3);
      const vint c = trace_com(es[i].first, g);
      if (c != v)  // Remove a self-loop edge
        nbrs->push_back({c, es[i].second});
    }

#ifdef DEBUG
    if (nbrs->size() > cap)
      std::cerr << "WARNING: edge accumulation buffer is reallocated\n";
#else
    static_cast<void>(cap);
#endif

    // combine edges before uncombined edges overflows a L2 cache
    // TODO: tuning
    if (nbrs->size() - icmb >= 2048) {
      const auto it = nbrs->begin() + icmb;
      icmb = compact(it, nbrs->end(), it) - nbrs->begin();
      nbrs->resize(icmb);
    }
  };

  push_edges(v);

  // `child` may be modified if another thread merges a vertex into `v`, but
  // this function is not responsible for prohibiting modification of `child`.
  while (g->vs[v].united_child != g->vs[v].a->child) {
    // The vertices in the list connected by `sibling` are already merged, and
    // hence they are never be modified by the other threads.
    const vint c = g->vs[v].a->child;
    vint       w;
    for (w = c; w != vmax && w != g->vs[v].united_child; w = g->vs[w].sibling)
      push_edges(w);

    // `c` and the descendants of `c` have been merged into `v`
    g->vs[v].united_child = c;
  }

  g->tot_nbrs.fetch_add(nbrs->size());

  g->es[v].clear();
  compact(nbrs->begin(), nbrs->end(), std::back_inserter(g->es[v]));
}

//
// Returns a vertex that yields the best modularity improvement when it is
// merged with `v`.
//
vint find_best(const graph& g, const vint v, const double vstr) {
  double dmax = 0.0;
  vint   best = v;
  for (const edge e : g.es[v]) {
    const double d =
        static_cast<double>(e.second) - vstr * g.vs[e.first].a->str / g.tot_wgt;
    if (dmax < d) {
      dmax = d;
      best = e.first;
    }
  }
  return best;
}

//
// Merges `v` into one of the neighbors of `v`
//
// [Return value]
// `v`    : `v` is not merged since it has no neighbor that improves modularity
// `vmax` : `v` is not merged due to contention with another threads
// others : `v` is merged into a vertex represented by the return value
//
vint merge(const vint v, std::vector<edge>* const nbrs, graph* const g) {
  // Aggregate edges of the members of community `v`
  // Aggregating before locking `g[v]` shortens the locking time
  unite(v, nbrs, g);

  // `.str < 0.0` means that modification of `g[v]` is prohibited (locked)
  const float vstr = g->vs[v].a->str.exchange(-1);

  // If `.child` was modified between the previous call of `unite()` and the
  // lock, aggregate edges again
  if (g->vs[v].a->child != g->vs[v].united_child) {
    unite(v, nbrs, g);
    g->n_reunite.fetch_add(1);
  }

  const vint u = find_best(*g, v, vstr);
  if (u == v) {
    // Rollback the strength if there is no neighbor that improves modularity
    g->vs[v].a->str = vstr;
  } else {
    // Rollback if `u` has a negative strength (i.e., `u` is locked)
    atom ua = g->vs[u].a;  // atomic load
    if (ua.str < 0.0) {
      g->vs[v].a->str = vstr;
      g->n_fail_lock.fetch_add(1);
      return vmax;
    }

    // `.sibling` can be accessed immediately by `unite()` after letting
    // `g->vs[u].a->child = v`, and so set `.sibling` properly in advance
    g->vs[v].sibling = ua.child;

    // Abort and rollback if CAS failed due to modification of `u`
    const atom _ua(ua.str + vstr, v);
    if (!g->vs[u].a.compare_exchange_weak(ua, _ua)) {
      g->vs[v].sibling = vmax;
      g->vs[v].a->str  = vstr;
      g->n_fail_cas.fetch_add(1);
      return vmax;
    }

    // Update the community of `v`
    g->coms[v] = u;
  }

  assert(u != v || is_toplevel(*g, v));
  assert(u == v || is_merged(*g, v));

  return u;
}

//
// Sorts the vertices into ascending order of (unweighted) degree
//
std::unique_ptr<std::pair<vint, vint>[]> merge_order(const graph& g) {
  // Co-locating vertex ID and its degree shows better locality
  auto ord = std::make_unique<std::pair<vint, vint>[]>(g.n());
  #pragma omp parallel for
  for (vint v = 0; v < g.n(); ++v)
    ord[v] = {v, static_cast<vint>(g.es[v].size())};

  __gnu_parallel::sort(&ord[0], &ord[g.n()], 
                       [](auto x, auto y) {return x.second < y.second;});
  return ord;
}

//
// Write `v` and the lineal descendants of `v` on the dendrogram to `it`
// (siblings of the children are not included)
//
template<typename OutputIt, typename G>
void descendants(const G& g, vint v, OutputIt it) {
  *it++ = v;
  while ((v = g.vs[v].a->child) != vmax)
    *it++ = v;
}

graph aggregate(std::vector<std::vector<edge> > adj) {
  graph      g(std::move(adj));
  const auto ord   = merge_order(g);
  const int  np    = omp_get_max_threads();
  size_t     npend = 0;
  double     tmax  = 0.0, ttotal = 0.0;
  std::vector<std::deque<vint> > topss(np);

  #pragma omp parallel reduction(+: npend) reduction(max: tmax) reduction(+: ttotal)
  {
    const double     tstart = now_sec();
    const int        tid    = omp_get_thread_num();
    std::deque<vint> tops, pends;

    std::vector<edge> nbrs;
    nbrs.reserve(g.n() * 2);  // heuristic value   TODO: tuning

    #pragma omp for schedule(static, 1)
    for (vint i = 0; i < g.n(); ++i) {
      pends.erase(boost::remove_if(pends, [&g, &tops, &nbrs](auto w) {
        const vint u = merge(w, &nbrs, &g);
        if (u == w) tops.push_back(w);
        return u != vmax;  // remove if the merge successed
      }), pends.end());

      const vint v = ord[i].first;
      const vint u = merge(v, &nbrs, &g);
      if      (u == v)    tops.push_back(v);
      else if (u == vmax) pends.push_back(v);
    }

    ttotal = now_sec() - tstart;
    tmax   = ttotal;

    // Merge the vertices in the pending state 
    #pragma omp barrier
    #pragma omp critical
    {
      npend = pends.size();
      for (const vint v : pends) {
        const vint u = merge(v, &nbrs, &g);
        if (u == v) tops.push_back(v);
        assert(u != vmax);  // The merge never fails
      }
      topss[tid] = std::move(tops);
    }
  }

  g.tops = join(topss);

  // `tops` does not have duplicated elements
  assert(([&g]() {
    auto tops = *g.tops;
    return g.tops->size() == boost::size(boost::unique(boost::sort(tops)));
  })());

  //std::cerr << "CPU time utilization rate: " <<  ttotal / (tmax * np)
  //          << "\nvertices left to be pended: " << npend
  //          << "\n`unite()` calls after lock: " << g.n_reunite
  //          << "\nmerge failures by negative-strength: " << g.n_fail_lock
  //          << "\nmerge failures by compare-and-swap: " << g.n_fail_cas
  //          << "\ntot_nbrs = " << g.tot_nbrs << std::endl;
  static_cast<void>(npend);  // suppress Wunused-but-set-variable
  static_cast<void>(tmax);

  assert(check_result(&g));
  return g;
}

std::unique_ptr<vint[]> compute_perm(const graph& g) {
  auto              perm = std::make_unique<vint[]>(g.n());
  auto              coms = std::make_unique<vint[]>(g.n());
  const vint        ncom = static_cast<vint>(g.tops->size());
  std::vector<vint> offsets(ncom + 1);

  const int  np    = omp_get_max_threads();
  const vint ntask = std::min<vint>(ncom, 128 * np);
  #pragma omp parallel
  {
    std::deque<vint> stack;

    #pragma omp for schedule(dynamic, 1)
    for (vint i = 0; i < ntask; ++i) {
      for (vint comid = i; comid < ncom; comid += ntask) {
        vint newid = 0;

        descendants(g, (*g.tops)[comid], std::back_inserter(stack));

        while (!stack.empty()) {
          const vint v = stack.back();
          stack.pop_back();

          coms[v] = comid;
          perm[v] = newid++;

          if (g.vs[v].sibling != vmax)
            descendants(g, g.vs[v].sibling, std::back_inserter(stack));
        }

        offsets[comid + 1] = newid;
      }
    }
  }

  boost::partial_sum(offsets, offsets.begin());
  assert(offsets.back() == g.n());

  #pragma omp parallel for schedule(static)
  for (vint v = 0; v < g.n(); ++v)
    perm[v] += offsets[coms[v]];

  // `perm` must contain `[0, g.n())`
  assert(([&g, &perm]() {
    std::vector<vint> sorted(&perm[0], &perm[g.n()]);
    return boost::equal(boost::sort(sorted),
                        boost::irange(static_cast<vint>(0), g.n()));
  })());

  return perm;
}

//============================================================================
//
// DEBUGGING & TESTING UTILITIES
//
//============================================================================

//
// True if attributes of `v` is consistent as a top-level vertex
//
bool is_toplevel(const graph& g, const vint v) {
  return g.vs[v].a->str >= 0.0 &&    // not locked
         g.vs[v].sibling == vmax &&  // no siblings
         g.coms[v] == v;
}

//
// True if attributes of `v` is consistent as a vertex that has been merged to
// another vertex
//
bool is_merged(const graph& g, const vint v) {
  return g.vs[v].a->str < 0.0 &&   // permanently locked
         g.coms[v] != v;
};

//
// Checks result of incremental aggregation using `assert`
//
bool check_result(graph* const pg) {
  static_cast<void>(pg);

#ifndef NDEBUG
  auto&      g     = *pg;
  const auto vall  = boost::irange(static_cast<vint>(0), g.n());
  const auto istop = [&g](const vint v) {return is_toplevel(g, v);};

  // For all vertex `v`, `trace_com(v)` is in `g.tops`
  {
    std::unordered_set<vint> topids;
    for (vint v = 0; v < g.n(); ++v) topids.insert(trace_com(v, &g));
    std::vector<vint> got(topids.begin(), topids.end());
    auto              ans = *g.tops;
    assert(boost::equal(boost::sort(ans), boost::sort(got)));
  }

  // `g.tops` includes only top-level vertices
  assert(boost::algorithm::all_of(*g.tops, istop));
  // The number of the top-level vertices is equal to the size of `g.tops`,
  // i.e., `g.tops` includes all the top-level vertices
  assert(boost::count_if(vall, istop) == static_cast<intmax_t>(g.tops->size()));

  // All the remaining communities are top-level
  assert(boost::algorithm::all_of(vall, [&g](auto v) {
    const vint c = trace_com(v, &g);
    return is_toplevel(g, c);
  }));

  // Every vertex `v` is consistent as a top-level vertex or a merged vertex
  assert(boost::algorithm::all_of(vall, [&g](auto v) {
    return is_toplevel(g, v) || is_merged(g, v);
  }));
#endif

  return true;
}

}  // namespace rabbit_order::aux

/*using aux::vint;
using aux::now_sec;
using aux::edge;
using aux::trace_com;
using aux::aggregate;
using aux::compute_perm;*/

}  // namespace rabbit_order
