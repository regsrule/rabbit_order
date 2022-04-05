//
// Optimized edge list reader
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

#include "edge_list.h"

namespace edge_list {
namespace aux {

off_t file_size(const int fd) {
  struct stat st;
  if (fstat(fd, &st) != 0)
    RO_DIE << "stat(2): " << strerror(errno);
  return st.st_size;   
}

std::vector<edge> read(const std::string& path) {
  const file_desc    fd(path);
  const mmapped_file mm(fd);
  const int          nthread = omp_get_max_threads();
  const size_t       zchunk  = 1024 * 1024 * 64;  // 64MiB
  const size_t       nchunk  = mm.size / zchunk + (mm.size % zchunk > 0);

  //
  // For load balancing, partition the file into small chunks (whose size is
  // defined as `zchunk`) and dynamically assign the chunks into threads
  //

  std::vector<std::deque<edge> > eparts(nthread);
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < nchunk; ++i) {
    const char* p = mm.data + zchunk * i;
    const char* q = mm.data + std::min(zchunk * (i + 1), mm.size);

    // Advance pointer `p` to the end of a line because it is possibly at the
    // middle of the line
    if (i > 0) p = std::find(p, q, '\n');

    if (p < q) {  // If `p == q`, do nothing
      q = std::find(q, mm.data + mm.size, '\n');  // Advance `q` likewise
      edge_parser(p, q)(std::back_inserter(eparts[omp_get_thread_num()]));
    }
  }

  // Compute indices to copy each element of `eparts` to
  std::vector<size_t> eheads(nthread + 1);
  for (int t = 0; t < nthread; ++t)
    eheads[t + 1] = eheads[t] + eparts[t].size();

  // Gather the edges read by each thread to a single array
  std::vector<edge> edges(eheads.back());
  #pragma omp parallel for schedule(guided, 1)
  for (int t = 0; t < nthread; ++t)
    boost::copy(eparts[t], edges.begin() + eheads[t]);

  return edges;
}

}  // namespace edge_list::aux

/*using aux::edge;
using aux::read;*/

}  // namespace edge_list
