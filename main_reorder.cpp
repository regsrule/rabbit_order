//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

#include "reorder.h"

int main(int argc, char* argv[]) {
  using boost::adaptors::transformed;

  // Parse command-line arguments
  if (argc != 2 && (argc != 3 || std::string("-c") != argv[1])) {
    std::cerr << "Usage: reorder [-c] GRAPH_FILE\n"
              << "  -c    Print community IDs instead of a new ordering\n";
    exit(EXIT_FAILURE);
  }
  const std::string graphpath = argc == 3 ? argv[2] : argv[1];
  const bool        commode   = argc == 3;

  std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;

  std::cerr << "Reading an edge-list file: " << graphpath << std::endl;
  auto       adj = read_graph(graphpath);
  const auto m   =
      boost::accumulate(adj | transformed([](auto& es) {return es.size();}),
                        static_cast<size_t>(0));
  std::cerr << "Number of vertices: " << adj.size() << std::endl;
  std::cerr << "Number of edges: "    << m          << std::endl;

/*  if (commode)
    detect_community(std::move(adj));
  else
    reorder(std::move(adj));*/


  return EXIT_SUCCESS;
}

