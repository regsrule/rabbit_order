//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//
#ifndef REORDER
#define REORDER
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>
#include "rabbit_order.h"
#include "edge_list.h"

using rabbit_order::vint;
typedef std::vector<std::vector<std::pair<vint, float> > > adjacency_list;

vint count_unused_id(const vint n, const std::vector<edge_list::edge>& edges); 

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange& es); 

adjacency_list read_graph(const std::string& graphpath);

template<typename InputIt>
typename std::iterator_traits<InputIt>::difference_type
count_uniq(const InputIt f, const InputIt l);

double compute_modularity(const adjacency_list& adj, const vint* const coms); 

void detect_community(adjacency_list adj);

void reorder(adjacency_list adj);

std::vector<std::vector<int>> reorder_set_graph(adjacency_list adj);

std::vector<std::vector<std::vector<int>>> store_partition_rabbit(adjacency_list adj, bool store_affinity);
#endif

