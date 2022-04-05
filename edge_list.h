//
// Optimized edge list reader
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

#ifndef EDGE_LIST
#define EDGE_LIST

#pragma once

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <unordered_map>
#include "rabbit_order.h"

namespace edge_list {
namespace aux {

using rabbit_order::vint;
typedef std::tuple<vint, vint, float> edge;

off_t file_size(const int fd);

struct file_desc {
  int fd;
  file_desc(const std::string& path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
      RO_DIE << "open(2): " << strerror(errno);
  }
  ~file_desc() {
    if (close(fd) != 0)
      RO_DIE << "close(2): " << strerror(errno);
  }
};

struct mmapped_file {
  size_t      size;
  const char* data;
  mmapped_file(const file_desc& fd) {
    size = file_size(fd.fd);
    data = static_cast<char*>(
        mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd.fd, 0));
    if (data == NULL)
      RO_DIE << "mmap(2): " << strerror(errno);
  }
  ~mmapped_file() {
    if (munmap(const_cast<char*>(data), size) != 0)
      RO_DIE << "munmap(2): " << strerror(errno);
  }
};

struct edge_parser {
  const char* const strfirst;
  const char* const strlast;
  const char*       crr;

  edge_parser(const char* const first, const char* const last)
      : strfirst(first), strlast(last), crr(first) {}

  template<typename OutputIt>
  void operator()(OutputIt dst) {
    while (crr < strlast) {
      eat_empty_lines();
      if (crr < strlast)
        *dst++ = eat_edge();
    }
  }

  edge eat_edge() {
    const vint s = eat_id();
    eat_separator();
    const vint t = eat_id();
    return edge {s, t, 1.0};  // FIXME: edge weight is not supported so far
  }

  vint eat_id() {
    //
    // Naive implementation is faster than library functions such as `atoi` and
    // `strtol`
    //
    const auto _crr = crr;
    vint       v    = 0;
    for (; crr < strlast && std::isdigit(*crr); ++crr) {
      const vint _v = v * 10 + (*crr - '0');
      if (_v < v)  // overflowed
        RO_DIE << "Too large vertex ID at line " << crr_line();
      v = _v;
    }
    if (_crr == crr)  // If any character has not been eaten
      RO_DIE << "Invalid vertex ID at line " << crr_line();
    return v;
  }

  void eat_empty_lines() {
    while (crr < strlast) {
      if      (*crr == '\r') ++crr;                                // Empty line
      else if (*crr == '\n') ++crr;                                // Empty line
      else if (*crr == '#' ) crr = std::find(crr, strlast, '\n');  // Comment
      else break;
    }
  }

  void eat_separator() {
    while (crr < strlast && (*crr == '\t' || *crr == ',' || *crr == ' '))
      ++crr;
  }

  // Only for error messages
  size_t crr_line() {
    return std::count(strfirst, crr, '\n');
  }
};

std::vector<edge> read(const std::string& path);

}  // namespace edge_list::aux

using aux::edge;
using aux::read;

}  // namespace edge_list

#endif
