#ifndef CONFIG_PARSER_H_
#define CONFIG_PARSER_H_

#include <boost/program_options.hpp>
#include <string>
#include <vector>

#include "Config.h"

class ConfigParser {
 public:
  ConfigParser() = default;
  auto parseCommandline(int argc, char** argv) -> bool;
  auto createConfig() -> Config;

 private:
  boost::program_options::variables_map variables_map_{};
  const char* print_prefix = "[Config] ";
};

#endif  // CONFIG_PARSER_H_