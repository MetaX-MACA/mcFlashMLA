#pragma once

#include <string>
#include <vector>
#include <sstream>

template <typename T>
void concat_str(std::stringstream &total_string, std::vector<T> info);

std::stringstream concat_total_strs(const std::string& prefix_flag, const std::string& debug_flag, std::vector<std::string> bool_info, std::vector<std::string> shape_info);