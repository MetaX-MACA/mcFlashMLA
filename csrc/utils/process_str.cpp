#include "process_str.h"

template <typename T>
void concat_str(std::stringstream &total_string, std::vector<T> info){
    for(int i=0; i < info.size(); i++) {
        if(i != 0) {
            total_string << "," << info[i];
        }
        else {
            total_string << info[i];
        }
    }
}

std::stringstream concat_total_strs(const std::string& prefix_flag, const std::string& debug_flag, std::vector<std::string> bool_info, std::vector<std::string> shape_info) {
    std::stringstream total_strs;
    total_strs << "[";
    total_strs << prefix_flag;
    total_strs << "]  ";

    total_strs << debug_flag;
    total_strs << "_Bool_Switch:";
    concat_str(total_strs, bool_info);
    total_strs << "   ";

    total_strs << debug_flag;
    total_strs << "_Shape_Info:";
    concat_str(total_strs, shape_info);

    return total_strs;
}