#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "flash_mla.h"


void shape_print(mcFlashAttn::Flash_fwd_mla_params params, bool Is_causal, const std::string& debug_flag);

void shape_print(SparsePrefillParams params, bool Is_causal, const std::string& debug_flag);

void debug_print(mcFlashAttn::Flash_fwd_mla_params params, const std::string& debug_flag);

void debug_print(SparsePrefillParams params, const std::string& debug_flag);