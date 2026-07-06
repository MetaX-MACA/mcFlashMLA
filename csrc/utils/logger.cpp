/**
 * @file logger.cpp
 * @brief
 * @version 0.1
 * @date 2021-09-02
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "logger.h"

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace mcFlashAttn {
namespace utils {

const char *Logger::LogLevelMap[] =
    {"\033[0;31mERR\033[0m", "WARN", "INFO", "DBG"};

std::string basename(const std::string &in) {
    auto loc = in.find_last_of("/");
    if (loc == std::string::npos) {
        return in;
    }
    return in.substr(loc + 1);
}

Logger &Logger::GetLogger() {
    static Logger s_log;
    return s_log;
}

Logger::Logger() {
    // log switch
    char *p = getenv(MHA_LOG_ENABLE);
    if(p != nullptr) {
        external_en_ = true;
        env_logext_enable_set_ = true;
    }

    // log level
    char *env_log_level = getenv(MHA_LOG_LEVEL);
    if (env_log_level != nullptr) {
        int level = strtol(env_log_level, nullptr, 10);
        SetLogLevel(level);
        env_log_level_set_ = true;
    }

    // output direction
    char *env_output = getenv(MHA_LOG_OUTPUT);
    if (env_output != nullptr) {
        if (strcmp(env_output, "stdout") == 0) {
            log_output_ = LOG_STDOUT;
        } else if (strcmp(env_output, "file") == 0) {
            log_output_ = LOG_FILE;
        } else if (strcmp(env_output, "syslog") == 0) {
            log_output_ = LOG_SYSLOG;
        }
        env_log_output_set_ = true;
    }

    if (external_en_) {
        switch (log_output_) {
            case LOG_STDOUT:
                of_ = &std::cout;
                break;
            default:
            case LOG_FILE:
                logfile_.open(filename_.c_str());
                of_ = &logfile_;
                break;
        }
    }
}

void Logger::SetLogLevel(int level) {
    log_level_ = std::min(level, static_cast<int32_t>(LOG_INFO));
}

uint32_t Logger::TimeStr(char *buf, uint32_t bufsize) {
    time_t t;
    struct tm *timeinfo;
    std::time(&t);
    timeinfo = std::localtime(&t);
    return std::strftime(buf, bufsize, ":%D %s: ", timeinfo);
}

uint32_t Logger::FmtLogHeader(char *buf, uint32_t bufsize, int level) {
    uint32_t off = 0;
    off += snprintf(buf + off, bufsize - off, "[MHA][%s] ", LogLevelMap[level]);
    off += TimeStr(buf + off, bufsize - off);
    return off;
}

void Logger::IntervalCheck() {
    ++counter_;
    if (counter_ >= kCheckInterval) {
        counter_ = 0;
    }
}

/*
    @brief: log simple
*/

void Logger::LogSimple(const char *fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_output_ != LOG_SYSLOG) {
        char message[4096];
        uint32_t off = 0;

        va_list args;
        va_start(args, fmt);
        off += std::vsnprintf(message + off, sizeof(message) - off, fmt, args);
        va_end(args);

        *of_ << message;
    }

    IntervalCheck();
}

/*
    @brief: log external
*/

void Logger::LogExternal(int level, const char *fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_output_ != LOG_SYSLOG) {
        char message[4096];
        uint32_t off = 0;

        off += snprintf(message, sizeof(message), "[MHA][%s] ", LogLevelMap[level]);

        va_list args;
        va_start(args, fmt);
        off += std::vsnprintf(message + off, sizeof(message) - off, fmt, args);
        va_end(args);

        *of_ << message;
    }
}

void Logger::Log(int level, const char *fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_output_ != LOG_SYSLOG) {
        char message[4096];
        uint32_t off = 0;

        off += FmtLogHeader(message, sizeof(message), level);

        va_list args;
        va_start(args, fmt);
        off += std::vsnprintf(message + off, sizeof(message) - off, fmt, args);
        va_end(args);
        *of_ << message;
    }

    IntervalCheck();
}

}  // namespace utils
}  // namespace mcFlashAttn
