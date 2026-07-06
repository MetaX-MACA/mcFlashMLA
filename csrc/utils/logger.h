/**
 * @file logger.h
 * @brief
 * @version 0.1
 * @date 2021-09-02
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef UTILS_LOGGER_H_
#define UTILS_LOGGER_H_

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief log init
 * usually do not use it. it will cover settings in environment
 * Only for tests you want run locally
 */
#define LOG_INIT(enable, level, output, filename)                                                  \
    mcFlashAttn::utils::Logger::GetLogger().Init(level, output, filename)

/**
 * @brief  log formating
 * LOG_F(mcFlashAttn::utils::LOG_INFO, "%s\n", "enjoy");
 * use quick api like LOG_INFO("%s\n", dflakjf);
 */
#define LOG_F(level, fmt, ...)                                                                     \
    do {                                                                                           \
        if (mcFlashAttn::utils::Logger::GetLogger().ExternalEn() &&                                      \
            mcFlashAttn::utils::Logger::Filter(level)) {                                                 \
            mcFlashAttn::utils::Logger::GetLogger().Log(level,                                           \
                                                  "%s:%d |\t" fmt,                                 \
                                                  mcFlashAttn::utils::basename(__FILE__).c_str(),        \
                                                  __LINE__,                                        \
                                                  ##__VA_ARGS__);                                  \
        }                                                                                          \
    } while (false)

#define LOG_ERR(fmt, ...)   LOG_F(mcFlashAttn::utils::LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  LOG_F(mcFlashAttn::utils::LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  LOG_F(mcFlashAttn::utils::LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG_F(mcFlashAttn::utils::LOG_DEBUG, fmt, ##__VA_ARGS__)


/**
 * @brief  log format simple
 * for logs you don't want header and timestamp
 * For example log in a for loop
 */
#define LOG_FS(level, fmt, ...)                                                                    \
    do {                                                                                           \
        if (mcFlashAttn::utils::Logger::GetLogger().ExternalEn() &&                                      \
            mcFlashAttn::utils::Logger::Filter(mcFlashAttn::utils::LOG_##level)) {                             \
            mcFlashAttn::utils::Logger::GetLogger().LogSimple(fmt, ##__VA_ARGS__);                       \
        }                                                                                          \
    } while (false)

#define LOG_FS_ERR(fmt, ...)   LOG_FS(ERROR, fmt, ##__VA_ARGS__)
#define LOG_FS_WARN(fmt, ...)  LOG_FS(WARNING, fmt, ##__VA_ARGS__)
#define LOG_FS_INFO(fmt, ...)  LOG_FS(INFO, fmt, ##__VA_ARGS__)
#define LOG_FS_DEBUG(fmt, ...) LOG_FS(DEBUG, fmt, ##__VA_ARGS__)

/**
 * @brief log external
 *
 *      log external's format is more simple than internal's. Only three levels err, warn and info
 *           are available!
 */
#define LOG_EXT(level, fmt, ...)                                                                   \
    do {                                                                                           \
        if (mcFlashAttn::utils::Logger::GetLogger().ExternalEn() &&                                      \
            mcFlashAttn::utils::Logger::Filter(level)) {                                                 \
            mcFlashAttn::utils::Logger::GetLogger().LogExternal(level, fmt, ##__VA_ARGS__);              \
        }                                                                                          \
    } while (false)

#define LOG_EXT_ERR(fmt, ...)  LOG_EXT(mcFlashAttn::utils::LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_EXT_WARN(fmt, ...) LOG_EXT(mcFlashAttn::utils::LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_EXT_INFO(fmt, ...) LOG_EXT(mcFlashAttn::utils::LOG_INFO, fmt, ##__VA_ARGS__)

/**
 * @brief  log stream
 *  Attention:
 *      LOG_S is not recommand to use, when internal log is disabled, StreamLogger function
 *          operator << still run,  though nothing will print because of the Filter function
 *          in that situation.
 *  Examples:
 *      LOG_S(INFO) << a << b << "\n";
 *      LOG_S(WARNING) << a << b << "\n";
 *
 *  ...
 */
#define LOG_S(level) mcFlashAttn::utils::StreamLogger(mcFlashAttn::utils::LOG_##level, __FILE__, __LINE__)

/**
 * @brief  print shape log
 */
#define LOG_FS_SHAPE(fmt, ...)                                                                     \
    do {                                                                                           \
            mcFlashAttn::utils::Logger::GetLogger().LogSimple(fmt, ##__VA_ARGS__);                 \
    } while (false)

#define LOG_SHAPE(fmt, ...) LOG_FS_SHAPE(fmt, ##__VA_ARGS__)

namespace mcFlashAttn {
namespace utils {

enum LogLevel : int32_t {
    LOG_ERROR = 0,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG,
};

enum LogOutput : int32_t {
    LOG_STDOUT = 0,
    LOG_FILE,
    LOG_SYSLOG,
};

std::string basename(const std::string &in);

// control env variables
#define MHA_LOG_ENABLE   "MHA_LOG_ENABLE"
#define MHA_LOG_LEVEL    "MHA_LOG_LEVEL"
#define MHA_LOG_OUTPUT   "MHA_LOG_OUTPUT"

class Logger {
 public:
    Logger();
    void Init(int level, int output, const std::string &filename) {
        if (!env_logext_enable_set_) {  // env variable has higher priority
            external_en_ = true;
        }
        if (!env_log_level_set_) {  // env variable has higher priority
            SetLogLevel(level);
        }
        if (!env_log_output_set_) {  // env variable has higher priority
            log_output_ = output;
        }

        if (logfile_.is_open()) {
            logfile_.close();
        }
        filename_ = filename;

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

    static Logger &GetLogger();
    static bool Filter(int level) { return level <= GetLogger().log_level_; }

    bool ExternalEn() const { return external_en_; }
    void SetLogLevel(int level);

    void Log(int level, const char *fmt, ...);
    /**
     * @brief For usage without header "info [time] xxx"
     * usally if you want log something in a for loop without header
     * and crlf, use this api.
     * @param fmt
     * @param ...
     */
    void LogSimple(const char *fmt, ...);

    /**
     * @brief For usage without header "info [time] xxx"
     */
    void LogExternal(int level, const char *fmt, ...);

 private:
    using LoggerClock = std::chrono::high_resolution_clock;
    static uint64_t MicroSeconds() {
        std::chrono::duration<uint64_t, std::micro> ts =
            std::chrono::duration_cast<std::chrono::microseconds>(
                LoggerClock::now().time_since_epoch());
        return ts.count();
    }
    void IntervalCheck();

    static uint32_t TimeStr(char *buf, uint32_t bufsize);
    static uint32_t FmtLogHeader(char *buf, uint32_t bufsize, int level);

    int log_level_       = LOG_INFO;
    uint32_t log_output_ = LOG_STDOUT;

    std::string ident_ = "mcFlashAttn";
    std::ofstream logfile_;
    std::ostream *of_ = &std::cout;
    std::mutex mutex_;
    std::string filename_ = "mcFlashAttn.txt";
    uint32_t counter_     = 0;
    bool env_logext_enable_set_{false};
    bool env_log_level_set_{false};
    bool env_log_output_set_{false};
    bool external_en_ = false;

    static constexpr int kCheckInterval = 1000;
    static const char *LogLevelMap[];
};

class StreamLogger {
 public:
    StreamLogger(int level, const char *filename, int linenum)
        : level_(level), filename_(filename), linenum_(linenum) {}
    ~StreamLogger() { Flush(); }
    void Flush() {
        if (Filter()) {
            Logger::GetLogger().Log(level_,
                                    "%s:%d |\t%s",
                                    basename(filename_).c_str(),
                                    linenum_,
                                    ss_.str().c_str());
            ss_.clear();
        }
    }

    template <class T>
    StreamLogger &operator<<(const T &v) {
        if (Filter()) {
            ss_ << v;
        }
        return *this;
    }

    StreamLogger &operator<<(std::ostream &(*f)(std::ostream &)) {
        if (Filter()) {
            f(ss_);
        }
        return *this;
    }

 private:
    bool Filter() const { return Logger::Filter(level_) && Logger::GetLogger().ExternalEn(); }
    int level_;
    std::string filename_;
    int linenum_;
    std::stringstream ss_;
};

}  // namespace utils
}  // namespace mcFlashAttn

#endif  // UTILS_LOGGER_H_
