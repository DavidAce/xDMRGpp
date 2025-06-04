#pragma once
#include "config/enums.h"
#include "debug/exceptions.h"
#include "io/filesystem.h"
#include "math/cast.h"
#include "tid/enums.h"
#include "tools/common/log.h"
#include <string>
#include <unordered_map>

class Loader {
    public:
    bool               file_exists;
    fs::path           file_path;
    fs::file_time_type file_date;

    explicit Loader(const fs::path &cfg_or_h5_file);

    void                      load();
    void                      unload();
    [[nodiscard]] std::string get_config_file_as_string();

    template<typename T>
    void load_parameter(const std::string &param_name, T &param_value) {
        try {
            T new_value = find_parameter<T>(param_name);
            param_value = new_value;
            if constexpr(std::is_enum_v<T>) {
                if constexpr(enum_is_bitflag_v<T>)
                    tools::log->debug("Loaded parameter: {:<64} = {:<20}", param_name, flag2str(param_value));
                else
                    tools::log->debug("Loaded parameter: {:<64} = {:<20}", param_name, enum2sv<T>(param_value));

            } else
                tools::log->debug("Loaded parameter: {:<64} = {:<20}", param_name, param_value);

        } catch(std::exception &ex) { tools::log->warn("Failed to read parameter [{}]: {}", param_name, ex.what()); }
    }

    private:
    std::unordered_map<std::string, std::string>     param_map;
    [[nodiscard]] static std::string                 remove_spaces(std::string str);
    [[nodiscard]] static std::string                 remove_leading_spaces(std::string str, std::string_view whitespace = " \t");
    [[nodiscard]] static bool                        is_parameterline(std::string_view s);
    [[nodiscard]] static std::string_view::size_type find_comment_character(std::string_view s);

    template<typename T>
    [[nodiscard]] T parse_param(const std::string &param_val, std::string_view param_name) {
        try {
            if(param_val.empty()) {
                if constexpr(std::is_same_v<T, std::string>) return std::string();
                throw except::range_error("Parameter [{}] has no value", param_name);
            }
            if constexpr(std::is_same_v<T, unsigned int>) {
                auto val = std::stoi(param_val);
                if(val < 0) throw except::runtime_error("Read negative value for unsigned parameter: {}", val);
                return static_cast<T>(val);
            }
            if constexpr(std::is_same_v<T, unsigned long>) return std::stoul(param_val);
            if constexpr(std::is_same_v<T, unsigned long long>) return std::stoull(param_val);
            if constexpr(std::is_same_v<T, int>) return std::stoi(param_val);
            if constexpr(std::is_same_v<T, long>) return std::stol(param_val);
            if constexpr(std::is_same_v<T, long long>) return std::stoll(param_val);
            if constexpr(std::is_same_v<T, size_t>) return std::stoul(param_val);
            if constexpr(std::is_same_v<T, double>) return std::stod(param_val);
            if constexpr(std::is_same_v<T, float>) return std::stof(param_val);
            if constexpr(std::is_enum_v<T>) return sv2enum<T>(param_val);
            if constexpr(std::is_same_v<T, std::string>) {
                if(param_val == "\"\"")
                    return std::string();
                else
                    return param_val;
            }
            if constexpr(std::is_same_v<T, bool>) {
                if(param_val == "true") return true;
                if(param_val == "false") return false;
                throw except::runtime_error("Expected true or false, got {}", param_val);
            }
            throw except::runtime_error("Type mismatch on parameter: {}", param_val);
        } catch(std::exception &ex) { throw except::runtime_error("Error parsing param: {}", ex.what()); } catch(...) {
            throw except::runtime_error("Error parsing param: Unknown error");
        }
    }

    template<typename T>
    [[nodiscard]] T find_parameter(std::string param_requested) {
        param_requested = remove_spaces(param_requested);
        std::transform(param_requested.begin(), param_requested.end(), param_requested.begin(), ::tolower);
        try {
            if(param_map.find(param_requested) != param_map.end()) {
                return parse_param<T>(param_map[param_requested], param_requested);
            } else
                throw except::range_error("Config file does not specify the requested parameter: [{}]", param_requested);
        } catch(std::range_error &ex) { throw std::runtime_error(ex.what()); } catch(std::exception &ex) {
            throw except::runtime_error("Error parsing the requested parameter: [{}]: {}", param_requested, ex.what());
        }
    }
};