#include "config/enums/StorageEvent.h"
#include "debug/exceptions.h"
#include "io/hdf5_types.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <fmt/format.h>
#include <h5pp/h5pp.h>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <sys/wait.h>
#include <vector>

namespace fs = std::filesystem;

namespace {
    struct StatusSnapshot {
        std::vector<StorageEvent> events;
        std::vector<uint64_t>     iters;
    };

    [[nodiscard]] std::string quote(const fs::path &path) { return fmt::format("\"{}\"", path.string()); }

    [[nodiscard]] std::string read_text_file(const fs::path &path) {
        std::ifstream in(path);
        if(not in) throw except::runtime_error("Failed to open file for reading: {}", path.string());
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    void write_text_file(const fs::path &path, std::string_view text) {
        std::ofstream out(path);
        if(not out) throw except::runtime_error("Failed to open file for writing: {}", path.string());
        out << text;
        if(not out) throw except::runtime_error("Failed to write file: {}", path.string());
    }

    void run_command(std::string_view command) {
        fmt::print("Running: {}\n", command);
        int status = std::system(std::string(command).c_str());
        if(status == -1) throw except::runtime_error("Failed to launch command: {}", command);
        if(not WIFEXITED(status) or WEXITSTATUS(status) != 0)
            throw except::runtime_error("Command failed with status {}: {}", status, command);
    }

    [[nodiscard]] StatusSnapshot read_status_snapshot(const fs::path &h5path) {
        h5pp::File file(h5path, h5pp::FileAccess::READONLY);
        auto       status_path = std::string("xDMRG/state_emid/status");
        if(not file.linkExists(status_path)) throw except::runtime_error("Missing status table in {}", h5path.string());

        StatusSnapshot snapshot;
        snapshot.events = file.readTableField<std::vector<StorageEvent>>(status_path, "event", h5pp::TableSelection::ALL);
        snapshot.iters  = file.readTableField<std::vector<uint64_t>>(status_path, "iter", h5pp::TableSelection::ALL);
        if(snapshot.events.size() != snapshot.iters.size())
            throw except::runtime_error("Status table fields have different lengths in {}", h5path.string());
        return snapshot;
    }

    [[nodiscard]] size_t count_event(const std::vector<StorageEvent> &events, StorageEvent event) {
        return static_cast<size_t>(std::count(events.begin(), events.end(), event));
    }

    [[nodiscard]] std::string replace_setting(std::string content, std::string_view key, std::string_view value) {
        auto pattern = std::regex(fmt::format(R"((^|\n){}\s*=\s*[^\n]*)", key));
        if(not std::regex_search(content, pattern)) throw except::runtime_error("Failed to find {} in the resume input file", key);
        return std::regex_replace(content, pattern, fmt::format("$1{}                                                 = {}", key, value),
                                  std::regex_constants::format_first_only);
    }

    [[nodiscard]] std::string make_run_config(const fs::path &template_cfg, const fs::path &run_cfg, size_t iter_max, std::string_view collision_policy) {
        auto content = read_text_file(template_cfg);
        content      = replace_setting(std::move(content), "xdmrg::iter_max", fmt::format("{}", iter_max));
        content      = replace_setting(std::move(content), "storage::file_collision_policy", collision_policy);
        content      = replace_setting(std::move(content), "storage::resume_policy", "IF_MAX_ITERS");
        content      = replace_setting(std::move(content), "storage::file_resume_policy", "FULL");
        content      = replace_setting(std::move(content), "storage::file_resume_name", "\"state_emid\"");
        content      = replace_setting(std::move(content), "storage::file_resume_iter", "-1ul");
        write_text_file(run_cfg, content);
        return run_cfg.string();
    }
} // namespace

int main() {
    auto input_cfg    = fs::path(XDMRG_RESUME_INPUT);
    auto output_dir   = fs::current_path() / "tests" / "xdmrg-resume";
    auto output_path  = output_dir / "output.h5";
    auto first_cfg    = output_dir / "first-run.cfg";
    auto resume_cfg   = output_dir / "resume-run.cfg";
    auto state_prefix = std::string("xDMRG/state_emid");

    fs::create_directories(output_dir);
    fs::remove(output_path);
    fs::remove(first_cfg);
    fs::remove(resume_cfg);

    auto first_cfg_path  = make_run_config(input_cfg, first_cfg, 2, "REPLACE");
    auto resume_cfg_path = make_run_config(input_cfg, resume_cfg, 4, "RESUME");

    run_command(fmt::format("{} --replace -t 1 -c {}", quote(XDMRG_BINARY), quote(first_cfg_path)));

    {
        h5pp::File file(output_path, h5pp::FileAccess::READONLY);
        if(not file.linkExists(state_prefix)) throw except::runtime_error("Missing resumed state prefix in {}", output_path.string());
        auto algorithm_can_resume = file.readAttribute<bool>(state_prefix, "algorithm_can_resume");
        auto algorithm_stop       = file.readAttribute<std::string>(state_prefix, "algorithm_stop");
        if(not algorithm_can_resume) throw except::runtime_error("Expected xDMRG/state_emid to be resumable");
        if(algorithm_stop != "MAX_ITERS") throw except::runtime_error("Expected first run to stop with MAX_ITERS, got {}", algorithm_stop);
    }

    auto first_snapshot      = read_status_snapshot(output_path);
    auto first_model_events  = count_event(first_snapshot.events, StorageEvent::MODEL);
    auto first_finish_events = count_event(first_snapshot.events, StorageEvent::FINISHED);
    if(first_snapshot.iters.empty()) throw except::runtime_error("First run did not write any xDMRG status entries");
    if(first_model_events != 1) throw except::runtime_error("Expected exactly one MODEL event after the first run, got {}", first_model_events);
    if(first_finish_events != 1) throw except::runtime_error("Expected exactly one FINISHED event after the first run, got {}", first_finish_events);

    auto first_last_iter = first_snapshot.iters.back();
    if(first_last_iter < 1) throw except::runtime_error("Expected first run to advance at least one iteration, got {}", first_last_iter);

    run_command(fmt::format("{} --resume -t 1 -c {}", quote(XDMRG_BINARY), quote(resume_cfg_path)));

    auto second_snapshot      = read_status_snapshot(output_path);
    auto second_model_events  = count_event(second_snapshot.events, StorageEvent::MODEL);
    auto second_finish_events = count_event(second_snapshot.events, StorageEvent::FINISHED);
    if(second_snapshot.iters.size() <= first_snapshot.iters.size())
        throw except::runtime_error("Resume run did not append any new xDMRG status entries");
    if(second_snapshot.iters.back() <= first_last_iter)
        throw except::runtime_error("Resume run did not advance the iteration counter: {} -> {}", first_last_iter, second_snapshot.iters.back());
    if(second_model_events != first_model_events)
        throw except::runtime_error("Resume run added a new MODEL event, which indicates restart instead of resume: {} -> {}", first_model_events,
                                    second_model_events);
    if(second_finish_events < 2)
        throw except::runtime_error("Expected two FINISHED events after the resume run, got {}", second_finish_events);

    fmt::print("xDMRG resume test passed: iter {} -> {}, MODEL events {}, FINISHED events {}\n", first_last_iter, second_snapshot.iters.back(),
               second_model_events, second_finish_events);
    return 0;
}
