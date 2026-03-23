
#include "config/enums/AlgorithmStop.h"
#include "config/enums/AlgorithmType.h"
#include "config/enums/ResumePolicy.h"
#include "config/enums/StorageEvent.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include <h5pp/h5pp.h>

std::string tools::common::h5::resume::extract_state_name(std::string_view state_prefix) {
    constexpr std::string_view state_pattern = "state_";
    std::string                state_name;
    std::string::size_type     start_pos = state_prefix.find(state_pattern);
    if(start_pos != std::string::npos) {
        std::string::size_type len = state_prefix.find('/', start_pos) - start_pos;
        state_name                 = state_prefix.substr(start_pos, len); // E.g. "/xDMRG/state_emid/results/..." would match state_emid
    }
    return state_name;
}

std::vector<std::string> tools::common::h5::resume::find_state_prefixes(const h5pp::File &h5file, AlgorithmType algo_type, std::string_view name) {
    std::string_view algo_name = enum2sv(algo_type);
    if(name.empty())
        tools::log->info("Searching for states for algorithm [{}] in file [{}]", algo_name, h5file.getFilePath());
    else
        tools::log->info("Searching for states matching [{}] for algorithm [{}] in file [{}]", name, algo_name, h5file.getFilePath());
    auto                     algo_prefix = fmt::format("/{}", algo_name);
    auto                     matches     = h5file.findGroups(name, algo_prefix, -1, 1);
    std::vector<std::string> state_prefixes;
    for(const auto &match : matches) {
        state_prefixes.emplace_back(fmt::format("{}/{}", algo_prefix, match));
        tools::log->debug(" -- found state prefix: {}", state_prefixes.back());
    }
    return state_prefixes;
}

std::vector<tools::common::h5::MpsInfo> tools::common::h5::resume::find_fully_stored_mps(const h5pp::File &h5file, std::string_view state_prefix) {
    std::vector<MpsInfo> mps_info;
    for(const auto &pfx : h5file.findGroups("mps", state_prefix)) {
        auto path  = fmt::format("{}/{}", state_prefix, pfx);
        auto iter  = h5file.readAttribute<size_t>(path, "iter");
        auto step  = h5file.readAttribute<size_t>(path, "step");
        auto event = h5file.readAttribute<std::optional<StorageEvent>>(path, "storage_event").value();
        mps_info.emplace_back(MpsInfo{path, iter, step, event});
    }
    // Sort according to step
    std::sort(begin(mps_info), end(mps_info), [](const MpsInfo &i1, const MpsInfo &i2) { return i1.step < i2.step; });
    return mps_info;
}

std::optional<AlgorithmStop> tools::common::h5::resume::find_algorithm_stop_reason(const h5pp::File &h5file, std::string_view state_prefix) {
    auto algorithm_stop_opt = h5file.readAttribute<std::optional<std::string>>(state_prefix, "algorithm_stop");
    if(algorithm_stop_opt.has_value()) return sv2enum<AlgorithmStop>(algorithm_stop_opt.value());

    // We try to read off the status field manually
    auto status_prefix = fmt::format("{}/status", state_prefix);
    if(h5file.linkExists(status_prefix)) {
        // Try to find the last "FINISHED" event
        auto events_and_stops =
            h5file.readTableField<std::vector<std::pair<StorageEvent, AlgorithmStop>>>(status_prefix, {"event", "algo_stop"}, h5pp::TableSelection::ALL);
        for(const auto &event_and_stop : iter::reverse(events_and_stops)) {
            tools::log->debug("-- checking status: {} - {}", enum2sv(event_and_stop.first), enum2sv(event_and_stop.second));
            if(event_and_stop.first == StorageEvent::FINISHED) { return event_and_stop.second; }
        }
    }

    // We could not figure out the stop reason ...
    return std::nullopt;
}

bool tools::common::h5::resume::check_algorithm_can_resume(const h5pp::File &h5file, std::string_view state_prefix) {
    auto algorithm_can_resume = h5file.readAttribute<std::optional<bool>>(state_prefix, "algorithm_can_resume");
    if(not algorithm_can_resume.has_value()) {
        // This may be an old file. We can detect the required mps and model manually and hope for the best
        auto algo_prefix     = state_prefix.substr(0, state_prefix.find_first_of('/'));
        auto mps_is_saved    = h5file.linkExists(fmt::format("{}/mps", state_prefix));
        auto model_is_saved  = h5file.linkExists(fmt::format("{}/model/hamiltonian", algo_prefix));
        algorithm_can_resume = mps_is_saved and model_is_saved;
    }
    return algorithm_can_resume.value();
}

std::vector<std::pair<std::string, AlgorithmStop>> tools::common::h5::resume::find_states_that_may_resume(const h5pp::File &h5file, ResumePolicy resume_policy,
                                                                                                          AlgorithmType    algo_type,
                                                                                                          std::string_view state_pattern) {
    // Let's first consider the new resume check method, which looks for attributes on the state_prefix.
    auto algo_prefix    = algo_type == AlgorithmType::ANY ? "/" : enum2sv(algo_type);
    auto state_prefixes = h5file.findGroups(state_pattern, algo_prefix, -1, 1); // MaxDepth == 1 considers states directly under algo, e.g. xDMRG/state_emid
    if(state_prefixes.empty()) return {};
    // Prepend the algo_prefix to each match
    if(algo_type != AlgorithmType::ANY)
        for(auto &prefix : state_prefixes) prefix = fmt::format("{}/{}", algo_prefix, prefix);

    auto states_that_may_resume = std::vector<std::pair<std::string, AlgorithmStop>>();
    for(const auto &state_prefix : state_prefixes) {
        auto algorithm_can_resume = check_algorithm_can_resume(h5file, state_prefix);
        if(algorithm_can_resume) {
            // We found a resumable state! Now let's check if it should resume.
            auto algorithm_stop = find_algorithm_stop_reason(h5file, state_prefix);
            if(not algorithm_stop.has_value()) {
                // We could not figure out the stop reason: we can't apply the policy if we don't know why the previous simulation stopped.
                tools::log->trace("Found a resumable state with unknown stop reason: {}", state_prefix);
                continue;
            }

            // Whether we resume depends on the resume policy
            tools::log->trace("Found a resumable state: {} | stop reason: {} | resume policy: {}", state_prefix, enum2sv(algorithm_stop.value()),
                              enum2sv(resume_policy));
            switch(resume_policy) {
                case ResumePolicy::IF_MAX_ITERS: {
                    if(algorithm_stop == AlgorithmStop::MAX_ITERS) states_that_may_resume.emplace_back(std::make_pair(state_prefix, algorithm_stop.value()));
                    break;
                }
                case ResumePolicy::IF_SATURATED: {
                    if(algorithm_stop == AlgorithmStop::SATURATED) states_that_may_resume.emplace_back(std::make_pair(state_prefix, algorithm_stop.value()));
                    break;
                }
                case ResumePolicy::IF_UNSUCCESSFUL: {
                    if(algorithm_stop != AlgorithmStop::SUCCESS) states_that_may_resume.emplace_back(std::make_pair(state_prefix, algorithm_stop.value()));
                    break;
                }
                case ResumePolicy::IF_SUCCESSFUL: {
                    if(algorithm_stop == AlgorithmStop::SUCCESS) states_that_may_resume.emplace_back(std::make_pair(state_prefix, algorithm_stop.value()));
                    break;
                }
                case ResumePolicy::ALWAYS: {
                    states_that_may_resume.emplace_back(std::make_pair(state_prefix, algorithm_stop.value()));
                    break;
                }
            }
        }
    }
    return states_that_may_resume;
}
