#pragma once
#include "config/enums.h"
#include <string_view>
class AlgorithmStatus;
struct StorageInfo {
    public:
    size_t        iter      = -1ul;
    size_t        step      = -1ul;
    long          position  = -1;
    long          direction = 0;
    long          bond_lim  = -1;
    long          bond_max  = -1;
    double        trnc_lim  = -1;
    AlgorithmType algo_type = AlgorithmType::ANY;
    AlgorithmStop algo_stop = AlgorithmStop::NONE;
    std::string   algo_name;
    ModelType     model_type;
    StorageEvent  storage_event = StorageEvent::NONE;
    std::string   state_name;
    bool          algorithm_has_finished  = false;
    bool          algorithm_has_succeeded = false;
    void          assert_well_defined() const;
    StoragePolicy get_table_storage_policy(std::string_view table_path) const;
    StoragePolicy get_dataset_storage_policy(std::string_view dset_path) const;
    StoragePolicy get_mpo_storage_policy(std::string_view model_path) const;
    StoragePolicy get_state_storage_policy() const;
    std::string   get_state_prefix() const;
    std::string   get_mps_prefix() const;
    StorageInfo(const AlgorithmStatus &status, std::string_view state_name, ModelType model_type);
};

// Metadata needed to uniquely identify a save point for a given HDF5 link
struct StorageAttrs {
    size_t       iter          = -1ul;
    size_t       step          = -1ul;
    long         bond_lim      = -1;
    long         bond_max      = -1;
    double       trnc_lim      = -1;
    StorageEvent storage_event = StorageEvent::NONE;
    mutable bool link_exists   = false;
    StorageAttrs()             = default;
    StorageAttrs(const StorageInfo &sinfo);
    bool operator==(const StorageAttrs &sinfo);
    bool operator==(const StorageInfo &sinfo);
};
