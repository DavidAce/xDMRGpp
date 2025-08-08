#pragma once
#include "meld-io/id.h"
#include "meld-io/meta.h"
#include <h5pp/details/h5ppInfo.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace h5pp {
    class File;
}

namespace tools::h5xf {
    void transferEnvironment(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtEnvDb, const h5pp::File &h5_src,
                             std::string tgtEnvBase, const FileId &fileId);

    void transferDatasets(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::DsetInfo>> &tgtDsetDb, const h5pp::File &h5_src,
                          std::unordered_map<std::string, h5pp::DsetInfo> &srcDsetDb, const PathId &pathid, const std::vector<DsetKey> &srcDsetKeys,
                          const FileId &fileId);

    void transferTables(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtTableDb,
                        std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<TableKey> &srcTableKeys,
                        const FileId &fileId);
    template<typename KeyT>
    void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                        std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<KeyT> &srcKeys,
                        const FileId &fileId);
}