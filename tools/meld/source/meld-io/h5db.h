#pragma once
#include "general/enums.h"
#include "meld-io/id.h"
#include "meld-io/meta.h"
#include <h5pp/details/h5ppFilesystem.h>
#include <h5pp/details/h5ppInfo.h>
#include <string>
#include <unordered_map>
#include <vector>
namespace h5pp {
    class File;
}

namespace tools::h5db {

    struct Keys {
        std::vector<DsetKey>  dsets;
        std::vector<TableKey> tables;
        std::vector<CronoKey> cronos;
        std::vector<FesUpKey> fesups;
        std::vector<RbdsKey>  rbdses;
        std::vector<RtesKey>  rteses;
        std::vector<ModelKey> models;

        [[nodiscard]] std::vector<std::string> get_algos() const {
            std::vector<std::string> v;
            for(auto &d : dsets) v.emplace_back(d.algo);
            for(auto &d : tables) v.emplace_back(d.algo);
            for(auto &d : cronos) v.emplace_back(d.algo);
            for(auto &d : fesups) v.emplace_back(d.algo);
            for(auto &d : rbdses) v.emplace_back(d.algo);
            for(auto &d : rteses) v.emplace_back(d.algo);
            for(auto &d : models) v.emplace_back(d.algo);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            return v;
        }
        [[nodiscard]] std::vector<std::string> get_states() const {
            std::vector<std::string> v;
            for(auto &d : dsets) v.emplace_back(d.state);
            for(auto &d : tables) v.emplace_back(d.state);
            for(auto &d : cronos) v.emplace_back(d.state);
            for(auto &d : fesups) v.emplace_back(d.state);
            for(auto &d : rbdses) v.emplace_back(d.state);
            for(auto &d : rteses) v.emplace_back(d.state);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            return v;
        }
    };

    template<typename ModelType>
    struct SrcDb {
        h5pp::fs::path                                   parent_path;
        std::unordered_map<std::string, h5pp::DsetInfo>  dset;
        std::unordered_map<std::string, h5pp::TableInfo> table;
        std::unordered_map<std::string, h5pp::TableInfo> crono;
        std::unordered_map<std::string, h5pp::TableInfo> fesup;
        std::unordered_map<std::string, h5pp::TableInfo> rbds;
        std::unordered_map<std::string, h5pp::TableInfo> rtes;
        std::unordered_map<std::string, ModelType>       model;
        void                                             clear() {
            dset.clear();
            table.clear();
            crono.clear();
            fesup.clear();
            rbds.clear();
            rtes.clear();
            model.clear();
        }
    };

    struct TgtDb {
        std::unordered_map<std::string, FileId>                    file;
        std::unordered_map<std::string, InfoId<h5pp::DsetInfo>>    dset;
        std::unordered_map<std::string, InfoId<h5pp::TableInfo>>   table;
        std::unordered_map<std::string, InfoId<BufferedTableInfo>> crono;
        std::unordered_map<std::string, InfoId<BufferedTableInfo>> fesup;
        std::unordered_map<std::string, InfoId<BufferedTableInfo>> rbds;
        std::unordered_map<std::string, InfoId<BufferedTableInfo>> rtes;
        std::unordered_map<std::string, InfoId<h5pp::TableInfo>>   model;
        std::unordered_map<std::string, InfoId<h5pp::TableInfo>>   env;
        void                                                       clear() {
            file.clear();
            dset.clear();
            table.clear();
            crono.clear();
            fesup.clear();
            rbds.clear();
            rtes.clear();
            model.clear();
            env.clear();
        }
    };

    std::unordered_map<std::string, FileId> loadFileDatabase(const h5pp::File &h5_tgt);

    template<typename InfoType, typename KeyType>
    std::unordered_map<std::string, InfoId<InfoType>> loadDatabase(const h5pp::File &h5_tgt, const std::vector<KeyType> &keys);

    void saveDatabase(h5pp::File &h5_tgt, const std::unordered_map<std::string, FileId> &fileIdDb);

    template<typename InfoType>
    void saveDatabase(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<InfoType>> &infoDb);

    FileIdStatus getFileIdStatus(const std::unordered_map<std::string, FileId> &fileIdDb, const FileId &newFileId);

}
