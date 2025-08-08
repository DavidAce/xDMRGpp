#include "h5xf.h"
#include "config/enums.h"
#include "debug/exceptions.h"
#include "general/enums.h"
#include "general/prof.h"
#include "general/seq.h"
#include "meld-io/logger.h"
#include "tid/tid.h"
#include <h5pp/h5pp.h>

namespace tools::h5xf {
    namespace internal {
        void copy_dset(h5pp::File &h5_tgt, const h5pp::File &h5_src, h5pp::DsetInfo &tgtInfo, h5pp::DsetInfo &srcInfo, hsize_t index, size_t axis,
                       SlabSelect ssel = SlabSelect::FULL) {
            auto t_scope = tid::tic_scope(__FUNCTION__);
            h5pp::DataInfo dataInfo;
            // The axis parameter must be  < srcInfo.rank+1
            if(axis >= srcInfo.dsetDims->size()) {
                // In this case we are collecting datasets onto a superdataset with dimension +1, enumerating them by that axis
                dataInfo.dataByte = srcInfo.dsetByte;
                dataInfo.dataSize = srcInfo.dsetSize;
                dataInfo.dataRank = tgtInfo.dsetRank; // E.g. if stacking rank 2 matrices then axis == 2 and so data rank must be 3.
                dataInfo.dataDims = std::vector<hsize_t>(axis + 1, 1ull);
                std::copy_n(srcInfo.dsetDims->begin(), srcInfo.dsetDims->size(), dataInfo.dataDims->begin());
                dataInfo.h5Space = h5pp::util::getMemSpace(dataInfo.dataSize.value(), dataInfo.dataDims.value());
                dataInfo.h5Type  = srcInfo.h5Type;
            } else {
                dataInfo.dataByte = srcInfo.dsetByte;
                dataInfo.dataSize = srcInfo.dsetSize;
                dataInfo.dataRank = srcInfo.dsetRank;
                dataInfo.dataDims = srcInfo.dsetDims;
                dataInfo.h5Space  = srcInfo.h5Space;
                dataInfo.h5Type   = srcInfo.h5Type;
            }
            tgtInfo.resizePolicy = h5pp::ResizePolicy::GROW;
            // tgtInfo.dsetSlab                   = h5pp::Hyperslab();
            // tgtInfo.dsetSlab->extent           = dataInfo.dataDims;
            // tgtInfo.dsetSlab->offset           = std::vector<hsize_t>(tgtInfo.dsetDims->size(), 0);
            // tgtInfo.dsetSlab->offset->at(axis) = index;

            // Sometimes there is an extra entry at the end in the source data. We should ignore it.
            // if(tgtInfo.dsetSlab->extent->size() >= 3 and tgtInfo.dsetDims->size() >= 3)
            // tgtInfo.dsetSlab->extent->at(2) = tgtInfo.dsetDims->at(2); // Copy the time dimension to ignore extra entries

            switch(ssel) {
                case SlabSelect::FULL: {
                    // Select the hyperslab to copy from the source
                    // dataInfo.dataSlab                   = tgtInfo.dsetSlab;
                    dataInfo.dataSlab                   = h5pp::Hyperslab();
                    dataInfo.dataSlab->extent           = dataInfo.dataDims;
                    dataInfo.dataSlab->offset           = std::vector<hsize_t>(dataInfo.dataDims->size(), 0);
                    dataInfo.dataSlab->offset->at(axis) = dataInfo.dataDims->at(axis) - 1; // Pick the last slice
                    dataInfo.dataSlab->extent->at(axis) = 1;                               // A single slice

                    tgtInfo.dsetSlab                   = h5pp::Hyperslab();
                    tgtInfo.dsetSlab->extent           = dataInfo.dataSlab->extent;
                    tgtInfo.dsetSlab->offset           = std::vector<hsize_t>(tgtInfo.dsetDims->size(), 0);
                    tgtInfo.dsetSlab->offset->at(axis) = index;

                    break;
                }
                case SlabSelect::MIDCOL: {
                    auto rows = dataInfo.dataDims.value().at(0);
                    auto cols = dataInfo.dataDims.value().at(1);
                    auto mcol = static_cast<decltype(cols)>(cols / 2);
                    // dataInfo.dataSlab                   = tgtInfo.dsetSlab;

                    dataInfo.dataSlab                   = h5pp::Hyperslab();
                    dataInfo.dataSlab->extent           = dataInfo.dataDims;
                    dataInfo.dataSlab->offset           = std::vector<hsize_t>(dataInfo.dataDims->size(), 0);
                    dataInfo.dataSlab->offset->at(axis) = dataInfo.dataDims->at(axis) - 1; // Pick the last slice
                    dataInfo.dataSlab->extent->at(axis) = 1;                               // A single slice

                    dataInfo.dataSlab->offset->at(0) = 0;
                    dataInfo.dataSlab->offset->at(1) = mcol;
                    // dataInfo.dataSlab->offset->at(axis) = 0;
                    dataInfo.dataSlab->extent->at(0) = rows;
                    dataInfo.dataSlab->extent->at(1) = 1;

                    tgtInfo.dsetSlab->offset->at(0)    = 0;
                    tgtInfo.dsetSlab->offset->at(1)    = 0;
                    tgtInfo.dsetSlab->offset->at(axis) = index;
                    tgtInfo.dsetSlab->extent->at(0)    = rows;
                    tgtInfo.dsetSlab->extent->at(1)    = 1;
                    //                    tools::logger::log->info("Copying midcol \ndata dims: {}\ndata slab: {}\ndset dims: {}\ndset slab: {}",
                    //                                             dataInfo.dataDims.value(),
                    //                                             dataInfo.dataSlab->string(),
                    //                                             tgtInfo.dsetDims.value(),
                    //                                             tgtInfo.dsetSlab->string());
                    break;
                }
                    //                case SlabSelect::MIDCOL: {
                    //                    auto rows                           = dataInfo.dataDims.value().at(0);
                    //                    auto cols                           = dataInfo.dataDims.value().at(1);
                    //                    auto mcol                           = static_cast<decltype(cols)>(cols / 2);
                    //                    dataInfo.dataSlab                   = tgtInfo.dsetSlab;
                    //                    dataInfo.dataSlab->offset->at(0)    = 0;
                    //                    dataInfo.dataSlab->offset->at(1)    = mcol;
                    //                    dataInfo.dataSlab->offset->at(axis) = 0;
                    //                    dataInfo.dataSlab->extent->at(0)    = rows;
                    //                    dataInfo.dataSlab->extent->at(1)    = 1;
                    //
                    //                    tgtInfo.dsetSlab->offset->at(0)    = 0;
                    //                    tgtInfo.dsetSlab->offset->at(1)    = mcol;
                    //                    tgtInfo.dsetSlab->offset->at(axis) = index;
                    //                    tgtInfo.dsetSlab->extent->at(0)    = rows;
                    //                    tgtInfo.dsetSlab->extent->at(1)    = 1;
                    //                    break;
                    //                }
            }

            if(tgtInfo.dsetSlab->extent->size() != tgtInfo.dsetSlab->offset->size()) {
                throw except::logic_error("target dsetSlab has mismatching ranks: \n "
                                          "target: offset {}:{} != extent {}:{} | dset {}\n"
                                          "source: dsetDims {}:{} | dset {} \n"
                                          "axis: {}",
                                          tgtInfo.dsetSlab->offset.value(), tgtInfo.dsetSlab->offset->size(), tgtInfo.dsetSlab->extent.value(),
                                          tgtInfo.dsetSlab->extent->size(), tgtInfo.dsetPath.value(), srcInfo.dsetDims.value(), srcInfo.dsetDims->size(),
                                          srcInfo.dsetPath.value(), axis

                );
            }
            if(std::any_of(tgtInfo.dsetSlab->extent->begin(), tgtInfo.dsetSlab->extent->end(), [](auto ext) -> bool { return ext == 0; })) {
                throw except::logic_error("target dsetSlab has zero volume extent: {}", tgtInfo.dsetSlab->extent.value());
            }
            {
                auto t_read = tid::tic_token("readDataset");
                auto data   = h5_src.readDataset<std::vector<std::byte>>(srcInfo); // Read the data into a generic buffer.
                t_read.toc();
                auto t_app = tid::tic_token("appendToDataset");
                // tools::logger::log->info("appending\nsrc: {}\ntgt: {}", dataInfo.string(), tgtInfo.string());
                // if(tgtInfo.dsetDims->back() > 1000) {
                //     tools::logger::log->warn("tgt too large!");
                // }
                h5_tgt.appendToDataset(data, dataInfo, tgtInfo, static_cast<size_t>(axis));
            }
            // Restore previous settings
            tgtInfo.dsetSlab = std::nullopt;
        }

        struct SearchResult {
            std::string                      root;
            std::string                      key;
            long                             hits;
            long                             depth;
            mutable std::vector<std::string> result;
            bool                             operator==(const SearchResult &rhs) const {
                return this->root == rhs.root and this->key == rhs.key and this->hits == rhs.hits and this->depth == rhs.depth;
            }
        };

        struct SearchHasher {
            auto operator()(const SearchResult &r) const { return std::hash<std::string>{}(fmt::format("{}|{}|{}|{}", r.root, r.key, r.hits, r.depth)); }
        };
    }

    void transferEnvironment(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtTableDb, const h5pp::File &h5_src,
                             std::string tgtEnvBase, const FileId &fileId) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        // tools::logger::log->trace("Copying environment to [{}]", tgtEnvBase);
        {
            auto envBuild           = H5T_EnvBuild::EnvBuild();
            envBuild.compiler       = h5_src.readDataset<std::string>(".env/xDMRG++/build/compiler");
            envBuild.compiler_flags = h5_src.readDataset<std::string>(".env/xDMRG++/build/compiler_flags");
            envBuild.linker_flags   = h5_src.readDataset<std::string>(".env/xDMRG++/build/linker_flags");
            envBuild.cpu_type       = h5_src.readDataset<std::string>(".env/xDMRG++/build/cpu_type");
            envBuild.datetime       = h5_src.readDataset<std::string>(".env/xDMRG++/build/datetime");
            envBuild.hostname       = h5_src.readDataset<std::string>(".env/xDMRG++/build/hostname");
            envBuild.os_name        = h5_src.readDataset<std::string>(".env/xDMRG++/build/os_name");
            envBuild.os_platform    = h5_src.readDataset<std::string>(".env/xDMRG++/build/os_platform");
            envBuild.os_release     = h5_src.readDataset<std::string>(".env/xDMRG++/build/os_release");
            envBuild.os_version     = h5_src.readDataset<std::string>(".env/xDMRG++/build/os_version");

            auto envExec     = H5T_EnvExec::EnvExec();
            envExec.cpu_type = h5_src.readDataset<std::string>(".env/xDMRG++/exec/cpu_type");
            envExec.datetime = h5_src.readDataset<std::string>(".env/xDMRG++/exec/datetime");
            envExec.hostname = h5_src.readDataset<std::string>(".env/xDMRG++/exec/hostname");

            auto envGit     = H5T_EnvGit::EnvGit();
            envGit.branch   = h5_src.readDataset<std::string>(".env/xDMRG++/git/branch");
            envGit.commit   = h5_src.readDataset<std::string>(".env/xDMRG++/git/commit");
            envGit.revision = h5_src.readDataset<std::string>(".env/xDMRG++/git/revision");

            auto tgtPathBuild = fmt::format("{}/build", tgtEnvBase);
            auto tgtPathExec  = fmt::format("{}/exec", tgtEnvBase);
            auto tgtPathGit   = fmt::format("{}/git", tgtEnvBase);
            if(tgtTableDb.find(tgtPathBuild) == tgtTableDb.end()) {
                auto t_create = tid::tic_scope("createTable");
                tools::logger::log->debug("Adding target table {}", tgtPathBuild);
                h5pp::TableInfo tableInfo = h5_tgt.getTableInfo(tgtPathBuild);
                if(not tableInfo.tableExists.value()) tableInfo = h5_tgt.createTable(H5T_EnvBuild::get_h5_type(), tgtPathBuild, "Build", std::nullopt, true);
                tgtTableDb[tgtPathBuild] = tableInfo;
            }
            if(tgtTableDb.find(tgtPathExec) == tgtTableDb.end()) {
                auto t_create = tid::tic_scope("createTable");
                tools::logger::log->debug("Adding target table {}", tgtPathExec);
                h5pp::TableInfo tableInfo = h5_tgt.getTableInfo(tgtPathExec);
                if(not tableInfo.tableExists.value()) tableInfo = h5_tgt.createTable(H5T_EnvExec::get_h5_type(), tgtPathExec, "Execution", std::nullopt, true);
                tgtTableDb[tgtPathExec] = tableInfo;
            }
            if(tgtTableDb.find(tgtPathGit) == tgtTableDb.end()) {
                auto t_create = tid::tic_scope("createTable");
                tools::logger::log->debug("Adding target table {}", tgtPathGit);
                h5pp::TableInfo tableInfo = h5_tgt.getTableInfo(tgtPathGit);
                if(not tableInfo.tableExists.value()) tableInfo = h5_tgt.createTable(H5T_EnvGit::get_h5_type(), tgtPathGit, "Git", std::nullopt, true);
                tgtTableDb[tgtPathGit] = tableInfo;
            }
            auto &tgtIdBuild = tgtTableDb[tgtPathBuild];
            auto &tgtIdExec  = tgtTableDb[tgtPathExec];
            auto &tgtIdGit   = tgtTableDb[tgtPathGit];

            // Determine the target index where to copy this record
            hsize_t indexBuild = tgtIdBuild.get_index(fileId.seed); // Positive if it has been read already, numeric_limits::max if it's new
            hsize_t indexExec  = tgtIdExec.get_index(fileId.seed);  // Positive if it has been read already, numeric_limits::max if it's new
            hsize_t indexGit   = tgtIdGit.get_index(fileId.seed);   // Positive if it has been read already, numeric_limits::max if it's new
            auto    t_copy     = tid::tic_scope("copyTableRecords");
            tools::logger::log->trace("Transferring environment tables");
            h5_tgt.writeTableRecords(envBuild, tgtPathBuild, indexBuild, 1);
            h5_tgt.writeTableRecords(envExec, tgtPathExec, indexExec, 1);
            h5_tgt.writeTableRecords(envGit, tgtPathGit, indexGit, 1);
            // Update the database
            tgtIdBuild.insert(fileId.seed, indexBuild);
            tgtIdExec.insert(fileId.seed, indexExec);
            tgtIdGit.insert(fileId.seed, indexGit);
        }
    }

    void transferDatasets(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::DsetInfo>> &tgtDsetDb, const h5pp::File &h5_src,
                          std::unordered_map<std::string, h5pp::DsetInfo> &srcDsetDb, const PathId &pathid, const std::vector<DsetKey> &srcDsetKeys,
                          const FileId &fileId) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        //        auto mem_xferdset = prof::mem_hwm_in_mb();
        for(const auto &srcKey : srcDsetKeys) {
            if(srcDsetDb.find(srcKey.key) == srcDsetDb.end()) throw std::logic_error(h5pp::format("Key [{}] was not found in source map", srcKey.key));
            auto &srcInfo = srcDsetDb[srcKey.key];
            if(not srcInfo.dsetExists or not srcInfo.dsetExists.value()) continue;
            auto tgtName = h5pp::fs::path(srcInfo.dsetPath.value()).filename().string();
            auto tgtPath = std::string{};
            if(srcKey.subgroup.empty())
                tgtPath = h5pp::format("{}/{}", pathid.tgt_path, tgtName);
            else
                tgtPath = h5pp::format("{}/{}/{}", pathid.tgt_path, srcKey.subgroup, tgtName);

            if(tgtDsetDb.find(tgtPath) == tgtDsetDb.end()) {
                auto t_create = tid::tic_scope("createDataset");
                auto tgtDims  = srcInfo.dsetDims.value();
                if(tgtDims.empty()) tgtDims = {0};                            // In case src is a scalar.
                while(tgtDims.size() < srcKey.axis + 1) tgtDims.push_back(1); // Initialize any new dimensions with ones
                tgtDims[srcKey.axis] = 0; // Create with 0 extent in the new axis direction, so that the dataset starts empty (zero volume)
                auto tgtChunk        = tgtDims;
                auto tgtMaxDims      = tgtDims;
                auto chunkSize =
                    std::clamp(5e4 / static_cast<double>(srcInfo.dsetByte.value()), 1., 10000.); // Determine a good chunksize between 1 and 10000 elements
                tgtChunk[srcKey.axis]   = static_cast<hsize_t>(chunkSize);                       // number of elements in the axis that we append into.
                tgtMaxDims[srcKey.axis] = H5S_UNLIMITED;
                if(srcKey.ssel == SlabSelect::MIDCOL) {
                    // When we only copy the middle column, the target dimension should have tgtDims[1] == 1 (i.e. a single column)
                    tgtDims[1]    = 1;
                    tgtChunk[1]   = 1;
                    tgtMaxDims[1] = 1;
                }

                if(srcKey.size == Size::VAR) {
                    auto        srcGroupPath    = h5pp::fs::path(srcInfo.dsetPath.value()).parent_path().string();
                    std::string statusTablePath = fmt::format("{}/status", srcGroupPath);
                    long        bond_max        = h5_src.readTableField<long>(statusTablePath, "bond_max", h5pp::TableSelection::FIRST);
                    tgtDims[0]                  = static_cast<hsize_t>(bond_max);
                }

                tools::logger::log->debug("Adding target dset {} | dims {} | chnk {}", tgtPath, tgtDims, tgtChunk);
                tgtDsetDb[tgtPath] = h5_tgt.createDataset(tgtPath, srcInfo.h5Type.value(), H5D_CHUNKED, tgtDims, tgtChunk, tgtMaxDims);
            }
            auto &tgtId   = tgtDsetDb[tgtPath];
            auto &tgtInfo = tgtId.info;
            // Determine the target index where to copy this record
            hsize_t index = tgtId.get_index(fileId.seed); // Positive if it has been read already
            index         = index != std::numeric_limits<hsize_t>::max() ? index : tgtInfo.dsetDims.value().at(srcKey.axis);
            internal::copy_dset(h5_tgt, h5_src, tgtInfo, srcInfo, index, srcKey.axis, srcKey.ssel);

            // Update the database
            tgtId.insert(fileId.seed, index);
        }
        //        auto mem_xferdset_end = prof::mem_hwm_in_mb();
        //        if(mem_xferdset_end - mem_xferdset > 0) tools::logger::log->info("    xfer dset : mem += {:.1f} MB", mem_xferdset_end - mem_xferdset);
    }

    void transferTables(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtTableDb,
                        std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<TableKey> &srcTableKeys,
                        const FileId &fileId) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        //        auto                      mem_xfertable = prof::mem_hwm_in_mb();
        std::vector<StorageEvent> srcEvents; // This stores the "event" row in the table. We need to make sure to only transfer StorageEvent::ITER_STATE
        for(const auto &srcKey : srcTableKeys) {
            if(srcTableDb.find(srcKey.key) == srcTableDb.end()) throw except::range_error("{}: Key [{}] was not found in source map", __FUNCTION__, srcKey.key);
            auto &srcInfo = srcTableDb[srcKey.key];
            if(not srcInfo.tableExists or not srcInfo.tableExists.value()) continue;
            auto tgtName = h5pp::fs::path(srcInfo.tablePath.value()).filename().string();
            auto tgtPath = pathid.table_path(tgtName);
            if(tgtTableDb.find(tgtPath) == tgtTableDb.end()) {
                auto t_create = tid::tic_scope("createTable");
                tools::logger::log->debug("Adding target table {}", tgtPath);
                h5pp::TableInfo tableInfo = h5_tgt.getTableInfo(tgtPath);
                if(not tableInfo.tableExists.value())
                    tableInfo = h5_tgt.createTable(srcInfo.h5Type.value(), tgtPath, srcInfo.tableTitle.value(), std::nullopt, true);
                tgtTableDb[tgtPath] = tableInfo;
            }
            auto &tgtId   = tgtTableDb[tgtPath];
            auto &tgtInfo = tgtId.info;

            // Determine the target index where to copy this record
            hsize_t index = tgtId.get_index(fileId.seed); // Positive if it has been read already, numeric_limits::max if it's new
            tools::logger::log->trace("Transferring table {} | src idx {} -> tgt idx {}", tgtPath, srcInfo.numRecords.value(), index);
            auto t_copy    = tid::tic_scope("copyTableRecords");
            auto equalType = h5pp::type::H5Tequal_recurse(srcInfo.h5Type.value(), tgtInfo.h5Type.value());
            if(equalType) {
                h5_tgt.copyTableRecords(srcInfo, tgtInfo, h5pp::TableSelection::LAST, index);
            } else {
                const auto &tgtFnames = tgtInfo.fieldNames.value();
                const auto &tgtFsizes = tgtInfo.fieldSizes.value();
                const auto &tgtFofsts = tgtInfo.fieldOffsets.value();
                const auto &srcFnames = srcInfo.fieldNames.value();
                const auto &srcFsizes = srcInfo.fieldSizes.value();
                const auto &srcFofsts = srcInfo.fieldOffsets.value();
                auto        tbs       = h5pp::TableSelection::LAST;
                auto [offset, extent] = h5pp::util::parseTableSelection(tbs, srcInfo.numRecords.value());
                auto srcBuffer        = std::vector(extent * srcInfo.recordBytes.value(), std::byte{});
                auto tgtBuffer        = std::vector(extent * tgtInfo.recordBytes.value(), std::byte{});
                h5pp::hdf5::readTableRecords(srcBuffer, srcInfo, offset, extent);
                for(hsize_t rec = 0; rec < extent; ++rec) {
                    for(hsize_t tfidx = 0; tfidx < tgtFnames.size(); ++tfidx) { // Target field idx
                        // Check if the target field name exists in the source
                        auto &tgtFname = tgtFnames.at(tfidx);
                        auto &tgtFsize = tgtFsizes.at(tfidx);
                        auto &tgtFofst = tgtFofsts.at(tfidx);
                        // Find the iterator to the corresponding fname in source
                        auto srcFiter = std::find(srcFnames.begin(), srcFnames.end(), tgtFname);
                        if(srcFiter == srcFnames.end()) continue;
                        size_t sfidx = static_cast<size_t>(std::distance(srcFnames.begin(), srcFiter)); // Source field idx
                        if(srcFsizes[sfidx] != tgtFsize) continue;
                        // We have found the corresponding field index in source. Now copy that field to target
                        // auto &srcFname = srcFnames.at(sfidx);
                        // auto &srcFsize = srcFsizes.at(sfidx);
                        // auto &srcFofst = srcFofsts.at(sfidx);
                        auto srcBegin = srcBuffer.begin() + rec * srcInfo.recordBytes.value() + srcFofsts[sfidx];
                        auto srcEnd   = srcBegin + srcFsizes.at(sfidx);
                        auto tgtBegin = tgtBuffer.begin() + rec * tgtInfo.recordBytes.value() + tgtFofst;
                        std::copy(srcBegin, srcEnd, tgtBegin);
                    }
                }
                h5pp::hdf5::writeTableRecords(tgtBuffer, tgtInfo, index, 1);
            }
            // Update the database
            tgtId.insert(fileId.seed, index);
        }
        //        auto mem_xfertable_end = prof::mem_hwm_in_mb();
        //        if(mem_xfertable_end - mem_xfertable > 0) tools::logger::log->info("    xfer table: mem += {:.1f} MB", mem_xfertable_end - mem_xfertable);
    }

    template<typename KeyT>
    void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                        std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<KeyT> &srcKeys,
                        const FileId &fileId) {
        // In this function we take series data from each srcTable and create multiple tables tgtTable, one for each
        // index (e.g. iter, bond dim, etc). Each entry in tgtTable corresponds to the same index point on different realizations.
        auto t_scope        = tid::tic_scope(__FUNCTION__);
        auto t_createTable  = tid::ur("createTable");
        auto t_readTableFld = tid::ur("readTableField");
        auto t_readTableRec = tid::ur("readTableRecords");
        auto t_copyTableRec = tid::ur("copyTableRecords");
        auto t_seedIndex    = tid::ur("seed-index");
        auto t_reclaim      = tid::ur("reclaim");

        for(const auto &srcKey : srcKeys) {
            std::vector<size_t>       srcIndices; // Stores the "<srcKey.indexkey>" column in the table (e.g. iter, bond_lim, and so on).
            std::vector<StorageEvent> srcEvents;  // Stores the "event" column in the table, so we only transfer the desired event type.

            if(srcTableDb.find(srcKey.key) == srcTableDb.end()) throw except::logic_error("Key [{}] was not found in source map", srcKey.key);
            auto  t_buffer = tid::ur(fmt::format("bufferRecords-{}", srcKey.classtag));
            auto &srcInfo  = srcTableDb[srcKey.key];
            if(not srcInfo.h5Type) { srcInfo.h5Type = H5Dget_type(srcInfo.h5Dset.value()); }
            auto tgtName = h5pp::fs::path(srcInfo.tablePath.value()).filename().string();
            // Iterate over all table elements. These should be a time series measured at every iteration
            // Note that there could in principle exist duplicate entries, which is why we can't trust the
            // "rec" iterator but have to get the iteration number from the table directly.

            //  Load the entire source table into memory while we distribute it to the different tables in the target file
            t_readTableRec.tic();
            hsize_t record_offset = 0;                          // Start reading from record 0 by default
            hsize_t record_extent = srcInfo.numRecords.value(); // Read all records by default
            if constexpr(sfinae::is_any_v<KeyT, RbdsKey, RtesKey>) {
                // We may have extra sets of entries. We want the last set.
                if(srcKey.expected_size > 0) {
                    record_extent = std::min(srcKey.expected_size, srcInfo.numRecords.value());
                    record_offset = srcInfo.numRecords.value() - record_extent;
                }
            }
            std::vector<std::byte> srcReadBuffer; //  This holds the source table
            h5pp::hdf5::readTableRecords(srcReadBuffer, srcInfo, record_offset, record_extent, h5_tgt.plists);
            t_readTableRec.toc();

            // Read the index and event fields so that we can filter entries
            auto indexfield = text::match(srcKey.indexkey, srcInfo.fieldNames.value());
            auto eventfield = text::match({"event"}, srcInfo.fieldNames.value());
            if(indexfield) {
                t_readTableFld.tic();
                h5pp::hdf5::readTableField(srcIndices, srcInfo, {indexfield.value()});
                tools::logger::log->trace("Retrieved index field {}", indexfield.value());
                t_readTableFld.toc();
            }
            if(eventfield) {
                t_readTableFld.tic();
                h5pp::hdf5::readTableField(srcEvents, srcInfo, {eventfield.value()});
                tools::logger::log->trace("Retrieved event field {}", eventfield.value());
                t_readTableFld.toc();
            }

            for(size_t rec = 0; rec < record_extent; rec++) {
                if(not srcEvents.empty()) {
                    if(not seq::has(srcKey.eventkey, srcEvents.at(rec))) {
                        continue; // Ignore this entry if it is not the desired event type
                    }
                }
                size_t srcIndex = rec + record_offset;                 // Used to generate a path with the correct index, eg. bond_<srcIndex>
                if(not srcIndices.empty()) srcIndex = srcIndices[rec]; // Replace with the desired index value -- this is only used to generate the target path
                auto tgtPath = pathid.create_target_path<KeyT>(tgtName, rec);
                if(tgtTableDb.find(tgtPath) == tgtTableDb.end()) {
                    t_createTable.tic();
                    tools::logger::log->debug("Adding target {} {}", srcKey.classtag, tgtPath);
                    h5pp::TableInfo tableInfo = h5_tgt.getTableInfo(tgtPath);
                    // Disabling compression is supposed to give a nice speedup. Read here:
                    // https://support.hdfgroup.org/HDF5/doc1.8/Advanced/DirectChunkWrite/UsingDirectChunkWrite.pdf
                    if(not tableInfo.tableExists.value()) {
                        auto recordBytes = H5Tget_size(srcInfo.h5Type.value());
                        auto targetBytes = 500 * 1024; // 500 kB
                        auto chunkSize   = static_cast<hsize_t>(std::ceil(targetBytes / static_cast<double>(recordBytes)));
                        //                        tools::logger::log->info("Chunk size: {} | {}", chunkSize, tgtPath);
                        tableInfo = h5_tgt.createTable(srcInfo.h5Type.value(), tgtPath, srcInfo.tableTitle.value(), chunkSize, 6);
                    }
                    tgtTableDb[tgtPath] = tableInfo;
                    tgtTableDb.at(tgtPath).buff.allocateBuffers(srcInfo.numRecords.value());
                    t_createTable.toc();
                }
                auto &tgtId   = tgtTableDb.at(tgtPath);
                auto &tgtInfo = tgtId.info;
                auto &tgtBuff = tgtId.buff;
                // Determine the target index where to copy this record
                // tgtInfo.numRecords is the total number of realizations registered until now
                // tgtId.get_index(fileId.seed) Gets the index number if it already exists, else
                // the size of the table (so we can append)
                hsize_t tgtIndex = tgtId.get_index(fileId.seed);
                tools::logger::log->trace("Transferring {} table {} ({} records) | {} {} @ idx {} -> tgt {}", srcKey.classtag, srcInfo.tablePath.value(),
                                          srcInfo.numRecords.value(), srcKey.indexkey, rec, srcIndex, tgtIndex);

                // read a source record at "srcIndex" into the "tgtIndex" position in the buffer
                auto equalType = h5pp::type::H5Tequal_recurse(srcInfo.h5Type.value(), tgtInfo.h5Type.value());
                t_buffer.tic();
                if(equalType) {
                    tgtBuff.insert(srcReadBuffer.begin() + static_cast<long>(rec * srcInfo.recordBytes.value()),
                                   srcReadBuffer.begin() + static_cast<long>((rec + 1) * srcInfo.recordBytes.value()), tgtIndex);

                } else {
                    const auto &tgtFnames = tgtInfo.fieldNames.value();
                    const auto &tgtFsizes = tgtInfo.fieldSizes.value();
                    const auto &tgtFofsts = tgtInfo.fieldOffsets.value();
                    const auto &srcFnames = srcInfo.fieldNames.value();
                    const auto &srcFsizes = srcInfo.fieldSizes.value();
                    const auto &srcFofsts = srcInfo.fieldOffsets.value();
                    // auto        tbs       = h5pp::TableSelection::LAST;
                    // auto [offset, extent] = h5pp::util::parseTableSelection(tbs, srcInfo.numRecords.value());
                    auto mapBuffer = std::vector(tgtInfo.recordBytes.value(), std::byte{});
                    for(hsize_t tfidx = 0; tfidx < tgtFnames.size(); ++tfidx) { // Target field idx
                        // Check if the target field name exists in the source
                        auto &tgtFname = tgtFnames.at(tfidx);
                        auto &tgtFsize = tgtFsizes.at(tfidx);
                        auto &tgtFofst = tgtFofsts.at(tfidx);
                        // Find the iterator to the corresponding fname in source
                        auto srcFiter = std::find(srcFnames.begin(), srcFnames.end(), tgtFname);
                        if(srcFiter == srcFnames.end()) continue;
                        size_t sfidx = static_cast<size_t>(std::distance(srcFnames.begin(), srcFiter)); // Source field idx
                        if(srcFsizes[sfidx] != tgtFsize) continue;
                        // We have found the corresponding field index in source. Now copy that field to target
                        // auto &srcFname = srcFnames.at(sfidx);
                        // auto &srcFsize = srcFsizes.at(sfidx);
                        // auto &srcFofst = srcFofsts.at(sfidx);
                        auto srcBegin = srcReadBuffer.begin() + rec * srcInfo.recordBytes.value() + srcFofsts[sfidx];
                        auto srcEnd   = srcBegin + srcFsizes.at(sfidx);
                        auto tgtBegin = mapBuffer.begin() + tgtFofst;
                        std::copy(srcBegin, srcEnd, tgtBegin);
                    }
                    tgtBuff.insert(mapBuffer.begin(), mapBuffer.end(), tgtIndex);
                }
                t_buffer.toc();
                if(srcInfo.reclaimInfo.has_value()) {
                    t_reclaim.tic();
                    // One or more table fields are vlen arrays.
                    // This means that HDF5 has allocated memory for those vlen arrays on a pointer somewhere in srcReadBuffer.
                    // This memory needs to be de-allocated to avoid a memory leak, which unfortunately means that we have to flush
                    // the buffer immediately before srcReadBuffer is overwritten in the next round.
                    // When calling tgtBuff.insert we copy the pointer to the data, but not the data itself.
                    // If we then overwrite srcReadBuffer, we lose track of the vlen memory pointed to previously.
                    tgtBuff.flush();
                    srcInfo.reclaim();
                    t_reclaim.toc();
                }
                // Update the database
                t_seedIndex.tic();
                tgtId.insert(fileId.seed, tgtIndex);
                t_seedIndex.toc();
            }
            tid::get(fmt::format("bufferRecords-{}", srcKey.classtag)) += t_buffer;
        }
        //        auto mem_xfercrono_end = prof::mem_hwm_in_mb();
        //        if(mem_xfercrono_end - mem_xfercrono > 0) tools::logger::log->info("    xfer crono: mem += {:.2f} MB", mem_xfercrono_end - mem_xfercrono);
        tid::get("createTable") += t_createTable;
        tid::get("readTableField") += t_readTableFld;
        tid::get("readTableRecords") += t_readTableRec;
        tid::get("copyTableRecords") += t_copyTableRec;
        tid::get("seed-index") += t_seedIndex;
        tid::get("reclaim") += t_reclaim;
    }
    template void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                                 std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<CronoKey> &srcKeys,
                                 const FileId &fileId);
    template void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                                 std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<FesUpKey> &srcKeys,
                                 const FileId &fileId);
    template void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                                 std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<RbdsKey> &srcKeys,
                                 const FileId &fileId);
    template void transferSeries(h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<BufferedTableInfo>> &tgtTableDb,
                                 std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid, const std::vector<RtesKey> &srcKeys,
                                 const FileId &fileId);
}
