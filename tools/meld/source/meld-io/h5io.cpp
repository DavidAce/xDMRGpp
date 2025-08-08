#include "h5io.h"
#include "debug/exceptions.h"
#include "general/prof.h"
#include "general/text.h"
#include "h5xf.h"
#include "meld-io/h5dbg.h"
#include "meld-io/id.h"
#include "meld-io/logger.h"
#include "meld-io/meta.h"
#include "meld-io/parse.h"
#include "meld-io/type.h"
#include "mpi/mpi-tools.h"
#include "qm/lbit.h"
#include "tid/tid.h"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <h5pp/h5pp.h>
#include <h5tb.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tools::h5io {

    namespace internal {
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

    void saveFailedJob(const h5pp::File &h5_src, const std::string &msg, const std::exception &ex) {
        auto failseed = -1l;
        auto failconf = std::string("invalid_config_file");
        auto failmore = std::string();
        try {
            failseed = tools::parse::extract_digits_from_h5_filename<long>(h5_src.getFileName());
        } catch(const std::exception &exseed) { failmore += fmt::format(" | {}", exseed.what()); }
        try {
            failconf = h5_src.readDataset<std::string>("common/config_filename");
        } catch(const std::exception &exconf) {
            failmore += fmt::format(" | {}", exconf.what());
            failconf += fmt::format(" | {}", h5_src.getFilePath());
        }
        auto pathjobs = fmt::format("{}", h5pp::fs::path(tools::h5io::h5_tgt_part_path).replace_extension(".job").string());
        auto patherrs = fmt::format("{}", h5pp::fs::path(tools::h5io::h5_tgt_part_path).replace_extension(".err").string());

        auto failskip = false;
        auto linebuff = std::string();
        auto filejobs = std::fstream(pathjobs, std::ios_base::app | std::ios_base::out);
        auto fileerrs = std::fstream(patherrs, std::ios_base::app | std::ios_base::out);
        auto failjobs = fmt::format("{} {}", failconf, failseed);
        while(std::getline(filejobs, linebuff)) {
            if(linebuff.find(failjobs) != std::string::npos) {
                tools::logger::log->info("Found previous fail job line: {}", failjobs);
                failskip = true;
                break;
            }
        }
        if(not failskip) {
            filejobs.clear();
            filejobs.seekg(0, std::ios::beg);
            filejobs << failjobs << '\n';
            fileerrs << failjobs << fmt::format("| {:<24}", msg) << ex.what() << failmore << '\n';
        }
    }

    std::string get_tmp_dirname(std::string_view exename) { return fmt::format("{}.{}", h5pp::fs::path(exename).filename(), getenv("USER")); }

    size_t get_num_decimals(double val) {
        auto s       = fmt::format("{}", val);
        auto pos_dot = s.find('.');
        if(pos_dot == std::string::npos) return 0;
        return s.substr(pos_dot + 1).size();
    }
    size_t get_max_decimals(std::initializer_list<double> l) {
        if(l.size() == 0) throw std::runtime_error("l is empty");
        std::vector<size_t> decs(l.size());
        std::transform(l.begin(), l.end(), decs.begin(), get_num_decimals);
        return *std::max_element(decs.begin(), decs.end());
    }

    template<typename H, typename C>
    std::string get_standardized_base(const ModelId<H, C> &M) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        if constexpr(std::is_same_v<H, sdual>) {
            size_t decimals = 4;
            return h5pp::format("L{1}/l{2:.{0}f}/d{3:+.{0}f}", decimals, M.model_size, M.h.lambda, M.h.delta);
        }
        if constexpr(std::is_same_v<H, majorana>) {
            size_t decimals = 4;
            return h5pp::format("L{1}/g{2:.{0}f}/d{3:+.{0}f}", decimals, M.model_size, M.h.g, M.h.delta);
        }
        if constexpr(std::is_same_v<H, lbit>) {
            size_t J_dec = get_max_decimals({M.h.J1_mean, M.h.J1_wdth, M.h.J2_mean, M.h.J2_wdth, M.h.J3_mean, M.h.J3_wdth});
            size_t x_dec = get_max_decimals({M.h.xi_Jcls});
            size_t u_dec = get_max_decimals({M.c.u_fmix, M.c.u_lambda});
            auto   L_str = fmt::format("L{}", M.model_size);
            auto   J_str = fmt::format("J[{1:+.{0}f}±{2:.{0}f}_{3:+.{0}f}±{4:.{0}f}_{5:+.{0}f}±{6:.{0}f}]", J_dec, M.h.J1_mean, M.h.J1_wdth, M.h.J2_mean,
                                       M.h.J2_wdth, M.h.J3_mean, M.h.J3_wdth);
            auto   x_str = fmt::format("x{1:.{0}f}", x_dec, M.h.xi_Jcls);
            // J2_span is special since it can be -1ul, meaning long range. We prefer putting L in the path rather than 18446744073709551615
            auto r_str      = M.h.J2_span == -1ul ? fmt::format("rL") : fmt::format("r{}", M.h.J2_span);
            auto u_du_str   = fmt::format("d[{}]", M.c.u_depth);
            auto u_f_str    = fmt::format("_f[{1:.{0}f}]", u_dec, M.c.u_fmix);
            auto u_l_str    = fmt::format("_l[{1:.{0}f}]", u_dec, M.c.u_lambda);
            auto u_wk_str   = fmt::format("_w[{0:.2}]", enum2sv(M.c.u_wkind));
            auto u_mk_str   = fmt::format("_m[{0:.2}]", enum2sv(M.c.u_mkind).substr(7));
            auto u_bond_str = fmt::format("_bond[{}]", M.c.u_bond);
            auto u_str      = fmt::format("u[{}{}{}{}{}{}]", u_du_str, u_f_str, u_l_str, u_wk_str, u_mk_str, u_bond_str);
            auto base       = fmt::format("{0}/{1}/{2}/{3}/{4}", L_str, J_str, x_str, r_str, u_str);
            tools::logger::log->info("creating base: {}", base);
            return base;
        }
    }
    template std::string get_standardized_base(const ModelId<sdual, nocircuit> &M);
    template std::string get_standardized_base(const ModelId<majorana, nocircuit> &M);
    template std::string get_standardized_base(const ModelId<lbit, lbit_circuit> &M);

    std::vector<std::string> findKeys(const h5pp::File &h5_src, const std::string &root, const std::vector<std::string> &expectedKeys, long hits, long depth,
                                      bool usecache) {
        auto                                                                      t_scope = tid::tic_scope(__FUNCTION__);
        static std::unordered_set<internal::SearchResult, internal::SearchHasher> cache;
        std::vector<std::string>                                                  result;
        for(const auto &key : expectedKeys) {
            internal::SearchResult searchQuery = {root, key, hits, depth, {}};
            std::string            cacheMsg;
            auto                   cachedIt = cache.find(searchQuery);
            bool                   cacheHit = cachedIt != cache.end() and                         // The search query has been processed earlier
                            ((hits > 0 and static_cast<long>(cachedIt->result.size()) >= hits) or // Asked for a up to "hits" reslts
                             (hits <= 0 and static_cast<long>(cachedIt->result.size()) >= 1)      // Asked for any number of results
                            );
            if(cacheHit and usecache) {
                for(auto &item : cachedIt->result) {
                    if(result.size() > 1 and std::find(result.begin(), result.end(), item) != result.end()) continue;
                    result.emplace_back(item);
                }
                cacheMsg = " | cache hit";
            } else {
                std::vector<std::string> found;
                auto                     key_split = text::split(key, "/");
                if(key_split.size() > 1) {
                    // We try to find the last key given the current root at infinite depth. Then we keep
                    // groups if all the key_split elements are present in the path
                    auto key_last = key_split.back();
                    found         = h5_src.findGroups(key_last, root, -1, -1);
                    // Prune found items if they do not have the expected path depth
                    text::erase_if(found, [&key_split](auto &item) -> bool {
                        auto rel_depth = std::count(item.begin(), item.end(), '/');
                        return rel_depth + 1 != static_cast<long>(key_split.size());
                    });

                    // Prune found items if they do not have the same groups as key_split
                    text::erase_if(found, [&key_split](auto &item) -> bool {
                        // Check that all parts of key_split exist in item
                        for(const auto &s : key_split) {
                            auto fuzz_pos    = s.find_first_of('*', 0);
                            auto key_in_item = item.find(s.substr(0, fuzz_pos)) != std::string::npos;
                            if(not key_in_item) return true;
                        }
                        return false;
                    });

                } else {
                    if(key.empty())
                        found.emplace_back(key);
                    else if(key.back() == '*') {
                        std::string_view key_match = std::string_view(key).substr(0, key.size() - 2);
                        for(auto &item : h5_src.findGroups(key_match, root, hits, depth)) {
                            if(not item.starts_with(key_match)) continue;
                            if(found.size() > 1 and std::find(found.begin(), found.end(), item) != found.end()) continue;
                            found.emplace_back(item);
                        }
                    } else {
                        for(auto &item : h5_src.findGroups(key, root, hits, depth)) {
                            tools::logger::log->info("Found item [{}]", item);
                            if(not item.ends_with(key)) continue;
                            if(found.size() > 1 and std::find(found.begin(), found.end(), item) != found.end()) continue;
                            found.emplace_back(item);
                        }
                    }
                }
                auto [it, ok] = cache.insert(searchQuery);
                it->result    = found;
                // Now we have built a cache hit. Add it to results
                for(auto &item : it->result) {
                    if(result.size() > 1 and std::find(result.begin(), result.end(), item) != result.end()) continue;
                    result.emplace_back(item);
                }
            }
            // Sort
            std::sort(result.begin(), result.end(), text::natcomp);
            tools::logger::log->trace("Search: root {} | key [{}] | result {}{}", root, key, result, cacheMsg);
        }
        return result;
    }

    template<typename H, typename C>
    std::vector<ModelKey> loadModel(const h5pp::File &h5_src, std::unordered_map<std::string, ModelId<H, C>> &srcModelDb,
                                    const std::vector<ModelKey> &srcKeys) {
        auto                  t_scope = tid::tic_scope(__FUNCTION__);
        std::vector<ModelKey> keys;
        for(const auto &srcKey : srcKeys) {
            auto hamiltonian_path = fmt::format("{}/{}/{}", srcKey.algo, srcKey.model, srcKey.name);
            auto key              = fmt::format("{}|{}", h5pp::fs::path(h5_src.getFilePath()).parent_path(), hamiltonian_path);
            if(srcModelDb.find(key) == srcModelDb.end() and h5_src.linkExists(hamiltonian_path)) {
                // Copy the model from the attributes in h5_src to a struct ModelId
                srcModelDb[key]   = ModelId<H, C>();
                auto &srcModelId  = srcModelDb[key];
                auto &hamiltonian = srcModelId.h;
                if constexpr(std::is_same_v<H, sdual>) {
                    auto h5tb_hamiltonian    = h5_src.readTableRecords<h5tb_ising_selfdual::table>(hamiltonian_path, h5pp::TableSelection::FIRST);
                    hamiltonian.J_mean       = h5tb_hamiltonian.J_mean;
                    hamiltonian.J_wdth       = h5tb_hamiltonian.J_wdth;
                    hamiltonian.h_mean       = h5tb_hamiltonian.h_mean;
                    hamiltonian.h_wdth       = h5tb_hamiltonian.h_wdth;
                    hamiltonian.lambda       = h5tb_hamiltonian.lambda;
                    hamiltonian.delta        = h5tb_hamiltonian.delta;
                    hamiltonian.distribution = static_cast<std::string_view>(h5tb_hamiltonian.distribution);
                }
                if constexpr(std::is_same_v<H, majorana>) {
                    auto h5tb_hamiltonian    = h5_src.readTableRecords<h5tb_ising_majorana::table>(hamiltonian_path, h5pp::TableSelection::FIRST);
                    hamiltonian.g            = h5tb_hamiltonian.g;
                    hamiltonian.delta        = h5tb_hamiltonian.delta;
                    hamiltonian.distribution = static_cast<std::string_view>(h5tb_hamiltonian.distribution);
                }
                if constexpr(std::is_same_v<H, lbit>) {
                    auto  h5tb_hamiltonian = h5_src.readTableRecords<h5tb_lbit::table>(hamiltonian_path, h5pp::TableSelection::FIRST);
                    auto  circuit_path     = fmt::format("{}/{}/unitary_circuit", srcKey.algo, srcKey.model);
                    auto  h5tb_circuit     = h5_src.readTableRecords<qm::lbit::UnitaryGateParameters>(circuit_path, h5pp::TableSelection::LAST);
                    auto &circuit          = srcModelId.c;

                    hamiltonian.J1_mean      = h5tb_hamiltonian.J1_mean;
                    hamiltonian.J2_mean      = h5tb_hamiltonian.J2_mean;
                    hamiltonian.J3_mean      = h5tb_hamiltonian.J3_mean;
                    hamiltonian.J1_wdth      = h5tb_hamiltonian.J1_wdth;
                    hamiltonian.J2_wdth      = h5tb_hamiltonian.J2_wdth;
                    hamiltonian.J3_wdth      = h5tb_hamiltonian.J3_wdth;
                    hamiltonian.xi_Jcls      = h5tb_hamiltonian.xi_Jcls;
                    hamiltonian.J2_span      = h5tb_hamiltonian.J2_span;
                    hamiltonian.distribution = static_cast<std::string_view>(h5tb_hamiltonian.distribution);

                    auto path_circuit = fmt::format("{}/{}/unitary_circuit", srcKey.algo, srcKey.model);
                    circuit.u_depth   = h5tb_circuit.layer + 1; // h5_src.readAttribute<size_t>(path_circuit, "u_depth");
                    circuit.u_fmix    = h5tb_circuit.f;         // h5_src.readAttribute<double>(path_circuit, "u_fmix");
                    circuit.u_lambda  = h5tb_circuit.l;         // h5_src.readAttribute<std::optional<double>>(path_circuit, "u_lambda");
                    circuit.u_wkind   = h5tb_circuit.wkind;     // circuit.u_wkind = h5_src.readAttribute<LbitCircuitGateWeightKind>(path_circuit, "u_wkind");
                    circuit.u_mkind   = h5tb_circuit.mkind;     // circuit.u_mkind = h5_src.readAttribute<LbitCircuitGateMatrixKind>(path_circuit, "u_mkind");

                    auto path_lbits = fmt::format("{}/{}/lbits", srcKey.algo, srcKey.model);
                    if(h5_src.linkExists(path_lbits)) {
                        auto u_bond = h5_src.readAttribute<std::optional<long>>(path_lbits, "u_bond");
                        if(not u_bond.has_value()) u_bond = text::extract_value_between<long>(key, "_bond", "]");
                        if(not u_bond.has_value()) throw except::logic_error("Failed to get u_bond value from string: {}", key);
                        circuit.u_bond = u_bond.value();
                    }
                }
                srcModelId.model_size = h5_src.readAttribute<size_t>(hamiltonian_path, "model_size");
                srcModelId.model_type = h5_src.readAttribute<std::string>(hamiltonian_path, "model_name");
                srcModelId.algorithm  = srcKey.algo;
                srcModelId.key        = key;
                srcModelId.path       = hamiltonian_path;
                srcModelId.basepath   = get_standardized_base(srcModelId);
                tools::logger::log->debug("srcModelId.basepath = {} | key {}", srcModelId.basepath, key);
            }
            keys.emplace_back(srcKey);
            keys.back().key = key;
        }
        return keys;
    }
    template std::vector<ModelKey> loadModel(const h5pp::File &h5_src, std::unordered_map<std::string, ModelId<sdual, nocircuit>> &srcModelDb,
                                             const std::vector<ModelKey> &srcKeys);
    template std::vector<ModelKey> loadModel(const h5pp::File &h5_src, std::unordered_map<std::string, ModelId<majorana, nocircuit>> &srcModelDb,
                                             const std::vector<ModelKey> &srcKeys);
    template std::vector<ModelKey> loadModel(const h5pp::File &h5_src, std::unordered_map<std::string, ModelId<lbit, lbit_circuit>> &srcModelDb,
                                             const std::vector<ModelKey> &srcKeys);

    template<typename H, typename C>
    void saveModel([[maybe_unused]] const h5pp::File &h5_src, h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtModelDb,
                   const ModelId<H, C> &modelId, const FileId &fileId) {
        auto              t_scope      = tid::tic_scope(__FUNCTION__);
        const std::string tgtModelPath = fmt::format("{}/{}", modelId.basepath, modelId.path);

        if(tgtModelDb.find(tgtModelPath) == tgtModelDb.end()) {
            tools::logger::log->debug("Copying model to tgtPath {}", tgtModelPath);
            tools::logger::log->trace("modelId key    {}", modelId.key);
            tools::logger::log->trace("modelId path   {}", modelId.path);
            tools::logger::log->trace("modelId bpath  {}", modelId.basepath);
            tools::logger::log->trace("modelId fields {}", modelId.h.fields);
            tgtModelDb[tgtModelPath] = h5_tgt.getTableInfo(tgtModelPath);

            auto &tgtId   = tgtModelDb[tgtModelPath];
            auto &tgtInfo = tgtModelDb[tgtModelPath].info;

            // It doesn't make sense to copy the whole hamiltonian table here:
            // It is specific to a single realization, but here we collect the fields common to all realizations.
            const auto &srcModelInfo    = h5_src.getTableInfo(modelId.path); // This will return a TableInfo to a an existing table in the source file
            auto        modelPath       = fmt::format("{}/{}/model", modelId.basepath, modelId.algorithm); // Generate a new path for the target group
            auto        hamiltonianPath = fmt::format("{}/hamiltonian", modelPath);                        // Generate a new path for the target table

            // Update an entry of the hamiltonian table with the relevant fields
            tools::logger::log->trace("Copying model {}", hamiltonianPath);
            tgtInfo = h5_tgt.createTable(modelId.h.get_h5_type(), hamiltonianPath, h5pp::format("{} Hamiltonian", modelId.algorithm));
            h5_tgt.writeTableRecords(modelId.h, hamiltonianPath);

            if constexpr(std::is_same_v<H, lbit>) {
                auto circuitPath = fmt::format("{}/unitary_circuit", modelPath); // Generate a new path for the target table for the circuit
                tools::logger::log->trace("Copying the unitary circuit {}", circuitPath);

                // Copy the table fields that exist in the src circuit
                h5_tgt.createTable(modelId.c.get_h5_type(), circuitPath, h5pp::format("Unitary Circuit", modelId.algorithm));
                h5_tgt.writeTableRecords(modelId.c, circuitPath);
            }

            // Update the database
            tgtId.insert(fileId.seed, 0);

            // Now write the same data as scalar attributes, which can be helpful sometimes
            h5_tgt.writeAttribute(modelId.model_size, hamiltonianPath, "model_size");
            h5_tgt.writeAttribute(modelId.model_type, hamiltonianPath, "model_type");
        }
    }
    template void saveModel(const h5pp::File &h5_src, h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtModelDb,
                            const ModelId<sdual, nocircuit> &modelId, const FileId &fileId);
    template void saveModel(const h5pp::File &h5_src, h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtModelDb,
                            const ModelId<majorana, nocircuit> &modelId, const FileId &fileId);
    template void saveModel(const h5pp::File &h5_src, h5pp::File &h5_tgt, std::unordered_map<std::string, InfoId<h5pp::TableInfo>> &tgtModelDb,
                            const ModelId<lbit, lbit_circuit> &modelId, const FileId &fileId);

    std::vector<DsetKey> gatherDsetKeys(const h5pp::File &h5_src, std::unordered_map<std::string, h5pp::DsetInfo> &srcDsetDb, const PathId &pathid,
                                        const std::vector<DsetKey> &srcKeys) {
        auto                 t_scope = tid::tic_scope(__FUNCTION__);
        std::vector<DsetKey> keys;

        std::string   srcParentPath = h5pp::fs::path(h5_src.getFilePath()).parent_path();
        h5pp::Options options;
        for(auto &srcKey : srcKeys) {
            // Make sure to only collect keys for objects that we have asked for.
            if(not pathid.match(srcKey.algo, srcKey.state)) {
                tools::logger::log->debug("Failed to match pathid {} to algo: [{}] and state: [{}]", pathid.src_path, srcKey.algo, srcKey.state);
                continue;
            }

            // A "srcKey" is a struct describing a dataset that we want, with name, type, size, etc
            // Here we look for a matching dataset in the given group path, and generate a key for it.
            // The srcKeys that match an existing dataset are returned in "keys", with an additional parameter ".key" which
            // is an unique identifier to find the DsetInfo object in the map srcDsetDb
            // The database srcDsetDb  is used to keep track the DsetInfo structs for found datasets.
            // auto path = fmt::format("{}/{}", pathid.src_path, srcKey.name);
            auto path = srcKey.create_source_path(pathid);
            auto key  = fmt::format("{}|{}", srcParentPath, path);
            if(srcDsetDb.find(key) == srcDsetDb.end()) {
                srcDsetDb[key] = h5_src.getDatasetInfo(path);
                if(srcDsetDb[key].dsetExists.value())
                    tools::logger::log->debug("Detected new source dataset {}", key);
                else
                    tools::logger::log->trace("Missing source dataset {}", key);
            } else {
                auto t_read = tid::tic_scope("readDsetInfo");

                auto &srcInfo = srcDsetDb[key];
                // We reuse the struct srcDsetDb[dsetKey] for every source file,
                // but each time have to renew the following fields
                srcInfo.h5File     = h5_src.openFileHandle();
                srcInfo.h5Dset     = std::nullopt;
                srcInfo.h5Space    = std::nullopt;
                srcInfo.dsetExists = std::nullopt;
                srcInfo.dsetSize   = std::nullopt;
                srcInfo.dsetDims   = std::nullopt;
                srcInfo.dsetByte   = std::nullopt;
                srcInfo.dsetPath   = path;
                h5pp::scan::readDsetInfo(srcInfo, srcInfo.h5File.value(), options, h5_src.plists);
            }
            auto &srcInfo = srcDsetDb[key];
            if(srcInfo.dsetExists and srcInfo.dsetExists.value()) {
                keys.emplace_back(srcKey);
                keys.back().key = key;
            } else {
                tools::logger::log->debug("Missing dataset [{}] in file [{}]", path, h5_src.getFilePath());
            }
        }
        return keys;
    }

    template<StrictTableSize strictTableSize, typename KeyT>
    std::vector<KeyT> gatherTableKeys(const h5pp::File &h5_src, std::unordered_map<std::string, h5pp::TableInfo> &srcTableDb, const PathId &pathid,
                                      const std::vector<KeyT> &srcKeys) {
        auto              t_scope = tid::tic_scope(__FUNCTION__);
        std::vector<KeyT> keys;
        h5pp::Options     options;
        std::string       srcParentPath = h5pp::fs::path(h5_src.getFilePath()).parent_path();
        for(auto &srcKey : srcKeys) {
            // Make sure to only collect keys for objects that we have asked for.
            if(not pathid.match(srcKey.algo, srcKey.state)) continue;

            // A "srcKey" is a struct describing a table that we want, with name, type, size, etc
            // Here we look for a matching table in the given group path, and generate a key for it.
            // The srcKeys that match an existing table are returned in "keys", with an additional parameter ".key" which
            // is an unique identifier to find the TableInfo object in the map srcTableDb
            // The database srcTableDb  is used to keep track the TableInfo structs for found datasets.
            auto path = srcKey.create_source_path(pathid);
            // auto path = fmt::format("{}/{}", pathid.src_path, srcKey.name);
            // auto                   path = fmt::format("{}/{}", pathid.src_path, srcKey.name);
            auto key = fmt::format("{}|{}", srcParentPath, path);
            if(srcTableDb.find(key) == srcTableDb.end()) {
                srcTableDb[key] = h5_src.getTableInfo(path);
                if(srcTableDb[key].tableExists.value()) tools::logger::log->debug("Detected new source {}: {}", srcKey.classtag, key);
            } else {
                auto t_read = tid::tic_scope("readTableInfo");

                auto &srcInfo = srcTableDb[key];
                // We reuse the struct srcDsetDb[key] for every source file,
                // but each time have to renew the following fields

                auto expectedType = srcInfo.h5Type;

                srcInfo.h5File      = h5_src.openFileHandle();
                srcInfo.h5Dset      = std::nullopt;
                srcInfo.h5Type      = std::nullopt;
                srcInfo.numRecords  = std::nullopt;
                srcInfo.tableExists = std::nullopt;
                srcInfo.tablePath   = path;
                h5pp::scan::readTableInfo(srcInfo, srcInfo.h5File.value(), options, h5_src.plists);

                if(srcInfo.tableExists.has_value() and srcInfo.tableExists.value() and srcInfo.h5Type.has_value()) {
                    if(expectedType.has_value() and !h5pp::type::H5Tequal_recurse(srcInfo.h5Type.value(), expectedType.value())) {
                        tools::logger::log->warn("Table type mismatch!");
                        // We have a mismatch! the type of the table may have changed. We should therefore read the table info from scratch
                        srcInfo = h5_src.getTableInfo(path); // Clear
                    }
                }
            }

            auto &srcInfo = srcTableDb[key];
            if(srcInfo.tableExists and srcInfo.tableExists.value()) {
                // Check that the tables are the same size before and after
                if constexpr(strictTableSize == StrictTableSize::TRUE) {
                    if constexpr(sfinae::is_any_v<KeyT, RbdsKey, RtesKey>) {
                        if(srcKey.expected_size == -1ul) {
                            // We should take the last N entries in these tables.
                            // Here we first take the iter attribute on the table, and then count table records with that iter value.
                            // The matching ones should be the last.
                            auto iter_attr       = h5_src.readAttribute<size_t>(srcInfo.tablePath.value(), "iter");
                            auto iters           = h5_src.readTableField<std::vector<size_t>>(srcInfo.tablePath.value(), {"iter"}, h5pp::TableSelection::ALL);
                            auto count           = std::count_if(iters.begin(), iters.end(), [&iter_attr](auto iter) -> bool { return iter == iter_attr; });
                            srcKey.expected_size = safe_cast<size_t>(count);
                        }
                    } else {
                        if(srcKey.expected_size == -1ul) { srcKey.expected_size = srcInfo.numRecords.value(); }
                        if(srcInfo.numRecords.value() != srcKey.expected_size) {
                            if(srcInfo.numRecords.value() < srcKey.expected_size)
                                throw except::range_error("Table size mismatch:\n"
                                                          "file: {}\n"
                                                          "dset: {}\n"
                                                          "records {} | expected {}",
                                                          h5_src.getFilePath(), srcInfo.tablePath.value(), srcInfo.numRecords.value(), srcKey.expected_size);
                        }
                    }
                }

                keys.emplace_back(srcKey);
                keys.back().key = key;
            } else {
                srcInfo.h5File = std::nullopt;
                tools::logger::log->debug("Missing {} [{}] in file [{}]", srcKey.classtag, path, h5_src.getFilePath());
            }
        }
        if constexpr(strictTableSize == StrictTableSize::TRUE) {
            if(keys.size() > 1) {
                // Check that all tables have the same number of records in them
                std::vector<std::pair<std::string, hsize_t>> tableRecords;
                tableRecords.reserve(keys.size());
                for(const auto &table : keys) {
                    auto &srcInfo = srcTableDb.at(table.key);
                    tableRecords.emplace_back(std::make_pair(srcInfo.tablePath.value(), srcInfo.numRecords.value()));
                }
                //            tools::logger::log->info("<{}> : numRecords: {}", sfinae::type_name<KeyT>(), numRecords);
                if(std::any_of(tableRecords.begin(), tableRecords.end(), [&tableRecords](auto rec) -> bool { return rec.second != tableRecords[0].second; }))
                    tools::logger::log->warn("unequal number of records in {} tables:\n{}\n{}\n", keys.front().classtag, h5_src.getFilePath(),
                                             fmt::join(tableRecords, "\n"));
                // throw except::runtime_error("unequal number of records in {} tables:\n{}\n", keys.front().classtag, fmt::join(tableRecords, "\n"));
            }
        }
        return keys;
    }

    template<typename ModelIdType>
    void merge(h5pp::File &h5_tgt, const h5pp::File &h5_src, const FileId &fileId, const tools::h5db::Keys &keys, tools::h5db::TgtDb &tgtdb) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        // Define reusable source Info
        static tools::h5db::SrcDb<ModelIdType> srcdb;
        h5pp::fs::path                         parent_path = h5pp::fs::path(h5_src.getFilePath()).parent_path();
        if(srcdb.parent_path != parent_path) {
            // Clear when moving to another set of seeds (new point on the phase diagram)
            srcdb.clear();
            srcdb.parent_path = parent_path;
        }

        // Start finding the required components in the source
        auto groups = tools::h5io::findKeys(h5_src, "/", keys.get_algos(), -1, 0);
        for(const auto &algo : groups) {
            // Extract the model
            auto modelKeys = tools::h5io::loadModel(h5_src, srcdb.model, keys.models);
            if(modelKeys.size() != 1) throw std::runtime_error("Exactly 1 model has to be loaded into keys");
            auto &modelId = srcdb.model[modelKeys.back().key];
            // Save the model to file if it hasn't
            tools::h5io::saveModel(h5_src, h5_tgt, tgtdb.model, modelId, fileId);
            auto tgt_base     = modelId.basepath;
            auto tgt_env_base = fmt::format("{}/{}/.env", modelId.basepath, modelId.algorithm);

            try {
                auto t_xfer = tid::tic_scope("xfer");
                tools::h5xf::transferEnvironment(h5_tgt, tgtdb.env, h5_src, tgt_env_base, fileId);
            } catch(const std::runtime_error &ex) {
                tools::logger::log->warn("Transfer failed in [{}]: {}", tgt_env_base, ex.what());
                saveFailedJob(h5_src, "transfer failed", ex);
            }

            // Next search for tables and datasets in the source file
            // and transfer them to the target file
            auto state_groups = tools::h5io::findKeys(h5_src, algo, keys.get_states(), -1, 0);
            for(const auto &state : state_groups) {
                auto pathid = PathId(tgt_base, algo, state);
                // Try gathering all the tables
                auto dsetKeys  = std::vector<DsetKey>();
                auto tableKeys = std::vector<TableKey>();
                auto fesupKeys = std::vector<FesUpKey>();
                auto rbdsKeys  = std::vector<RbdsKey>();
                auto rtesKeys  = std::vector<RtesKey>();
                auto cronoKeys = std::vector<CronoKey>();

                //                auto mem_gather = prof::mem_hwm_in_mb();

                try {
                    auto t_gather = tid::tic_scope("gather");
                    dsetKeys      = tools::h5io::gatherDsetKeys(h5_src, srcdb.dset, pathid, keys.dsets);
                    tableKeys     = tools::h5io::gatherTableKeys<StrictTableSize::FALSE>(h5_src, srcdb.table, pathid, keys.tables);
                    fesupKeys     = tools::h5io::gatherTableKeys<StrictTableSize::FALSE>(h5_src, srcdb.fesup, pathid, keys.fesups);
                    rbdsKeys      = tools::h5io::gatherTableKeys<StrictTableSize::TRUE>(h5_src, srcdb.rbds, pathid, keys.rbdses);
                    rtesKeys      = tools::h5io::gatherTableKeys<StrictTableSize::TRUE>(h5_src, srcdb.rtes, pathid, keys.rteses);
                    cronoKeys     = tools::h5io::gatherTableKeys<StrictTableSize::TRUE>(h5_src, srcdb.crono, pathid, keys.cronos);
                } catch(const std::runtime_error &ex) {
                    tools::logger::log->error("key gather failed in [{}|{}]: {}", h5_src.getFilePath(), pathid.src_path, ex.what());
                    saveFailedJob(h5_src, "key gathering failed", ex);
                    continue; // File is broken. Do not transfer.
                }
                //                auto mem_gather_end = prof::mem_hwm_in_mb();
                //                if(mem_gather_end > mem_gather) tools::logger::log->info("gather  : mem += {:.1f} MB", mem_gather_end - mem_gather);
                //                auto mem_xfer = prof::mem_hwm_in_mb();

                try {
                    auto t_xfer = tid::tic_scope("xfer");
                    tools::h5xf::transferDatasets(h5_tgt, tgtdb.dset, h5_src, srcdb.dset, pathid, dsetKeys, fileId);
                    tools::h5xf::transferTables(h5_tgt, tgtdb.table, srcdb.table, pathid, tableKeys, fileId);
                    tools::h5xf::transferSeries(h5_tgt, tgtdb.fesup, srcdb.fesup, pathid, fesupKeys, fileId);
                    tools::h5xf::transferSeries(h5_tgt, tgtdb.rbds, srcdb.rbds, pathid, rbdsKeys, fileId);
                    tools::h5xf::transferSeries(h5_tgt, tgtdb.rtes, srcdb.rtes, pathid, rtesKeys, fileId);
                    tools::h5xf::transferSeries(h5_tgt, tgtdb.crono, srcdb.crono, pathid, cronoKeys, fileId);

                } catch(const std::runtime_error &ex) {
                    tools::logger::log->warn("Transfer failed in [{}]: {}", pathid.src_path, ex.what());
                    saveFailedJob(h5_src, "transfer failed", ex);
                }
                //                h5_tgt.flush();
                //                auto mem_xfer_end = prof::mem_hwm_in_mb();
                //                if(mem_xfer_end > mem_xfer) tools::logger::log->info("transfer: mem += {:.1f} MB", mem_xfer_end - mem_xfer);

                //                try {
                //                    auto t_dsets  = tid::tic_scope("dsets");
                //                    auto dsetKeys = tools::h5io::gatherDsetKeys(h5_src, srcdb.dset, pathid, keys.dsets);
                //                    tools::h5xf::transferDatasets(h5_tgt, tgtdb.dset, h5_src, srcdb.dset, pathid, dsetKeys, fileId);
                //                } catch(const std::runtime_error &ex) { tools::logger::log->warn("Dset transfer failed in [{}]: {}", pathid.src_path,
                //                ex.what()); }
                //
                //                try {
                //                    auto t_table   = tid::tic_scope("table");
                //                    auto tableKeys = tools::h5io::gatherTableKeys<StrictTableSize::FALSE>(h5_src, srcdb.table, pathid, keys.tables);
                //                    tools::h5xf::transferTables(h5_tgt, tgtdb.table, srcdb.table, pathid, tableKeys, fileId);
                //                } catch(const std::runtime_error &ex) { tools::logger::log->error("Table transfer failed in [{}]: {}", pathid.src_path,
                //                ex.what()); } try {
                //                    auto t_fesup   = tid::tic_scope("fesup");
                //                    auto fesupKeys = tools::h5io::gatherTableKeys<StrictTableSize::FALSE>(h5_src, srcdb.fesup, pathid, keys.fesups);
                //                    tools::h5xf::transferSeries(h5_tgt, tgtdb.fesup, srcdb.fesup, pathid, fesupKeys, fileId);
                //                } catch(const std::runtime_error &ex) { tools::logger::log->error("Fesup transfer failed in [{}]: {}", pathid.src_path,
                //                ex.what()); } try {
                //                    auto t_fesdn   = tid::tic_scope("fesdn");
                //                    auto fesdnKeys = tools::h5io::gatherTableKeys<StrictTableSize::TRUE>(h5_src, srcdb.fesdn, pathid, keys.fesdns);
                //                    tools::h5xf::transferSeries(h5_tgt, tgtdb.fesdn, srcdb.fesdn, pathid, fesdnKeys, fileId);
                //                } catch(const std::runtime_error &ex) { tools::logger::log->error("Fesdn transfer failed in [{}]: {}", pathid.src_path,
                //                ex.what()); } try {
                //                    auto t_crono   = tid::tic_scope("crono");
                //                    auto cronoKeys = tools::h5io::gatherTableKeys<StrictTableSize::TRUE>(h5_src, srcdb.crono, pathid, keys.cronos);
                //                    tools::h5xf::transferSeries(h5_tgt, tgtdb.crono, srcdb.crono, pathid, cronoKeys, fileId);
                //                } catch(const std::runtime_error &ex) { tools::logger::log->error("Crono transfer failed in[{}]: {}", pathid.src_path,
                //                ex.what()); }
            }
        }

        auto t_close = tid::tic_scope("close");

        // Check that there are no errors hiding in the HDF5 error-stack
        auto num_errors = H5Eget_num(H5E_DEFAULT);
        if(num_errors > 0) {
            H5Eprint(H5E_DEFAULT, stderr);
            throw std::runtime_error(fmt::format("Error when treating file [{}]", h5_src.getFilePath()));
        }
    }
    template void merge<ModelId<sdual, nocircuit>>(h5pp::File &h5_tgt, const h5pp::File &h5_src, const FileId &fileId, const tools::h5db::Keys &keys,
                                                   tools::h5db::TgtDb &tgtdb);
    template void merge<ModelId<majorana, nocircuit>>(h5pp::File &h5_tgt, const h5pp::File &h5_src, const FileId &fileId, const tools::h5db::Keys &keys,
                                                      tools::h5db::TgtDb &tgtdb);
    template void merge<ModelId<lbit, lbit_circuit>>(h5pp::File &h5_tgt, const h5pp::File &h5_src, const FileId &fileId, const tools::h5db::Keys &keys,
                                                     tools::h5db::TgtDb &tgtdb);

    void writeProfiling(h5pp::File &h5_tgt) {
        auto t_scope = tid::tic_scope(__FUNCTION__);
        H5T_profiling::register_table_type();
        for(const auto &t : tid::get_tree("", tid::level::normal)) {
            auto tablepath = h5pp::format(".db/prof_{}/{}", mpi::world.id, t->get_label());
            if(not h5_tgt.linkExists(tablepath)) h5_tgt.createTable(H5T_profiling::h5_type, tablepath, "H5MBL Profiling", {100});
            H5T_profiling::item entry{t->get_time(), t->get_time_avg(), t->get_tic_count()};
            h5_tgt.writeTableRecords(entry, tablepath, 0);
        }
    }
}