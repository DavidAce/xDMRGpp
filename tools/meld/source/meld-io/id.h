#pragma once
#include "config/enums.h"
#include "general/text.h"
#include <deque>
#include <h5pp/details/h5ppFormat.h>
#include <h5pp/details/h5ppHid.h>
#include <h5pp/details/h5ppInfo.h>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct BufferedTableInfo {
    private:
    struct ContiguousBuffer {
        hsize_t                offset = 0; // In table units
        hsize_t                extent = 0; // In table units
        std::vector<std::byte> rawdata;
    };
    hsize_t count = 0;

    public:
    h5pp::TableInfo              *info = nullptr;
    std::vector<ContiguousBuffer> recordBuffer;
    size_t                        maxRecords = 100; // Will hold maxRecords seeds for each crono iter. We would like this to be max 1 GB worth of data
    BufferedTableInfo();
    BufferedTableInfo(h5pp::TableInfo *info_);
    BufferedTableInfo &operator=(h5pp::TableInfo *info_);
    ~BufferedTableInfo();

    void    insert(const std::vector<std::byte> &entry, hsize_t index /* index in units of table entries */);
    void    insert(const std::vector<std::byte>::const_iterator begin, const std::vector<std::byte>::const_iterator end, hsize_t index);
    void    allocateBuffers(size_t expectedIters);
    hsize_t get_count(); /* Number of inserts */
    size_t  getNumRecordsInBuffer();
    void    flush();
};

struct FileStats {
    size_t               files = 0;
    size_t               count = 0;
    uintmax_t            bytes = 0;
    double               elaps = 0.0;
    [[nodiscard]] double get_speed() const {
        if(elaps == 0)
            return 0.;
        else
            return static_cast<double>(count) / elaps;
    }
};

struct FileId {
    long   seed      = -1;
    char   path[512] = {};
    size_t hash      = {};
    FileId()         = default;
    FileId(long seed_, std::string_view path_, size_t hash_);
    [[nodiscard]] std::string string() const;
};

struct lbit_circuit {
    size_t                      u_depth;
    double                      u_fmix;
    double                      u_tstd;
    double                      u_lambda;
    LbitCircuitGateWeightKind   u_wkind;
    LbitCircuitGateMatrixKind   u_mkind;
    long                        u_bond = -1;
    static constexpr auto       fields = std::array<std::string_view, 7>{"u_fmix", "u_depth", "u_lambda", "u_wkind", "u_mkind", "u_bond"};
    static const h5pp::hid::h5t get_h5_type();
};

struct nocircuit {};

struct lbit {
    double                J1_mean;
    double                J2_mean;
    double                J3_mean;
    double                J1_wdth;
    double                J2_wdth;
    double                J3_wdth;
    size_t                J2_span;
    double                xi_Jcls;
    h5pp::vstr_t          distribution;
    static constexpr auto fields = std::array<std::string_view, 8>{"J1_mean", "J2_mean", "J3_mean", "J1_wdth", "J2_wdth", "J3_wdth", "J2_span", "xi_Jcls"};
    static const h5pp::hid::h5t get_h5_type();
};

struct sdual {
    double                      J_mean;
    double                      J_wdth;
    double                      h_mean;
    double                      h_wdth;
    double                      lambda;
    double                      delta;
    h5pp::vstr_t                distribution;
    static constexpr auto       fields = std::array<std::string_view, 6>{"J_mean", "J_wdth", "h_mean", "h_wdth", "lambda", "delta"};
    static const h5pp::hid::h5t get_h5_type();
};

struct majorana {
    double                      g;
    double                      delta;
    h5pp::vstr_t                distribution;
    static constexpr auto       fields = std::array<std::string_view, 6>{"g", "delta"};
    static const h5pp::hid::h5t get_h5_type();
};

template<typename Hamiltonian, typename Circuit>
struct ModelId {
    Hamiltonian h;
    Circuit     c;
    size_t      model_size;
    std::string model_type;
    std::string algorithm;
    std::string key;
    std::string path;
    std::string basepath;
};

struct PathId {
    public:
    std::string src_path, tgt_path;
    std::string base, algo, state;
    PathId(std::string_view base_, std::string_view algo_, std::string_view state_);
    [[nodiscard]] bool        match(std::string_view algo_pattern, std::string_view state_pattern) const;
    [[nodiscard]] std::string dset_path(std::string_view dsetname) const;
    [[nodiscard]] std::string table_path(std::string_view tablename) const;
    template<typename KeyT>
    [[nodiscard]] std::string create_target_path(std::string_view tablename, size_t index) const {
        /*
         * When collecting a "KeyT" kind of table with tablename = <name>:
         *      - the source paths are at <base>/<algo>/<state>/<classtag>/indexpfx_index/<name>
         *      - we want all entries of event type KeyT::eventkey in each <name>
         *      - we collect the contribution from each realization to each <KeyT::indexpfx><index> separately
         *      - the target path <base>/<algo>/<state>/<KeyT::classtag>/<KeyT::indexpfx><index>/<name>, collects all the contributions
         *      - We pad <index> with zeros on the left, for alphanumeric sorting
         */
        return h5pp::format("{}/{}/{}/{}/{}{:0>4}/{}", base, algo, state, KeyT::classtag, KeyT::indexpfx, index, tablename);
    }
};

template<typename InfoType>
struct InfoId {
    private:
    bool                              modified = false;
    std::unordered_map<long, hsize_t> db;

    public:
    InfoType info = InfoType();
    InfoId()      = default;
    InfoId(long seed_, hsize_t index_);
    InfoId(const InfoType &info_);
    bool    db_modified() const { return modified; }
    bool    has_index(long seed) const { return db.find(seed) != db.end(); }
    hsize_t get_index(long seed) const {
        auto res = db.find(seed);
        if(res != db.end())
            return res->second;
        else
            return db.size();
    }
    void insert(long seed, hsize_t index) {
        auto res = db.insert({seed, index});
        if(res.second) modified = true;
    }
    [[nodiscard]] const std::unordered_map<long, hsize_t> &get_db() const { return db; }
};

template<>
struct InfoId<BufferedTableInfo> {
    private:
    bool modified = false;
    //    std::unordered_map<long, hsize_t> db;
    std::vector<std::pair<long, hsize_t>> db;

    public:
    h5pp::TableInfo   info = h5pp::TableInfo();
    BufferedTableInfo buff = BufferedTableInfo();
    InfoId()               = default;
    InfoId(long seed_, hsize_t index_);
    InfoId(const h5pp::TableInfo &info_);
    InfoId &operator=(const h5pp::TableInfo &info_);

    void allocateBuffers(size_t expectedIters) {
        buff.allocateBuffers(expectedIters);
        db.reserve(expectedIters);
    }
    bool db_modified() const { return modified; }
    bool has_index(long seed) const {
        //        return db.contains(seed);
        auto rit = std::find_if(std::rbegin(db), std::rend(db), [&seed](const auto &p) -> bool { return p.first == seed; });
        return rit != std::rend(db);
        //        return db.find(seed) != db.end();
    }
    hsize_t get_index(long seed) const {
        //        auto res = db.find(seed);

        auto res = std::find_if(std::rbegin(db), std::rend(db), [&seed](const auto &p) -> bool { return p.first == seed; });
        if(res != db.rend())
            return res->second;
        else
            return db.size();
    }
    void insert(long seed, hsize_t index) {
        if(!has_index(seed)) {
            db.emplace_back(std::pair<long, hsize_t>{seed, index});
            modified = true;
        }

        //        auto res = db.insert({seed, index});
        //        if(res.second) modified = true;
    }

    [[nodiscard]] const auto &get_db() const { return db; }
};

struct SeedId {
    long    seed  = -1;
    hsize_t index = std::numeric_limits<hsize_t>::max();
    SeedId()      = default;
    SeedId(long seed_, hsize_t index_) : seed(seed_), index(index_) {}
    [[nodiscard]] std::string string() const { return h5pp::format("seed {} | index {}", index, seed); }
};

class H5T_FileId {
    public:
    static inline h5pp::hid::h5t h5_type;
    H5T_FileId();
    static void register_table_type();
};

class H5T_SeedId {
    public:
    static inline h5pp::hid::h5t h5_type;
    H5T_SeedId();
    static void register_table_type();
};

class H5T_profiling {
    public:
    static inline h5pp::hid::h5t h5_type;

    struct item {
        double time;
        double avg;
        size_t count;
    };

    H5T_profiling();
    static void register_table_type();
};

class H5T_EnvBuild {
    static inline h5pp::hid::h5t h5_type;
    public:

    struct EnvBuild {
        h5pp::fstr_t<16>  compiler;
        h5pp::fstr_t<512> compiler_flags;
        h5pp::fstr_t<512> linker_flags;
        h5pp::fstr_t<256> cpu_type;
        h5pp::fstr_t<32>  datetime;
        h5pp::fstr_t<64>  hostname;
        h5pp::fstr_t<64>  os_name;
        h5pp::fstr_t<64>  os_platform;
        h5pp::fstr_t<64>  os_release;
        h5pp::fstr_t<64>  os_version;
    };
    H5T_EnvBuild();
    static void register_table_type();
    static const h5pp::hid::h5t get_h5_type();
};

class H5T_EnvExec {
    static inline h5pp::hid::h5t h5_type;
    public:

    struct EnvExec {
        h5pp::fstr_t<256> cpu_type;
        h5pp::fstr_t<32>  datetime;
        h5pp::fstr_t<64>  hostname;
    };
    H5T_EnvExec();
    static void register_table_type();
    static const h5pp::hid::h5t get_h5_type();

};

class H5T_EnvGit {
    static inline h5pp::hid::h5t h5_type;
    public:

    struct EnvGit {
        h5pp::fstr_t<64> branch;
        h5pp::fstr_t<64> commit;
        h5pp::fstr_t<64> revision;
    };
    H5T_EnvGit();
    static void register_table_type();
    static const h5pp::hid::h5t get_h5_type();

};
