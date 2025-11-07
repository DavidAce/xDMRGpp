#pragma once
#include "enums.h"
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace tid {

#if defined(TID_DISABLE)
    inline constexpr bool enable = false;
#else
    inline constexpr bool enable = true;
#endif

    class ur;
    using ur_umap_t = std::unordered_map<std::string, std::shared_ptr<tid::ur>>;
    namespace internal {
        class ur_node_t;

        struct string_hash {
            using is_transparent = void;
            [[nodiscard]] size_t operator()(const char *txt) const { return std::hash<std::string_view>{}(txt); }
            [[nodiscard]] size_t operator()(std::string_view txt) const { return std::hash<std::string_view>{}(txt); }
            [[nodiscard]] size_t operator()(const std::string &txt) const { return std::hash<std::string>{}(txt); }
        };
    }

    /*! \brief RAII-style token for tid::ur
     *
     * Calls tic() on construct, toc() on destruct
     */
    class token {
        private:
        ur         &t;
        std::string temp_prefix;

        public:
        explicit token(ur &t_, double add_time = 0);
        token(ur &t_, std::string_view prefix_, double add_time = 0);
        ~token() noexcept;
        void tic() noexcept;
        void toc() noexcept;
        ur  &ref() noexcept;
        ur  *operator->() noexcept;
    };

    /*! \brief The object that measures time
     *
     * Trivia: 'ur' means 'clock' in Swedish
     */
    class ur {
        private:
        using hresclock = std::chrono::high_resolution_clock;
        hresclock::time_point tic_timepoint;
        hresclock::time_point toc_timepoint;
        hresclock::time_point lap_timepoint;
        hresclock::time_point start_timepoint;
        hresclock::duration   delta_time    = hresclock::duration::zero();
        hresclock::duration   measured_time = hresclock::duration::zero();
        hresclock::duration   lap_time      = hresclock::duration::zero();
        size_t                count         = 0;
        bool                  is_measuring  = false;
        std::string           label;
        level                 lvl = level::normal;

        public:
        ur() = default;
        explicit ur(std::string_view label, level l = level::normal) noexcept; // Constructor
        void tic() noexcept;
        void toc() noexcept;

        void reset();
        void set_label(std::string_view label) noexcept;
        void set_time(double other_time_in_seconds) noexcept;
        void add_time(double other_time_in_seconds) noexcept;
        void set_count(size_t count) noexcept;
        void add_count(size_t count) noexcept;
        void set_level(level l) noexcept;
        void start_lap() noexcept;

        [[nodiscard]] std::string get_label() const noexcept;
        [[nodiscard]] double      get_age() const;
        [[nodiscard]] double      get_time() const;
        [[nodiscard]] double      get_time_avg() const;
        [[nodiscard]] size_t      get_tic_count() const;
        [[nodiscard]] double      get_last_interval() const;
        [[nodiscard]] double      restart_lap();
        [[nodiscard]] double      get_lap() const;
        [[nodiscard]] level       get_level() const;

        ur &operator=(double other_time_in_seconds) noexcept;
        ur &operator+=(double other_time_in_seconds) noexcept;
        ur &operator-=(double other_time_in_seconds) noexcept;
        ur &operator+=(const ur &rhs) noexcept;
        ur &operator-=(const ur &rhs) noexcept;

        [[nodiscard]] token tic_token(double add_time = 0) noexcept; /*!< Gives a token RAII-style tic-toc */
        [[nodiscard]] token tic_token(
            std::string_view prefix,
            double add_time = 0) noexcept; /*!< Gives a token RAII-style tic-toc, and temporarily sets a new global prefix for subsequent tic/tokens */
        friend class token;
        friend class internal::ur_node_t;

        using ur_umap_t = std::unordered_map<std::string, std::shared_ptr<tid::ur>, internal::string_hash, std::equal_to<>>;
        ur_umap_t ur_under;                                // For making a tree of ur-objects
        ur       &operator[](std::string_view label);      // For adding leafs to the tree
        ur       &insert(std::string_view label, level l); // For adding leafs to the tree
    };

    [[nodiscard]] ur   &get(std::string_view key, level l = level::parent, bool unscoped = false);
    [[nodiscard]] ur   &get_unscoped(std::string_view key, level l = level::parent);
    [[nodiscard]] token tic_token(std::string_view key, level l = level::parent, double add_time = 0);
    [[nodiscard]] token tic_scope(std::string_view key, level l = level::parent, double add_time = 0);

    extern void add(std::string_view key, std::string_view label = "");
    extern void tic(std::string_view key, level l = level::parent);
    extern void toc(std::string_view key, level l = level::parent);
    extern void reset(const std::vector<std::string> &excl = {});
    extern void reset(std::string_view expr);
    extern void set_prefix(std::string_view);

    namespace internal {
        struct ur_ref_t {
            std::string                           key;
            std::reference_wrapper<const tid::ur> ref;
            double                                sum               = 0.0; /*!< The sum of all the ur-times under this */
            double                                frac              = 1.0; /*!< The fraction of this/parent */
            size_t                                tree_max_key_size = 0;
            [[nodiscard]] std::string             str() const;
            const ur                             *operator->() const;
        };
        inline tid::level current_level = tid::normal;
        inline tid::ur    dummy("disabled", level::disabled);

        using tid_db_unordered_map_t = std::unordered_map<std::string, ur, string_hash, std::equal_to<>>;
        inline tid_db_unordered_map_t tid_db;

#if defined(_OPENMP)
        inline std::vector<std::string> ur_prefix(static_cast<unsigned long>(omp_get_max_threads())); // Each thread has its own prefix
#else
        inline std::vector<std::string> ur_prefix(1); // Only one thread
#endif
        extern const std::string &ur_prefix_get();
        extern void               ur_prefix_set(std::string_view key);
        extern void               ur_prefix_push_back(std::string_view key);
        extern void               ur_prefix_pop_back(std::string_view key);
        template<typename T = std::vector<std::string_view>>
        T split(std::string_view strv, std::string_view delims);
    }
    [[nodiscard]] std::vector<internal::ur_ref_t> get_tree(const tid::ur &u, std::string_view prefix = "", level l = level::normal);
    [[nodiscard]] std::vector<internal::ur_ref_t> get_tree(std::string_view prefix = "", level l = level::normal);
    [[nodiscard]] std::vector<internal::ur_ref_t> search(const tid::ur &u, std::string_view match);
    [[nodiscard]] std::vector<internal::ur_ref_t> search(std::string_view match);
    void                                          set_level(level l);
    //    void                                                 merge_thread_enries();
    void print_tree(const tid::ur &u, std::string_view prefix = "", level l = level::normal);
    void print_tree(std::string_view prefix = "", level l = level::normal);
}