#include "tid.h"
#include <algorithm>
#include <deque>
#include <fmt/format.h>
#include <string>
#include <string_view>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace tid {
    namespace internal {

        const ur *ur_ref_t::operator->() const { return &ref.get(); }

        [[nodiscard]] std::string ur_ref_t::str() const {
            return fmt::format("{0:<{1}} {2:>8.3f} s | sum {3:>8.3f} s | {4:>6.2f} % | avg {5:>8.2e} s | level {6} | count {7}", key, tree_max_key_size,
                               ref.get().get_time(), sum, 100 * frac, ref.get().get_time_avg(), level2sv(ref.get().get_level()), ref.get().get_tic_count());
        }

        template<typename T>
        T split(std::string_view strv, std::string_view delims) {
            T output;
            //            output.reserve(strv.length() / 4);
            auto first = strv.begin();
            while(first != strv.end()) {
                const auto second = std::find_first_of(first, std::cend(strv), std::cbegin(delims), std::cend(delims));
                const auto pos    = static_cast<std::string_view::size_type>(std::distance(strv.begin(), first));
                const auto len    = static_cast<std::string_view::size_type>(std::distance(first, second));
                if(first != second) output.emplace_back(strv.substr(pos, len));
                if(second == strv.end()) break;
                first = std::next(second);
            }
            return output;
        }
        template std::vector<std::string_view> split(std::string_view strv, std::string_view delims);
        template std::deque<std::string_view>  split(std::string_view strv, std::string_view delims);

        const std::string &ur_prefix_get() {
#if defined(_OPENMP)
            size_t thread_number = static_cast<size_t>(omp_get_thread_num());
#else
            size_t thread_number = 1;
#endif
            return ur_prefix[thread_number];
        }
        void ur_prefix_set(std::string_view key) {
#if defined(_OPENMP)
            size_t thread_number = static_cast<size_t>(omp_get_thread_num());
#else
            size_t thread_number = 1;
#endif
            ur_prefix[thread_number] = key;
#if defined(_OPENMP)
            if(not omp_in_parallel()) {
                // Keep all the other ur_prefixes up to date as well
                for(auto &other_prefix : ur_prefix) other_prefix = key;
            }
#endif
        }
        void ur_prefix_push_back(std::string_view key) {
            if(key.empty()) return;

#if defined(_OPENMP)
            size_t thread_number = static_cast<size_t>(omp_get_thread_num());
#else
            size_t thread_number = 1;
#endif
            auto &prefix    = ur_prefix[thread_number];
            auto  key_nodot = key.substr(key.find_first_not_of('.'));
            if(prefix.empty()) {
                prefix = key_nodot;
            } else {
                prefix = fmt::format("{}.{}", prefix, key_nodot);
            }
#if defined(_OPENMP)
            if(not omp_in_parallel()) {
                // Keep all the other ur_prefixes up to date as well
                for(auto &other_prefix : ur_prefix) other_prefix = prefix;
            }
#endif
        }
        void ur_prefix_pop_back(std::string_view key) {
            if(key.empty()) return;
#if defined(_OPENMP)
            size_t thread_number = static_cast<size_t>(omp_get_thread_num());
#else
            size_t thread_number = 1;
#endif
            auto &prefix    = ur_prefix[thread_number];
            auto  key_nodot = key.substr(key.find_first_not_of('.'));
            auto  pos       = prefix.rfind(key_nodot);
            if(pos != std::string::npos) { prefix.erase(pos, key_nodot.size()); }
            while(not prefix.empty() and prefix.back() == '.') prefix.pop_back();
#if defined(_OPENMP)
            if(not omp_in_parallel()) {
                // Keep all the other ur_prefixes up to date as well
                for(auto &other_prefix : ur_prefix) other_prefix = prefix;
            }
#endif
        }

    }

    ur &get(std::deque<std::string_view> &keys, level l, ur &u) {
        if(keys.empty()) return u;
        auto &u_sub = u.insert(keys[0], l);
        keys.pop_front();
        return get(keys, l, u_sub);
    }
    ur &get(std::string_view key, level l, bool unscoped) {
        std::string prefix_key;
        const auto &current_prefix = internal::ur_prefix_get();
        if(unscoped) {
            prefix_key = key;
        } else {                         // Append to get prefix.key
            if(current_prefix.empty()) { // No earlier prefix so just use key as prefix
                prefix_key = key;
            } else {
                prefix_key = fmt::format("{}.{}", current_prefix, key);
            }
        }

        // If the element does not exist we insert it here
        // We insert missing elements recursively starting from the front of the prefix
        // Example: prefix == this.is.a.tid.timer
        //          Then we would try to find "this", "is", "a" ... in that order

        if(prefix_key.empty()) throw std::runtime_error(fmt::format("Invalid key: {}", prefix_key));
        auto sp = tid::internal::split<std::deque<std::string_view>>(prefix_key, ".");
        // Finding and inserting the element must be done atomically in threaded contexts!
        //        auto it       = tid::internal::tid_db.end();
        bool emplaced = false;
        auto it       = tid::internal::tid_db.find(sp[0]);
        if(it == tid::internal::tid_db.end()) {
#pragma omp critical(tidget)
            {
                std::tie(it, emplaced) = tid::internal::tid_db.emplace(std::make_pair(std::string(sp[0]), tid::ur(sp[0])));
                if(emplaced and l != level::parent) it->second.set_level(l);
            }
        }
        sp.pop_front();
        return get(sp, l, it->second);
    }

    ur &get_unscoped(std::string_view key, level l) { return get(key, l, true); }

    token tic_token(std::string_view key, level l, double add_time) {
        if(internal::current_level < l) return internal::dummy.tic_token();
#if defined(_OPENMP)
        if(omp_in_parallel()) {
            // We have to determine whether the threads have forked already
            if(internal::ur_prefix_get().find('@') != std::string::npos) {
                // We found a fork! No need to add @ again
                return tid::get(key, l).tic_token(add_time);
            }
            // No thread fork found!
            // Append the thread number to avoid data races on ur objects
            auto thread_key = fmt::format("{}@{}", key, omp_get_thread_num());
            return tid::get(thread_key, l).tic_token(add_time);
        }
#endif
        return tid::get(key, l).tic_token(add_time);
    }

    token tic_scope(std::string_view key, level l, double add_time) {
        if(internal::current_level < l) return internal::dummy.tic_token();
#if defined(_OPENMP)
        if(omp_in_parallel()) {
            // We have to determine whether the threads have forked already
            if(internal::ur_prefix_get().find('@') != std::string::npos) {
                // We found a fork! No need to add @ again
                return tid::get(key, l).tic_token(key, add_time);
            }
            // No thread fork found!
            // Append the thread number to avoid data races on ur objects
            auto thread_key = fmt::format("{}@{}", key, omp_get_thread_num());
            return tid::get(thread_key, l).tic_token(thread_key, add_time);
        }
#endif
        return tid::get(key, l).tic_token(key, add_time);
    }

    void tic(std::string_view key, level l) { get(key, l).tic(); }

    void toc(std::string_view key, level l) { get(key, l).toc(); }

    void reset(std::string_view expr) {
        for(auto &[key, ur] : tid::internal::tid_db) {
            if(key.find(expr) != std::string_view::npos) ur.reset();
        }
    }

    void reset(const std::vector<std::string> &excl) {
        for(auto &[key, ur] : tid::internal::tid_db) {
            for(const auto &e : excl) {
                if(key == e) continue;
            }
            ur.reset();
        }
    }

    //    void set_prefix(std::string_view prefix) { internal::ur_prefix = std::string(prefix); }

    std::vector<internal::ur_ref_t> get_tree(const tid::ur &u, std::string_view prefix, level l) {
        std::string key;
        if(prefix.empty())
            key = u.get_label();
        else
            key = fmt::format("{}.{}", prefix, u.get_label());

        auto tree = std::vector<internal::ur_ref_t>{{.key = key, .ref = u, .sum = 0.0, .frac = 1.0}};
        for(const auto &un : u.ur_under) {
            tree.front().sum += un.second->get_time(); // Add times under
            for(const auto &t : get_tree(*un.second, key, l)) {
                //                if(un.second->get_time() == 0) {
                //                    // If the intermediate node did not measure time, add the times under it instead
                //                    tree.front().sum += t.sum;
                //                }
                tree.emplace_back(t);
            }
        }

        // Sort the tree
        if(tree.size() > 1)
            std::sort(std::next(tree.begin()), tree.end(), [](const internal::ur_ref_t &t1, const internal::ur_ref_t &t2) -> bool { return t1.key < t2.key; });

        // Prune the tree based on level
        tree.erase(std::remove_if(tree.begin(), tree.end(), [&l](auto &t) -> bool { return t->get_level() > l; }), tree.end());

        // Find the longest key in the tree
        auto   max_it       = std::max_element(tree.begin(), tree.end(), [](const auto &a, const auto &b) -> bool { return a.key.size() < b.key.size(); });
        size_t max_key_size = 0;
        if(max_it != tree.end()) max_key_size = max_it->key.size();

        // Calculate the fractions and set the maximum key size
        for(auto &t : tree) {
            t.tree_max_key_size = max_key_size;
            if(tree.front()->get_time() == 0) break;
            if(&t == &tree.front()) continue;
            auto t_parent = tree.front()->get_time() == 0.0 ? tree.front().sum : tree.front()->get_time();
            t.frac        = t->get_time() / t_parent;
        }

        return tree;
    }

    std::vector<internal::ur_ref_t> get_tree(std::string_view prefix, level l) {
        //        merge_thread_enries();
        std::vector<internal::ur_ref_t> tree;
        for(const auto &[key, u] : tid::internal::tid_db) {
            // fmt::print("tid_db: {}\n", key);
            if(key == prefix or prefix.empty()) {
                auto t = get_tree(u, "", l);
                tree.insert(tree.end(), t.begin(), t.end());
            }
        }
        // Find the longest key in the tree
        auto max_it = std::max_element(tree.begin(), tree.end(), [](const auto &a, const auto &b) -> bool { return a.key.size() < b.key.size(); });
        if(max_it != tree.end()) {
            // Set the max key size
            for(auto &t : tree) t.tree_max_key_size = max_it->key.size();
        }

        return tree;
    }

    //    void merge_thread_entries() {
    //        // Merge threaded items containing "@"
    //        // We can essentially just add them all up in the last one and disable the rest
    //        for(auto &[key, u] : tid::internal::tid_db) {
    //            auto atpos = key.find('@');
    //            if(atpos != std::string::npos) {
    //                // We have found a threaded item such as "fLBIT.run.step@1.gen_swap_gates"
    //                auto ntpos          = key.substr(atpos).find_first_of('.');
    //                auto label          = fmt::format("{}@*{}", key.substr(0, atpos), key.substr(atpos + ntpos));
    //                auto [it, emplaced] = tid::internal::tid_db.try_emplace(label, tid::ur(label, u.get_level()));
    //                auto &ur_merge      = it->second;
    //                auto  add_count     = u.get_tic_count() / static_cast<size_t>(omp_get_max_threads());
    //                ur_merge.set_count(add_count);
    //                ur_merge += u.get_time() / omp_get_max_threads();
    //                u.set_level(level::disabled);
    //            }
    //        }

    //        auto thread_tree = std::vector<internal::ur_ref_t>();
    //        for(auto &t : tree) {
    //            auto atpos = t.key.find('@');
    //            if(atpos != std::string::npos) {
    //                // We have found a threaded item such as "fLBIT.run.step@1.gen_swap_gates"
    //                // We make a new leaf with the symbol @* instead of @<thread_number>
    //                auto ntpos = t.key.substr(atpos).find_first_of('.');
    //                auto tname = fmt::format("{}@*{}", t.key.substr(0, atpos), t.key.substr(atpos + ntpos));
    //                if(std::find(thread_tree.begin(), thread_tree.end(), [](const internal::ur_ref_t &u) -> bool { return u.key == tname; }) ==
    //                thread_tree.end()) {
    //                    thread_tree.emplace_back(internal::ur_ref_t{.key=tname, });
    //                }
    //            }
    //        }
    //    }

    std::vector<internal::ur_ref_t> search(const tid::ur &u, std::string_view match) {
        std::vector<internal::ur_ref_t> matches;
        for(const auto &t : get_tree(u, "", level::highest)) {
            if(t.key.find(match) != std::string_view::npos) matches.push_back(t);
        }
        return matches;
    }

    std::vector<internal::ur_ref_t> search(std::string_view match) {
        std::vector<internal::ur_ref_t> matches;
        for(const auto &t : get_tree("", level::highest)) {
            if(t.key.find(match) != std::string_view::npos) matches.push_back(t);
        }
        return matches;
    }
    void set_level(level l) { internal::current_level = l; }

    void print_tree(const tid::ur &u, std::string_view prefix, level l) {
        for(const auto &t : tid::get_tree(u, prefix)) {
            if(t->get_level() <= l) fmt::print("{}\n", t.str());
        }
    }
    void print_tree(std::string_view prefix, level l) {
        for(const auto &t : tid::get_tree(prefix)) {
            if(t->get_level() <= l) fmt::print("{}\n", t.str());
        }
    }
}