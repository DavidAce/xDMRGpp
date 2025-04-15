#include "tid.h"
#include <stdexcept>

namespace tid {
    ur::ur(std::string_view label_, level l) noexcept: label(label_), lvl(l) {}

    void ur::tic() noexcept {
        if constexpr (tid::enable) {
            if (lvl == level::disabled) return;
            if (is_measuring)
                fprintf(stderr, "tid: error in tid::ur [%s]: called tic() twice: this timer is already active\n",
                        label.c_str());
            //            if(is_measuring) throw std::runtime_error("Called tic() twice: this timer is already measuring: " + label);
            tic_timepoint = hresclock::now();
            is_measuring = true;
            count++;
        }
    }

    void ur::toc() noexcept {
        if constexpr (tid::enable) {
            if (lvl == level::disabled) return;
            //            if(not is_measuring) throw std::runtime_error("Called toc() twice or without prior tic()");
            if (not is_measuring)
                fprintf(stderr, "tid: error in tid::ur [%s]: called toc() twice or without prior tic()\n",
                        label.c_str());
            toc_timepoint = hresclock::now();
            delta_time = toc_timepoint - tic_timepoint;
            measured_time += delta_time;
            lap_time += delta_time;
            is_measuring = false;
        }
    }

    token ur::tic_token(double add_time) noexcept { return token(*this, add_time); }

    token ur::tic_token(std::string_view prefix,  double add_time) noexcept { return {*this, prefix, add_time}; }

    void ur::set_label(std::string_view label_) noexcept { label = label_; }

    void ur::set_time(double new_time) noexcept {
        if constexpr (tid::enable) {
            measured_time = std::chrono::duration_cast<hresclock::duration>(std::chrono::duration<double>(new_time));
        }
    }

    void ur::add_time(double new_time) noexcept {
        if constexpr (tid::enable) {
            measured_time += std::chrono::duration_cast<hresclock::duration>(std::chrono::duration<double>(new_time));
        }
    }

    void ur::set_count(size_t count_) noexcept {
        if constexpr (tid::enable) count = count_;
    }

    void ur::add_count(size_t count_) noexcept {
        if constexpr (tid::enable) count += count_;
    }

    void ur::start_lap() noexcept {
        if constexpr (tid::enable) {
            lap_time = hresclock::duration::zero();
            lap_timepoint = hresclock::now();
        }
    }

    void ur::set_level(level l) noexcept { lvl = l; }

    std::string ur::get_label() const noexcept { return label; }

    double ur::get_age() const {
        if constexpr (tid::enable) {
            return std::chrono::duration_cast<std::chrono::duration<double>>(
                    hresclock::now() - start_timepoint).count();
        } else
            return 0.0;
    }

    double ur::get_time() const {
        if constexpr (tid::enable) {
            if (is_measuring) {
                return std::chrono::duration_cast<std::chrono::duration<double>>(
                        measured_time + hresclock::now() - tic_timepoint).count();
            } else
                return std::chrono::duration_cast<std::chrono::duration<double>>(measured_time).count();
        } else
            return 0.0;
    }

    double ur::get_time_avg() const { return get_time() / static_cast<double>(count); }

    size_t ur::get_tic_count() const { return count; }

    double ur::get_last_interval() const {
        if constexpr (tid::enable) {
            if (is_measuring) {
                auto delta_temp = hresclock::now() - tic_timepoint;
                return std::chrono::duration_cast<std::chrono::duration<double>>(delta_temp).count();
            } else
                return std::chrono::duration_cast<std::chrono::duration<double>>(delta_time).count();
        } else
            return 0.0;
    }

    double ur::get_lap() const {
        if constexpr (tid::enable) {
            if (is_measuring) {
                // From the measured time, subtract the time since the lap time point
                auto measured_time_since_tic = hresclock::now() - tic_timepoint;
                auto measured_time_until_lap =
                        lap_timepoint >= tic_timepoint ? lap_timepoint - tic_timepoint : hresclock::duration::zero();
                auto measured_time_since_lap =
                        std::chrono::duration_cast<std::chrono::duration<double>>(
                                lap_time + measured_time_since_tic - measured_time_until_lap).count();
                return measured_time_since_lap;
            } else
                return std::chrono::duration_cast<std::chrono::duration<double>>(lap_time).count();
        } else
            return 0.0;
    }

    double ur::restart_lap() {
        if constexpr (tid::enable) {
            double lap = get_lap();
            start_lap();
            return lap;
        } else
            return 0.0;
    }

    level ur::get_level() const { return lvl; }

    void ur::reset() {
        if (tid::enable) {
            measured_time = hresclock::duration::zero();
            delta_time = hresclock::duration::zero();
            lap_time = hresclock::duration::zero();
            start_timepoint = hresclock::now();
            lap_timepoint = hresclock::now();
            is_measuring = false;
        }
    }

    ur &ur::operator=(double other_time_in_seconds) noexcept {
        if constexpr (tid::enable) {
            this->measured_time = std::chrono::duration_cast<hresclock::duration>(
                    std::chrono::duration<double>(other_time_in_seconds));
            this->delta_time = this->measured_time;
        }
        return *this;
    }

    ur &ur::operator+=(double other_time_in_seconds) noexcept {
        if constexpr (tid::enable) {
            this->measured_time += std::chrono::duration_cast<hresclock::duration>(
                    std::chrono::duration<double>(other_time_in_seconds));
            this->delta_time = std::chrono::duration_cast<hresclock::duration>(
                    std::chrono::duration<double>(other_time_in_seconds));
        }
        return *this;
    }

    ur &ur::operator-=(double other_time_in_seconds) noexcept {
        if constexpr (tid::enable) {
            this->measured_time -= std::chrono::duration_cast<hresclock::duration>(
                    std::chrono::duration<double>(other_time_in_seconds));
            this->delta_time = -std::chrono::duration_cast<hresclock::duration>(
                    std::chrono::duration<double>(other_time_in_seconds));
        }
        return *this;
    }

    ur &ur::operator+=(const ur &rhs) noexcept {
        if constexpr (tid::enable) {
            this->measured_time += rhs.measured_time;
            this->delta_time = rhs.delta_time;
            this->count += rhs.count;
        }
        return *this;
    }

    ur &ur::operator-=(const ur &rhs) noexcept {
        if constexpr (tid::enable) {
            measured_time -= rhs.measured_time;
            delta_time = rhs.delta_time;
            count -= std::min(rhs.count, count);
        }
        return *this;
    }

    ur &ur::operator[](std::string_view label_) {
        if (label_.find('.') != std::string_view::npos) {
            throw std::runtime_error("ur error [" + std::string(label_) + "]: label cannot have '.'");
        }
        auto it = ur_under.find(label_);
        bool emplaced = false;
        if (it == ur_under.end()) {
#pragma omp critical(urbracket)
            {
                std::tie(it, emplaced) = ur_under.emplace(
                        std::make_pair(std::string(label_), std::make_shared<tid::ur>(label_)));
                if (emplaced) it->second->set_level(get_level());
            }
            if (it == ur_under.end()) throw std::logic_error("ur::operator[]: emplace failed");
        }
        return *it->second;
    }

    ur &ur::insert(std::string_view label_, level l) {
        if (label_.find('.') != std::string_view::npos) {
            throw std::runtime_error("ur error [" + std::string(label_) + "]: label cannot have '.'");
        }
        auto it = ur_under.find(label_);
        bool emplaced = false;
        if (it == ur_under.end()) {
#pragma omp critical(urinsert)
            {
                std::tie(it, emplaced) = ur_under.emplace(
                        std::make_pair(std::string(label_), std::make_shared<tid::ur>(label_)));
                if (emplaced) it->second->set_level(l == level::parent ? get_level() : l);
            }
            if (it == ur_under.end()) throw std::logic_error("ur::insert: emplace failed");
        }
        return *it->second;
    }
}
