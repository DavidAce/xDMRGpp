/*
MIT License

Copyright (c) 2017  Joe Hood

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "../plot.h"
#include "math/cast.h"
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <numeric>

size_t map_to_idx(double val, double in_min, double in_max, size_t num_steps) {
    auto max_index = static_cast<double>(num_steps - 1);
    auto rel_range = max_index / (std::max(in_max, in_min) - std::min(in_max, in_min));
    auto rel_value = (std::max(val, in_min) - std::min(val, in_min)) * rel_range;
    auto rel_index = max_index - rel_value;
    //    fmt::print("val {} | rel {} range {} num {} | idx {}\n", val, rel_value, rel_range, num_steps, rel_index);
    return safe_cast<size_t>(std::floor(std::abs(rel_index)));
}

std::vector<double> resample(const std::vector<double> &ydata, size_t newlength) {
    auto oldlength = ydata.size();

    if(oldlength == newlength) {
        return ydata;
    } else {
        auto newdata = std::vector<double>(newlength);
        auto factor  = static_cast<double>(oldlength) / static_cast<double>(newlength);

        for(size_t newindex = 0; newindex < newlength; newindex++) {
            auto x       = static_cast<double>(newindex) * factor;
            auto x1      = floor(x);
            auto x2      = x1 + 1.0;
            auto y1      = ydata.at(std::min<size_t>(std::max<size_t>(0, safe_cast<size_t>(x1)), oldlength - 1));
            auto y2      = ydata.at(std::min<size_t>(std::max<size_t>(0, safe_cast<size_t>(x2)), oldlength - 1));
            auto y       = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
            auto safeidx = std::min<size_t>(std::max<size_t>(0, newindex), newlength - 1);
            if(safeidx >= newdata.size())
                throw std::runtime_error(
                    fmt::format("safeidx out of bounds: {} >= {} | newlength {} | oldlength {}", safeidx, newdata.size(), newlength, oldlength));
            newdata.at(safeidx) = y;
        }

        newdata.front() = ydata.front();
        newdata.back()  = ydata.back();
        return newdata;
    }
}

AsciiPlotter::AsciiPlotter(std::string_view title) : title(title) {}

AsciiPlotter::AsciiPlotter(std::string_view title, size_t width, size_t height) : width(width), height(height), title(title) {}

void AsciiPlotter::addPlot(const std::vector<double> &xdata_, const std::vector<double> &ydata_, std::string_view label, char marker) {
    if(xdata_.empty()) {
        xdata.resize(ydata_.size());
        std::iota(xdata.begin(), xdata.end(), 0);
    } else {
        xdata = xdata_;
    }
    markers.emplace_back(marker);
    labels.emplace_back(label);
    ydata.emplace_back(ydata_);
}

void AsciiPlotter::addPlot(const std::vector<double> &ydata_, std::string_view label, char marker) { addPlot(std::vector<double>{}, ydata_, label, marker); }

void AsciiPlotter::addStaticPlot(const std::vector<double> &xdata_, const std::vector<double> &ydata_, std::string_view label, char marker) {
    if(xdata_.empty()) {
        xdata_s.resize(ydata_.size());
        std::iota(xdata_s.begin(), xdata_s.end(), 0);
    } else {
        xdata_s = xdata_;
    }
    markers_s.emplace_back(marker);
    labels_s.emplace_back(label);
    ydata_s.emplace_back(ydata_);
}

void AsciiPlotter::addStaticPlot(const std::vector<double> &ydata_, std::string_view label, char marker) {
    addStaticPlot(std::vector<double>{}, ydata_, label, marker);
}

void AsciiPlotter::show() {
    if(xdata.empty()) throw std::runtime_error("AsciiPlotter::show(): xdata is empty");

    auto xmin = xdata.front();
    auto xmax = xdata.back();

    auto ymax    = std::numeric_limits<double>::quiet_NaN();
    auto ymin    = std::numeric_limits<double>::quiet_NaN();
    auto minelem = [](const auto &vec) -> double {
        auto res = 0.0;
        for(const auto &v : vec) {
            if(std::isinf(res) or std::isnan(res)) continue;
            res = std::min(res, v);
        }
        return res;
    };
    auto maxelem = [](const auto &vec) -> double {
        auto res = 0.0;
        for(const auto &v : vec) {
            if(std::isinf(res) or std::isnan(res)) continue;
            res = std::max(res, v);
        }
        return res;
    };
    for(const auto &y : ydata_s) {
        ymax = std::isnan(ymax) ? maxelem(y) : std::max(ymax, maxelem(y));
        ymin = std::isnan(ymin) ? minelem(y) : std::min(ymin, minelem(y));
    }
    for(const auto &y : ydata) {
        auto max_it = std::max_element(y.begin(), y.end());
        auto min_it = std::min_element(y.begin(), y.end());
        if(max_it != y.end()) ymax = std::isnan(ymax) ? *max_it : std::max(ymax, *max_it);
        if(min_it != y.end()) ymin = std::isnan(ymin) ? *min_it : std::min(ymin, *min_it);
    }
    std::vector<std::string> canvas(height, std::string(width, ' '));
    for(size_t curve = 0; curve < ydata_s.size(); curve++) {
        auto resampled = resample(ydata_s.at(curve), width);
        for(size_t col = 0; col < width; col++) {
            size_t row             = map_to_idx(resampled.at(col), ymin, ymax, height);
            canvas.at(row).at(col) = markers_s.at(curve);
        }
    }
    for(size_t curve = 0; curve < ydata.size(); curve++) {
        auto resampled = resample(ydata.at(curve), width);
        for(size_t col = 0; col < width; col++) {
            size_t row = map_to_idx(resampled.at(col), ymin, ymax, height);
            if(row < height) canvas.at(row).at(col) = markers.at(curve);
        }
    }

    // margin
    auto lmargin = fmt::format("{0:^{1}}", "", margin);

    // title:
    fmt::print("\n{0}{1:^{2}}\n", lmargin, title, width);

    // main plot plane:
    fmt::print(" {0:8.2g} ┌{1:─^{2}}┐\n", ymax, "", width);
    for(const auto &rowstr : canvas) { fmt::print("{}│{}│\n", lmargin, rowstr); }
    fmt::print(" {0:8.2g} └{1:─^{2}}┘\n", ymin, "", width);

    // xaxis
    fmt::print("{0}{1:<{4}}{2}{3:>{4}}\n", lmargin, fmt::format("{:<8.2g}", xmin), xlabel, fmt::format("{:>8.2g}", xmax), (2 + width - xlabel.size()) / 2);

    // legend:
    if(legend) {
        fmt::print("{0}┌{1:─^{2}}┐\n", lmargin, "", width);
        for(size_t curve = 0; curve < labels_s.size(); curve++)
            fmt::print("{0}│{1:<{2}}│\n", lmargin, fmt::format("    {} {}", markers_s.at(curve), labels_s.at(curve)), width);
        for(size_t curve = 0; curve < labels.size(); curve++)
            fmt::print("{0}│{1:<{2}}│\n", lmargin, fmt::format("    {} {}", markers.at(curve), labels.at(curve)), width);
        fmt::print("{0}└{1:─^{2}}┘\n", lmargin, "", width);
    }
}

void AsciiPlotter::set_margin(size_t margin_) { margin = margin_; }

void AsciiPlotter::set_xlabel(std::string_view label) { xlabel = label; }

void AsciiPlotter::set_ylabel(std::string_view label) { ylabel = label; }

void AsciiPlotter::enable_legend() { legend = true; }
