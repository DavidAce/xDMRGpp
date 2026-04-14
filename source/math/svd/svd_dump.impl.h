
#pragma once
#include "../svd.h"
#include "config/blas_backend.h"
#include "config/settings.h"
#include <Eigen/src/Core/util/Macros.h>
#include <h5pp/h5pp.h>
namespace svd::internal {
    template<typename Scalar>
    DumpSVD<Scalar>::~DumpSVD() {
        if(svd_save == save::NONE) return;
        auto directory  = h5pp::fs::path(settings::storage::output_filepath).parent_path().string();
        auto filepath   = fmt::format("{}/svd-save-{}.h5", directory, settings::input::seed);
        auto file       = h5pp::File(filepath, h5pp::FilePermission::READWRITE);
        auto group_num  = 0;
        auto group_name = fmt::format("svd_{}", group_num);
        if(svd_save == save::ALL)
            while(file.linkExists(group_name)) group_name = fmt::format("svd_{}", ++group_num);
        if(svd_save == save::LAST) group_name = "svd-last";
        if(svd_save == save::FAIL) group_name = "svd-fail";
        std::string sfx;
        if(std::is_same_v<Scalar, fp32>) sfx = "fp32";
        if(std::is_same_v<Scalar, fp64>) sfx = "fp64";
        if(std::is_same_v<Scalar, fp128>) sfx = "fp128";
        if(std::is_same_v<Scalar, cx32>) sfx = "cx32";
        if(std::is_same_v<Scalar, cx64>) sfx = "cx64";
        if(std::is_same_v<Scalar, cx128>) sfx = "cx128";

        if(A.size() > 0) file.writeDataset(A, fmt::format("{}/A_{}", group_name, sfx), H5D_layout_t::H5D_CHUNKED);
        if(U.size() > 0) file.writeDataset(U, fmt::format("{}/U_{}", group_name, sfx), H5D_layout_t::H5D_CHUNKED);
        if(S.size() > 0) file.writeDataset(S, fmt::format("{}/S_{}", group_name, sfx), H5D_layout_t::H5D_CHUNKED);
        if(VT.size() > 0) file.writeDataset(VT, fmt::format("{}/VT_{}", group_name, sfx), H5D_layout_t::H5D_CHUNKED);

        file.writeAttribute(settings::input::seed, group_name, "seed");
        file.writeAttribute(rank_max, group_name, "rank_max");
        file.writeAttribute(enum2sv(svd_lib), group_name, "svd_lib");
        file.writeAttribute(enum2sv(svd_rtn), group_name, "svd_rtn");
        file.writeAttribute(truncation_lim, group_name, "truncation_lim");
        file.writeAttribute(switchsize_gejsv, group_name, "switchsize_gejsv");
        file.writeAttribute(switchsize_gesvd, group_name, "switchsize_gesvd");
        file.writeAttribute(switchsize_gesdd, group_name, "switchsize_gesdd");
        file.writeAttribute(info, group_name, "info");
        file.writeAttribute(truncation_error, group_name, "truncation_error");

        if(svd_lib == svd::lib::lapack) {
            file.writeAttribute(std::string(::config::blas::backend_name()), group_name, "blas_backend");
            file.writeAttribute(::config::blas::description(), group_name, "blas_backend_description");
        } else if(svd_lib == svd::lib::eigen) {
            auto eigen_version = fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
            file.writeAttribute(eigen_version, group_name, "Eigen Version");
        }
    }
}
