
#pragma once
#include "../svd.h"
#include "config/settings.h"
#include <Eigen/src/Core/util/Macros.h>
#include <h5pp/h5pp.h>


template<typename Scalar>
svd::internal::DumpSVD<Scalar>::~DumpSVD() {
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

    if(svd_lib == svd::lib::lapacke) {
#if defined(SVD_SAVE_OPENBLAS_ATTRIBUTES)
        file.writeAttribute(OPENBLAS_VERSION, "OPENBLAS_VERSION", group_name);
        file.writeAttribute(openblas_get_num_threads(), "openblas_get_num_threads", group_name);
        file.writeAttribute(openblas_get_parallel(), "openblas_parallel_mode", group_name);
        file.writeAttribute(openblas_get_corename(), "openblas_get_corename", group_name);
        file.writeAttribute(openblas_get_config(), "openblas_get_config()", group_name);
        file.writeAttribute(OPENBLAS_GEMM_MULTITHREAD_THRESHOLD, "OPENBLAS_GEMM_MULTITHREAD_THRESHOLD", group_name);
#endif

#if defined(SVD_SAVE_MKL_ATTRIBUTES)
        MKLVersion Version;
        mkl_get_version(&Version);
        file.writeAttribute(Version.MajorVersion, "Intel-MKL-MajorVersion", group_name);
        file.writeAttribute(Version.MinorVersion, "Intel-MKL-MinorVersion", group_name);
        file.writeAttribute(Version.UpdateVersion, "Intel-MKL-UpdateVersion", group_name);
#endif
#if defined(SVD_SAVE_FLEXIBLAS_ATTRIBUTES)
        file.writeAttribute(FLEXIBLAS_DEFAULT_LIB_PATH, group_name, "FLEXIBLAS_DEFAULT_LIB_PATH");
        file.writeAttribute(FLEXIBLAS_VERSION, group_name, "FLEXIBLAS_VERSION");
#endif

    } else if(svd_lib == svd::lib::eigen) {
        auto eigen_version = fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
        file.writeAttribute(eigen_version, group_name, "Eigen Version");
    }
}