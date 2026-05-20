
if(DEFINED DMRG_ENABLE_FP32)
    message(FATAL_ERROR "The option [DMRG_ENABLE_FP32] has been removed. Replace by:\n"
            "DMRG_ENABLE_32BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_CX32)
    message(FATAL_ERROR "The option [DMRG_ENABLE_CX32] has been removed. Replace by:\n"
            "DMRG_ENABLE_32BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_FP64)
    message(FATAL_ERROR "The option [DMRG_ENABLE_FP64] has been removed. Replace by:\n"
            "DMRG_ENABLE_64BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_CX64)
    message(FATAL_ERROR "The option [DMRG_ENABLE_CX64] has been removed. Replace by:\n"
            "DMRG_ENABLE_64BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_FP80)
    message(FATAL_ERROR "The option [DMRG_ENABLE_FP80] has been removed. Replace by:\n"
            "DMRG_ENABLE_80BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_CX80)
    message(FATAL_ERROR "The option [DMRG_ENABLE_CX80] has been removed. Replace by:\n"
            "DMRG_ENABLE_80BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_FP128)
    message(FATAL_ERROR "The option [DMRG_ENABLE_FP128] has been removed. Replace by:\n"
            "DMRG_ENABLE_128BIT:BOOL=[TRUE|FALSE]")
endif()
if(DEFINED DMRG_ENABLE_CX128)
    message(FATAL_ERROR "The option [DMRG_ENABLE_CX128] has been removed. Replace by:\n"
            "DMRG_ENABLE_128BIT:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_DOWNLOAD_METHOD)
    message(FATAL_ERROR "The variable [DMRG_DOWNLOAD_METHOD] has been deprecated. Replace by:\n"
            "DMRG_PACKAGE_MANAGER:STRING=[find|cmake|fetch|find-or-cmake|find-or-fetch|conan]")
endif()

if(DMRG_DEPS_IN_SUBDIR)
    message(DEPRECATION "The option [DMRG_DEPS_IN_SUBDIR] will be ignored.")
endif()

if(DMRG_PRINT_INFO)
    message(DEPRECATION "The option [DMRG_PRINT_INFO] has been deprecated. Replace by:\n"
            "CMake CLI option --loglevel=[TRACE|DEBUG|VERBOSE|STATUS...] or\n"
            "set CMAKE_MESSAGE_LOG_LEVEL=[TRACE|DEBUG|VERBOSE|STATUS...]"
            )
endif()

if(DMRG_PROFILE_BUILD)
    message(FATAL_ERROR "The option [DMRG_PROFILE_BUILD] has been deprecated. Replace by:\n"
            "COMPILER_PROFILE_BUILD:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_ENABLE_ASAN)
    message(FATAL_ERROR "The option [DMRG_ENABLE_ASAN] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_ASAN:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_ENABLE_USAN)
    message(FATAL_ERROR "The option [DMRG_ENABLE_USAN] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_USAN:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_ENABLE_LTO)
    message(FATAL_ERROR "The option [DMRG_ENABLE_LTO] has been deprecated. Replace by:\n"
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_ENABLE_PCH)
    message(FATAL_ERROR "The option [DMRG_ENABLE_PCH] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_PCH:BOOL=[TRUE|FALSE]")
endif()

if(DMRG_ENABLE_CCACHE)
    message(FATAL_ERROR "The option [DMRG_ENABLE_CCACHE] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_CCACHE:BOOL=[TRUE|FALSE]")
endif()

if(COMPILER_MARCH)
    message(FATAL_ERROR "The option [COMPILER_MARCH] has been deprecated.")
endif()
if(COMPILER_MTUNE)
    message(FATAL_ERROR "The option [COMPILER_MTUNE] has been deprecated.")
endif()
if(COMPILER_ENABLE_LTO)
    message(FATAL_ERROR "The option [COMPILER_ENABLE_LTO] has been deprecated.. Replace by:\n"
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=[TRUE|FALSE]")
endif()
if(COMPILER_ENABLE_MOLD)
    message(FATAL_ERROR "The option [COMPILER_ENABLE_MOLD] has been deprecated.. Replace by:\n"
                    "\tCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold\n"
                    "\tCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold")
endif()
if(DMRG_MARCH)
    message(FATAL_ERROR "The option [DMRG_MARCH] has been deprecated.")
endif()
if(DMRG_MTUNE)
    message(FATAL_ERROR "The option [DMRG_MTUNE] has been deprecated.")
endif()

if(DMRG_ENABLE_MKL)
    message(FATAL_ERROR "The option [DMRG_ENABLE_MKL] has been deprecated.")
endif()
if(DMRG_ENABLE_THREADS)
    message(FATAL_ERROR "The option [DMRG_ENABLE_MKL] has been deprecated.")
endif()
if(DMRG_ENABLE_RSVD)
    message(FATAL_ERROR "The option [DMRG_ENABLE_RSVD] has been deprecated.")
endif()

if(DMRG_ENABLE_COVERAGE)
    message(FATAL_ERROR "The option [DMRG_ENABLE_COVERAGE] has been deprecated. Replace by: \n"
            "Use COMPILER_ENABLE_COVERAGE:BOOL=[TRUE|FALSE]")
endif()
