if(DMRG_ENABLE_DOCS)
    list(APPEND CONDA_ROOT
         ${CONDA_PREFIX}
         $ENV{HOME}/anaconda3
         $ENV{HOME}/miniconda3
         $ENV{HOME}/anaconda3/envs/dmrg
         $ENV{HOME}/miniconda3/envs/dmrg)

    list(APPEND Doxygen_ROOT ${CONDA_ROOT})
    list(APPEND Sphinx_ROOT ${CONDA_ROOT})

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR
                "Documentation build requires a Python 3 interpreter.\n"
                "Activate your conda environment before configuring CMake, for example:\n"
                "  conda activate dmrg")
    endif()

    find_package(Doxygen QUIET)
    if(NOT Doxygen_FOUND OR NOT DOXYGEN_EXECUTABLE)
        message(FATAL_ERROR
                "Documentation build requires the 'doxygen' executable.\n"
                "Install it, for example, with:\n"
                "  conda activate dmrg\n"
                "  conda install -c conda-forge doxygen\n"
                "or use your system package manager, e.g.:\n"
                "  sudo apt install doxygen")
    endif()

    find_package(Sphinx QUIET)
    if(NOT Sphinx_FOUND OR NOT SPHINX_EXECUTABLE)
        message(FATAL_ERROR
                "Documentation build requires the 'sphinx-build' executable.\n"
                "Install the Python documentation packages in your conda environment:\n"
                "  conda activate dmrg\n"
                "  python -m pip install sphinx breathe myst-parser sphinx-rtd-theme\n"
                "Then re-run CMake from the same activated environment.")
    endif()

    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import sphinx, breathe, myst_parser, sphinx_rtd_theme"
            RESULT_VARIABLE DMRG_DOCS_PY_DEPS_RESULT
            OUTPUT_QUIET
            ERROR_QUIET)
    if(NOT DMRG_DOCS_PY_DEPS_RESULT EQUAL 0)
        message(FATAL_ERROR
                "Documentation build is missing required Python packages.\n"
                "Install them in your conda environment with:\n"
                "  conda activate dmrg\n"
                "  python -m pip install sphinx breathe myst-parser sphinx-rtd-theme")
    endif()

    set(DMRG_DOCS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(DMRG_DOCS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(DMRG_DOXYGEN_OUTPUT_DIR ${DMRG_DOCS_BUILD_DIR}/doxygen)
    set(DMRG_DOXYGEN_XML_DIR ${DMRG_DOXYGEN_OUTPUT_DIR}/xml)
    set(DMRG_SPHINX_SOURCE_DIR ${DMRG_DOCS_SOURCE_DIR}/source)
    set(DMRG_SPHINX_HTML_DIR ${DMRG_DOCS_BUILD_DIR}/sphinx)
    set(DMRG_DOXYGEN_CONFIG_SRC ${DMRG_DOCS_SOURCE_DIR}/Doxyfile)
    set(DMRG_DOXYGEN_CONFIG_BIN ${DMRG_DOCS_BUILD_DIR}/Doxyfile)

    file(READ "${DMRG_DOXYGEN_CONFIG_SRC}" DMRG_DOXYFILE_CONTENT)
    string(REPLACE "PROJECT_NUMBER         = 2.1.0"
                   "PROJECT_NUMBER         = ${PROJECT_VERSION}"
                   DMRG_DOXYFILE_CONTENT "${DMRG_DOXYFILE_CONTENT}")
    string(REPLACE "OUTPUT_DIRECTORY       ="
                   "OUTPUT_DIRECTORY       = ${DMRG_DOXYGEN_OUTPUT_DIR}"
                   DMRG_DOXYFILE_CONTENT "${DMRG_DOXYFILE_CONTENT}")
    string(REPLACE "INPUT                  = ../source"
                   "INPUT                  = ${PROJECT_SOURCE_DIR}/source"
                   DMRG_DOXYFILE_CONTENT "${DMRG_DOXYFILE_CONTENT}")
    file(GENERATE OUTPUT "${DMRG_DOXYGEN_CONFIG_BIN}" CONTENT "${DMRG_DOXYFILE_CONTENT}")

    message(STATUS "Documentation Python interpreter: ${Python3_EXECUTABLE}")
    message(STATUS "Documentation Doxygen executable: ${DOXYGEN_EXECUTABLE}")
    message(STATUS "Documentation Sphinx executable: ${SPHINX_EXECUTABLE}")
    message(STATUS "Documentation Doxygen XML output: ${DMRG_DOXYGEN_XML_DIR}")
    message(STATUS "Documentation Sphinx HTML output: ${DMRG_SPHINX_HTML_DIR}")

    add_custom_target(dmrg-docs
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${DMRG_DOXYGEN_OUTPUT_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${DMRG_SPHINX_HTML_DIR}"
            COMMAND "${DOXYGEN_EXECUTABLE}" "${DMRG_DOXYGEN_CONFIG_BIN}"
            COMMAND "${SPHINX_EXECUTABLE}" -b html
                    -Dbreathe_projects.DMRG++="${DMRG_DOXYGEN_XML_DIR}"
                    "${DMRG_SPHINX_SOURCE_DIR}" "${DMRG_SPHINX_HTML_DIR}"
            WORKING_DIRECTORY "${DMRG_DOCS_SOURCE_DIR}"
            USES_TERMINAL
            COMMENT "Generating documentation with Doxygen and Sphinx"
            VERBATIM)
endif()
