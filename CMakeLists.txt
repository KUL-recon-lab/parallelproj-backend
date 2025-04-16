cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()


project(PARALLELPROJ LANGUAGES C CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(USE_CUDA "Enable CUDA build" ON)
else()
    option(USE_CUDA "Enable CUDA build" OFF)
endif()

# Add an option to build the Python interface
option(BUILD_PYTHON_INTERFACE "Build the Python interface using pybind11" ON)

# Add an option to build the tests
option(BUILD_TESTS "Build the tests" ON)

# Enable automatic symbol export on Windows for shared libraries
if(WIN32 AND BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

################################################################################
### DERIVE VERSION FROM GIT ####################################################
################################################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git describe --tags --long --dirty
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_DESCRIBE)
        # Match versions with optional pre-release identifiers (e.g., v2.0.0-rc)
        string(REGEX MATCH "v?([0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9]+)?)" PROJECT_VERSION "${GIT_DESCRIBE}")
        string(REGEX REPLACE "^v" "" PROJECT_VERSION "${PROJECT_VERSION}")
    else()
        set(PROJECT_VERSION "0.0.0") # Fallback version
    endif()
else()
    get_filename_component(FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME)
    string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9]+)?)" PROJECT_VERSION "${FOLDER_NAME}")
    if(NOT PROJECT_VERSION)
        set(PROJECT_VERSION "0.0.0") # Fallback version
    endif()
endif()

string(REGEX MATCHALL "[0-9]+" VERSION_PARTS "${PROJECT_VERSION}")
list(GET VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET VERSION_PARTS 2 PROJECT_VERSION_PATCH)

message(STATUS "Building version ${PROJECT_VERSION}")

option(DEBUG "Enable debug printing" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG)
endif()

################################################################################
### DEFINE TARGETS #############################################################
################################################################################

# Set the number of parallel jobs for Link Time Optimization (LTO)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS 4) # Adjust the number of jobs as needed
endif()

if(USE_CUDA)
    message(STATUS "CUDA build enabled.")

    set(LIB_SOURCES
        src/joseph3d_fwd.cu
        src/joseph3d_back.cu
        src/cuda_utils.cu
    )

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES all)
    set(PARALLELPROJ_CUDA 1)
    
    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_link_libraries(parallelproj PRIVATE CUDA::cudart)
    target_compile_definitions(parallelproj PUBLIC PARALLELPROJ_CUDA=1) 
    target_compile_definitions(parallelproj PRIVATE PARALLELPROJ_EXPORTS)

    set(TEST_SOURCES
        tests/test_nontof_fwd_back.cpp
        tests/test_nontof_fwd_back_cuda.cu
    )
else()
    message(STATUS "CUDA build disabled.")

    set(LIB_SOURCES
        src/joseph3d_fwd.cpp
        src/joseph3d_back.cpp
    )

    find_package(OpenMP REQUIRED)
    set(PARALLELPROJ_CUDA 0)

    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(parallelproj PUBLIC PARALLELPROJ_CUDA=0)
    target_compile_definitions(parallelproj PRIVATE PARALLELPROJ_EXPORTS)

    set(TEST_SOURCES
        tests/test_nontof_fwd_back.cpp
    )
endif()

target_include_directories(parallelproj
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


################################################################################
### BUILD THE PYTHON INTERFACE #################################################
################################################################################

# Build the Python interface if enabled
if(BUILD_PYTHON_INTERFACE)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python3 found at: ${Python3_EXECUTABLE}")
    message(STATUS "Python3_SITELIB: ${Python3_SITELIB}")

    # Add an option for the Python interface install destination
    set(PYTHON_INTERFACE_INSTALL_DIR ${Python3_SITELIB})
    message(STATUS "Python interface will be installed to: ${PYTHON_INTERFACE_INSTALL_DIR}")

    # Enable the use of FindPython in pybind11
    set(PYBIND11_FINDPYTHON ON)
    # Find pybind11
    find_package(pybind11 REQUIRED)

    find_path(DLPACK_INCLUDE_DIR
    NAMES dlpack/dlpack.h
    DOC "Path to the DLPack include directory")

    # Check if DLPack was found
    if (NOT DLPACK_INCLUDE_DIR)
        message(FATAL_ERROR "DLPack not found. Please install DLPack via conda or provide the path to dlpack.h.")
    else()
        message(STATUS "DLPack found at: ${DLPACK_INCLUDE_DIR}")
    endif()

    # Add the Python module
    pybind11_add_module(parallelproj_backend
        python/bindings.cpp
    )

    target_include_directories(parallelproj_backend PRIVATE ${CMAKE_SOURCE_DIR}/include/parallelproj)
    # Link the core library to the Python module
    target_link_libraries(parallelproj_backend PRIVATE parallelproj)

    # Pass the PARALLELPROJ_CUDA definition to the Python module
    target_compile_definitions(parallelproj_backend PRIVATE PARALLELPROJ_CUDA=${PARALLELPROJ_CUDA})

    # Ensure parallelproj is built before parallelproj_backend
    add_dependencies(parallelproj_backend parallelproj)

    # Install parallelproj_backend to the specified destination
    install(TARGETS parallelproj_backend
        EXPORT parallelprojTargets
        LIBRARY DESTINATION ${PYTHON_INTERFACE_INSTALL_DIR}
        RUNTIME DESTINATION ${PYTHON_INTERFACE_INSTALL_DIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

################################################################################
### BUILD TESTS ################################################################
################################################################################

if(BUILD_TESTS)
    enable_testing()
    
    set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/tests)

    # List all the .txt files that need to be copied
    set(TEST_DATA_FILES
        img.txt
        vstart.txt
        vend.txt
        expected_fwd_vals.txt
    )

    # Copy each file to the build directory
    foreach(TEST_FILE ${TEST_DATA_FILES})
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${TEST_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_DATA_DIR}/${TEST_FILE} ${CMAKE_BINARY_DIR}/${TEST_FILE}
            DEPENDS ${TEST_DATA_DIR}/${TEST_FILE}
        )
        list(APPEND TEST_DATA_OUTPUTS ${CMAKE_BINARY_DIR}/${TEST_FILE})
    endforeach()

    # Add a custom target to ensure the files are copied before running tests
    add_custom_target(copy_test_data ALL DEPENDS ${TEST_DATA_OUTPUTS})

    # Add the directory containing file_utils.h to the include paths
    include_directories(${CMAKE_SOURCE_DIR}/tests)

    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests)
        add_dependencies(${TEST_NAME} copy_test_data) # Ensure test data is copied before running tests
        if(USE_CUDA)
            target_link_libraries(${TEST_NAME} PRIVATE parallelproj CUDA::cudart)
        else()
            target_link_libraries(${TEST_NAME} PRIVATE parallelproj OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    if(BUILD_PYTHON_INTERFACE)
        ### ADD THE PYTHON INTERFACE TESTS ###
        find_program(PYTEST_EXECUTABLE pytest REQUIRED)

        add_test(NAME python_interface_test COMMAND ${Python3_EXECUTABLE} -m  pytest -v ${CMAKE_SOURCE_DIR}/tests)

        set_tests_properties(python_interface_test PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:parallelproj_backend>$<SEMICOLON>$ENV{PYTHONPATH}") 
    endif()
endif()

################################################################################
### INSTALL AND CONFIGS ########################################################
################################################################################

# Install the library and headers
install(TARGETS parallelproj
    EXPORT parallelprojTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export the build tree for other projects
export(EXPORT parallelprojTargets
    FILE ${CMAKE_BINARY_DIR}/parallelprojTargets.cmake
    NAMESPACE parallelproj::
)

# Install the export configuration
install(EXPORT parallelprojTargets
    FILE parallelprojTargets.cmake
    NAMESPACE parallelproj::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
)

# Create and install a package configuration file
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(${CMAKE_SOURCE_DIR}/cmake/parallelprojConfig.cmake.in
    ${CMAKE_BINARY_DIR}/parallelprojConfig.cmake
    @ONLY
)

install(FILES
    ${CMAKE_BINARY_DIR}/parallelprojConfig.cmake
    ${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
)

################################################################################
### UNINSTALL SCRIPT ###########################################################
################################################################################

# Create an uninstall script
if(NOT TARGET uninstall)
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
        ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

