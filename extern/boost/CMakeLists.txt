# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2026 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include(FetchContent)
# Require a recent Boost that dependant traccc requires (for fiber).
# Tell the user what's happening.
message( STATUS "Building recent Boost as part of the scheduler tests" )

# Components required by the project.
set(BOOST_ENABLE_CMAKE ON )
set(BOOST_SKIP_INSTALL_RULES OFF)
set(BUILD_SHARED_LIBS ON)

FetchContent_Declare( Boost OVERRIDE_FIND_PACKAGE
   URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
   URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
   DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable( Boost )
find_package(Boost 1.90 REQUIRED)

# Stolen from Bosst's CMake files. Unclear why we should repeat here (needed by ACTS).
add_library(Boost::boost INTERFACE IMPORTED)
set_property(TARGET Boost::boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES Boost::headers)
