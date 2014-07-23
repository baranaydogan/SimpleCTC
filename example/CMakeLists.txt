cmake_minimum_required(VERSION 2.8)

PROJECT(simpleExample_Project)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -O2")
endif()

SET(CUSTOM_CMAKE_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/../src/cmake/modules/)
SET(CMAKE_MODULE_PATH "${CUSTOM_CMAKE_MODULES}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
include_directories(${OpenMP_INCLUDE_DIRS})

find_package(MPI)
include_directories(${MPI_INCLUDE_DIR})

find_package(PETSc)
include_directories(${PETSC_INCLUDE_DIR})

find_package(SLEPc)
include_directories(${SLEPC_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/ )
file(GLOB SimpleCTC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src/*)
add_library(lSimpleCTC STATIC ${SimpleCTC_SRC})

SET(example_SRC main.cpp)

ADD_EXECUTABLE(simpleExample ${example_SRC})
	
TARGET_LINK_LIBRARIES(simpleExample
	rt
	mpi
	lSimpleCTC
	${PETSC_LIBRARIES}
	${SLEPC_LIBRARIES}
	)

INSTALL(TARGETS simpleExample DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})