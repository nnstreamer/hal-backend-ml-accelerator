CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(hal-backend-ml CXX)
INCLUDE(GNUInstallDirs)

option(ENABLE_DUMMY "Enable dummy-passthrough backend" OFF)
option(ENABLE_VIVANTE "Enable vivante backend" OFF)
option(ENABLE_SNPE "Enable snpe backend" OFF)

SET(HAL_LIBDIR ${CMAKE_HAL_LIBDIR_PREFIX})
SET(HAL_LICENSEDIR ${CMAKE_HAL_LICENSEDIR_PREFIX})

INCLUDE(FindPkgConfig)

# Common Options
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -fomit-frame-pointer -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Set linker flags. Let built shared libraries link so in /hal/lib directory.
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed -Wl,--rpath=${HAL_LIBDIR}")
SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed -Wl,--rpath=${HAL_LIBDIR}")

SET(UTIL_SRCS
  ${PROJECT_SOURCE_DIR}/src/hal-backend-ml-util.cc
)

pkg_check_modules(pkgs REQUIRED
  hal-rootstrap
)

FOREACH(flag ${pkgs_CFLAGS})
  SET(EXTRA_CFLAGS "${CMAKE_C_FLAGS} ${flag}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
ENDFOREACH(flag)

# dummy-passthrough
IF(ENABLE_DUMMY)
SET(DUMMY_PASSTHROUGH_SRCS
${PROJECT_SOURCE_DIR}/src/hal-backend-ml-dummy-passthrough.cc
)

SET(DUMMY_PASSTHROUGH_LIBRARY_NAME "hal-backend-ml-dummy-passthrough")
ADD_LIBRARY(${DUMMY_PASSTHROUGH_LIBRARY_NAME} SHARED ${DUMMY_PASSTHROUGH_SRCS} ${UTIL_SRCS})
TARGET_LINK_LIBRARIES(${DUMMY_PASSTHROUGH_LIBRARY_NAME} ${pkgs_LDFLAGS})
INSTALL(TARGETS ${DUMMY_PASSTHROUGH_LIBRARY_NAME} DESTINATION ${HAL_LIBDIR} COMPONENT RuntimeLibraries)
ENDIF()

# vivante
IF(ENABLE_VIVANTE)
SET(VIVANTE_SRCS
${PROJECT_SOURCE_DIR}/src/hal-backend-ml-vivante.cc
)

SET(VIVANTE_LIBRARY_NAME "hal-backend-ml-vivante")
pkg_check_modules(vivante_build_dep_pkgs REQUIRED
  hal-rootstrap
)

FOREACH(flag ${vivante_build_dep_pkgs_CFLAGS})
  SET(EXTRA_CFLAGS "${CMAKE_C_FLAGS} ${flag}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
ENDFOREACH(flag)

ADD_LIBRARY(${VIVANTE_LIBRARY_NAME} SHARED ${VIVANTE_SRCS} ${UTIL_SRCS})
TARGET_LINK_LIBRARIES(${VIVANTE_LIBRARY_NAME} ${vivante_build_dep_pkgs_LDFLAGS})
INSTALL(TARGETS ${VIVANTE_LIBRARY_NAME} DESTINATION ${HAL_LIBDIR} COMPONENT RuntimeLibraries)
ENDIF()

# snpe
IF(ENABLE_SNPE)
pkg_check_modules(snpe_build_dep_pkgs REQUIRED
  hal-rootstrap
)

FOREACH(flag ${snpe_build_dep_pkgs_CFLAGS})
  SET(EXTRA_CFLAGS "${CMAKE_C_FLAGS} ${flag}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
ENDFOREACH(flag)

SET(SNPE_SRCS
${PROJECT_SOURCE_DIR}/src/hal-backend-ml-snpe.cc
)

SET(SNPE_LIBRARY_NAME "hal-backend-ml-snpe")
ADD_LIBRARY(${SNPE_LIBRARY_NAME} SHARED ${SNPE_SRCS} ${UTIL_SRCS})
TARGET_LINK_LIBRARIES(${SNPE_LIBRARY_NAME} ${snpe_build_dep_pkgs_LDFLAGS})
INSTALL(TARGETS ${SNPE_LIBRARY_NAME} DESTINATION ${HAL_LIBDIR} COMPONENT RuntimeLibraries)
ENDIF()
