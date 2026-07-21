# Copyright (c) 2026, Oracle and/or its affiliates.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 2.0,
# as published by the Free Software Foundation.

IF(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
  PROJECT(mysql_vector_diskann_runtime LANGUAGES CXX)
ENDIF()

ADD_LIBRARY(mysql_vector_diskann_runtime STATIC
  src/mysql_vector_diskann_runtime.cc
)

TARGET_INCLUDE_DIRECTORIES(mysql_vector_diskann_runtime
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

IF(TARGET mysql_vector_diskann_offline_adapter)
  TARGET_COMPILE_DEFINITIONS(mysql_vector_diskann_runtime
    PRIVATE MYSQL_VECTOR_DISKANN_RUNTIME_WITH_OFFLINE_ADAPTER=1)
  TARGET_LINK_LIBRARIES(mysql_vector_diskann_runtime
    PRIVATE mysql_vector_diskann_offline_adapter)
ENDIF()

SET_TARGET_PROPERTIES(mysql_vector_diskann_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
