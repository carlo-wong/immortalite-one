cmake_minimum_required(VERSION 3.15...3.29)
project(immortalite_one LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Avoid pybind11 LTO/IPO flag probes that break under some CMake/generator combos.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(IMMORTALITE_ONE_SOURCES
    cpp/bindings.cpp
    cpp/board.cpp
    cpp/movegen.cpp
    cpp/encoding.cpp
    cpp/mcts.cpp
)

pybind11_add_module(_native MODULE ${IMMORTALITE_ONE_SOURCES})

target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

target_compile_definitions(_native PRIVATE
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# Emit the extension next to engine/__init__.py so `from engine import _native` works
# for both editable and plain builds (single-config Ninja).
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine"
)

# scikit-build-core: wheel.install-dir = "engine" means DESTINATION "." lands in engine/.
install(TARGETS _native
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)
