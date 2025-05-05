cmake_minimum_required(VERSION 3.15)

project(TyNET VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(MyExecutable main.cpp)
target_link_libraries(MyExecutable PRIVATE Catch2::Catch2WithMain)