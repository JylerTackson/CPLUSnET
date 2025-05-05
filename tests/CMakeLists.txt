cmake_minimum_required(VERSION 3.10)
project(RunTests)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/..thirdParty/Catch2/src)

#add more executable tests as needed
add_executable(helloWorldTest helloWorldTest.cpp)

target_link_libraries(helloWorldTest Catch2::Catch2)

enable_testing()
add_test(NAME RunTests COMMAND RunTests)