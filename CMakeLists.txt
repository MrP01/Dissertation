cmake_minimum_required(VERSION 3.16)
project(dissertation)

add_definitions("-std=c++23")
add_definitions("-O3")
add_definitions("-g")  # gcc adds debug information (confer `man gcc`)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Charts)

add_subdirectory(simulator/)
