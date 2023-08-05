add_executable(simulator main.cpp Box.cpp Simulator.cpp)
target_link_libraries(simulator PUBLIC Qt6::Widgets Qt6::Charts)

add_executable(experiments experiments.cpp Box.cpp)
