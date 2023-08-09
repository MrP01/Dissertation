add_executable(simulator main.cpp Box.cpp Simulator.cpp)
target_link_libraries(simulator PUBLIC Qt6::Widgets Qt6::Charts)

add_executable(experiments1d experiments.cpp Box.cpp)
add_executable(experiments2d experiments.cpp Box.cpp)
add_executable(experiments3d experiments.cpp Box.cpp)
target_compile_definitions(experiments1d PRIVATE "DIMENSION=1")
target_compile_definitions(experiments2d PRIVATE "DIMENSION=2")
target_compile_definitions(experiments3d PRIVATE "DIMENSION=3")
