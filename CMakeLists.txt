cmake_minimum_required(VERSION 3.20)
project(llm_tracer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v5.0.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ftxui)

add_executable(llm_tracer
    src/main.cpp
    src/model_hook.cpp
    src/model_selector.cpp
    src/demo_runner.cpp
    src/tui_app.cpp
)

target_include_directories(llm_tracer PRIVATE src)

target_link_libraries(llm_tracer PRIVATE
    ftxui::component
    ftxui::dom
    ftxui::screen
)

if(MSVC)
    target_compile_options(llm_tracer PRIVATE /W3 /utf-8)
else()
    target_compile_options(llm_tracer PRIVATE -Wall -Wextra)
endif()
