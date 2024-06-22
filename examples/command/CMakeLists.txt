if (WHISPER_SDL2)
    # command
    set(TARGET command)
    add_executable(${TARGET} command.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET}
        PRIVATE
        common common-sdl whisper
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        ${CMAKE_THREAD_LIBS_INIT})
endif ()
