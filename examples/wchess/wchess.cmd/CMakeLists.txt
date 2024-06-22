if (WHISPER_SDL2)
    set(TARGET wchess)
    add_executable(${TARGET} wchess.cmd.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET}
        PRIVATE
        wchess-core common-sdl
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        ${CMAKE_THREAD_LIBS_INIT})
endif ()
