if (WHISPER_SDL2)
    # talk
    set(TARGET talk)
    add_executable(${TARGET} talk.cpp gpt-2.cpp)
    target_link_libraries(${TARGET}
        PRIVATE
        common common-sdl whisper
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        ${CMAKE_THREAD_LIBS_INIT})

    include(DefaultTargetOptions)
endif ()
