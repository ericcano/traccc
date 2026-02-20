# Copy benchmark scripts into the build directory at configure time

set(RUN_SCAN_BUILD_DIR "${CMAKE_BINARY_DIR}/run-scan")

set(RUN_SCAN_FILES
    run-scan.py
    run-scan-wrapper.sh)

# one custom command per file, using cmake -E copy_if_different
set(RUN_SCAN_COPY_TARGETS)
foreach(f IN LISTS RUN_SCAN_FILES)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
    set(dst "${RUN_SCAN_BUILD_DIR}/${f}")

    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUN_SCAN_BUILD_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}" "${dst}"
        DEPENDS "${src}"
        COMMENT "copy ${f} into build/run-scan"
        VERBATIM
    )

    list(APPEND RUN_SCAN_COPY_TARGETS "${dst}")
endforeach()

# the all‑target ensures the copies are performed during `cmake --build .`
add_custom_target(run_scan_copy ALL
    DEPENDS ${RUN_SCAN_COPY_TARGETS})
