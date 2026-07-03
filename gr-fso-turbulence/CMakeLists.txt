# gr-fso-turbulence — GNU Radio OOT module for FSO atmospheric fading.
# Pure-Python module: no compiled components, so no C++ toolchain is required.

cmake_minimum_required(VERSION 3.16)
project(gr-fso-turbulence NONE)

find_package(Gnuradio "3.10" REQUIRED COMPONENTS runtime)
include(GrPython)

GR_PYTHON_INSTALL(
    FILES
        python/fso_turbulence/__init__.py
        python/fso_turbulence/fso_fading_channel.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/fso_turbulence
)

install(
    FILES grc/fso_turbulence_fso_fading_channel.block.yml
    DESTINATION ${GRC_BLOCKS_DIR}
)
