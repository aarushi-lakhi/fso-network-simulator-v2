# Built by symlinking this directory into ns-3's scratch/ (see README.md);
# scratch/CMakeLists.txt add_subdirectory()s any subdir with a CMakeLists.txt.
# Requires the ai (contrib/ai) and fso-channel (contrib/fso-channel) modules.
build_exec(
  EXECNAME fso-rl-env
  SOURCE_FILES fso-rl-env.cc
  LIBRARIES_TO_LINK
    ${libai}
    ${libfso-channel}
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libmobility}
    ${libflow-monitor}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/fso-rl-env
)
