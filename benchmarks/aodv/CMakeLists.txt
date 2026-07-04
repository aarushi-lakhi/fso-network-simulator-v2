# Built by symlinking this directory into ns-3's scratch/ as
# fso-aodv-baseline (see setup/link_fso_modules.sh), exactly like
# ns3-rl-router/sim. Requires the fso-channel contrib module.
build_exec(
  EXECNAME fso-aodv-baseline
  SOURCE_FILES fso-aodv-baseline.cc
  LIBRARIES_TO_LINK
    ${libaodv}
    ${libfso-channel}
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libmobility}
    ${libflow-monitor}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/fso-aodv-baseline
)
