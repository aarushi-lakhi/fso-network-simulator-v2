build_lib_example(
  NAME fso-5node-mesh
  SOURCE_FILES fso-5node-mesh.cc
  LIBRARIES_TO_LINK
    ${libfso-channel}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
    ${libmobility}
)
