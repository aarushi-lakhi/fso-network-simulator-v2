build_lib(
  LIBNAME fso-channel
  SOURCE_FILES
    helper/fso-topology-helper.cc
    model/gamma-gamma-fso-loss-model.cc
  HEADER_FILES
    helper/fso-topology-helper.h
    model/gamma-gamma-fso-loss-model.h
  LIBRARIES_TO_LINK
    ${libpropagation}
    ${libmobility}
    ${libnetwork}
    ${libpoint-to-point}
  TEST_SOURCES
    test/fso-channel-test-suite.cc
)
