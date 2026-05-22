# boost::math (header-only) supplies the inverse regularized incomplete gamma
# function used by the correlated fading model; same Boost discovery pattern
# as contrib/ai.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

build_lib(
  LIBNAME fso-channel
  SOURCE_FILES
    helper/fso-topology-helper.cc
    model/correlated-gamma-gamma-fading.cc
    model/gamma-gamma-fso-loss-model.cc
  HEADER_FILES
    helper/fso-topology-helper.h
    model/correlated-gamma-gamma-fading.h
    model/gamma-gamma-fso-loss-model.h
  LIBRARIES_TO_LINK
    ${libpropagation}
    ${libmobility}
    ${libnetwork}
    ${libpoint-to-point}
  TEST_SOURCES
    test/fso-channel-test-suite.cc
    test/fso-channel-correlated-test-suite.cc
)
