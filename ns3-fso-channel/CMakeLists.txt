build_lib(
  LIBNAME fso-channel
  SOURCE_FILES
    model/gamma-gamma-fso-loss-model.cc
  HEADER_FILES
    model/gamma-gamma-fso-loss-model.h
  LIBRARIES_TO_LINK
    ${libpropagation}
    ${libmobility}
    ${libnetwork}
)
