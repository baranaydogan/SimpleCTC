INCLUDE(CheckIncludeFile)

find_path (SLEPC_DIR include/slepc.h
  HINTS ENV SLEPC_DIR
  PATHS
  /usr/lib/slepcdir/3.3		#Debian
  /usr/lib/slepcdir/3.2		#Debian
  /usr/lib/slepcdir/3.1		#Debian
  /usr/lib/slepcdir/3.0.0	#Debian
  $ENV{HOME}/slepc
  DOC "SLEPc Directory")

SET(SLEPC_INCLUDE_DIR "${SLEPC_DIR}/include/")
CHECK_INCLUDE_FILE( ${SLEPC_INCLUDE_DIR}/slepc.h HAVE_SLEPC_H )
FIND_LIBRARY(SLEPC_LIB_SLEPC     slepc )
SET(SLEPC_LIBRARIES ${SLEPC_LIB_SLEPC} CACHE STRING "SLEPc libraries" FORCE)
if (HAVE_SLEPC_H AND SLEPC_DIR AND SLEPC_LIBRARIES )
  set(HAVE_SLEPC 1)
  set(SLEPC_FOUND ON)
endif()
set(SLEPC_INCLUDES ${SLEPC_INCLUDE_DIR} CACHE STRING "SLEPc include path" FORCE)
MARK_AS_ADVANCED( SLEPC_DIR SLEPC_LIB_SLEPC SLEPC_INCLUDES SLEPC_LIBRARIES )
