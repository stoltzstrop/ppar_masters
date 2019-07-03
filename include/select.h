#ifndef SELECT_H
#define SELECT_H

/*
 * Macros to select which data layouts used in abstractCL
 * mutually exclusive here, but doesn't have to be necessarily
 */

#ifdef NONE
#include "grid_structs_none.h"
#include "CLGridGetSetNone.h"
#endif
#ifdef FULL
#include "grid_structs_full.h"
#include "CLGridGetSetFull.h"
#endif
#ifdef ONELAYER
#include "grid_structs_onelayer.h"
#include "CLGridGetSetOneLayer.h"
#endif
#ifdef TWOLAYER
#include "grid_structs_twolayer.h"
#include "CLGridGetSetTwoLayer.h"
#endif
#ifdef TWOTWOLAYER
#include "grid_structs_twotwolayer.h"
#include "CLGridGetSetTwoTwoLayer.h"
#endif
#ifdef STRUCTARR
#include "grid_structs_structarr.h"
#include "CLGridGetSetStructArr.h"
#endif

#endif
