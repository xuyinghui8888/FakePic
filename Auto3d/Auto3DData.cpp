#include "Auto3DData.h"
#include "../Basic/SafeGuard.h"
#include "../Config/JsonHelper.h"
using namespace AUTO3D;
using namespace rttr;
RTTR_REGISTRATION
{
registration::class_<Auto3DData>("AUTO3D::Auto3DData").constructor<>()
	.property("mouth_data_", &AUTO3D::Auto3DData::mouth_data_)
	;
}
