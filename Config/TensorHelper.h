#ifndef TENSOR_HELPER_H
#define TENSOR_HELPER_H
#include "../Basic/CGPBaseHeader.h"
#include "JsonBased.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	//get tensor type
	namespace tensorHelper 
	{
		void rawToBinary(const json& input);
	}
}

#endif
