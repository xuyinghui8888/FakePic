#ifndef TEXTURE_RECONSTRUCTION_H
#define TEXTURE_RECONSTRUCTION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
namespace CGP
{
	class RecTexture
	{
		void generateTexture(std::shared_ptr<ConstData> const_data, std::shared_ptr<TextureResult> mesh_res);
	};

}

#endif
