#ifndef CNODEBH_H
#define CNODEBH_H

#include "TriAnyTopology.h"
#include "SurfaceMeshReader.h"
#include "ContactTarget.h"

namespace SDF
{
	const unsigned int max_tri_cell = 4;

	class CNodeBH  // Barnes and Hut
	{
	public:

	  CNodeBH()
	  {
		ichild_ = -1;
		n_tri_cell_ = 0;
	  }

	public:
	  double cent_[3];
	  double hw_;
	  int ichild_;
	  ////
	  double centroid_[3];
	  double weight_;
	  ////
	  unsigned int n_tri_cell_;
	  unsigned int aIndTriCell[max_tri_cell];

	};

}


#endif
