

#include "ContactTarget.h"

using namespace SDF;

double CContactTarget3D_Array::Projection
 (const double p[3], unsigned int ino0,
  double n[3]) const
{
  double max_pd = -1;
  double max_n[3] = {1,0,0};
  for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
    double n0[3];
    double diff[3] = { p[0]-trans[0],p[1]-trans[1],p[2]-trans[2] };
    double pd0 = apCT[ipct]->Projection(diff,ino0,n0);
    if( ipct == 0 || pd0 > max_pd ){
      max_pd = pd0;
      max_n[0] = n0[0];
      max_n[1] = n0[1];
      max_n[2] = n0[2];
    }
  }
  n[0] = max_n[0];
  n[1] = max_n[1];       
  n[2] = max_n[2];
  return max_pd;
}

void CContactTarget3D_Array::ProjectionDerivative
(const double p[3], unsigned int ino0,
unsigned int iparam_s,
 double dn[3], double& dd) const
{
  double n[3];
  int max_ipct = -1;
  double max_pd = -1;
  double max_n[3] = {1,0,0};
  for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
    double n0[3];
    const double diff[3] = {p[0]-trans[0],p[1]-trans[1],p[2]-trans[2]};
    double pd0 = apCT[ipct]->Projection(diff,0,n0);
    if( ipct == 0 || pd0 > max_pd ){
      max_pd = pd0;
      max_n[0] = n0[0];
      max_n[1] = n0[1];
      max_n[2] = n0[2];
      max_ipct = ipct;
    }
  }
  n[0] = max_n[0];
  n[1] = max_n[1];       
  n[2] = max_n[2];
  //////
  unsigned int nparam=0;
  for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
    unsigned int nparam0 = apCT[ipct]->NParam();
    if( iparam_s >= nparam && iparam_s < nparam+nparam0 ){
      if( ipct == max_ipct ){
        apCT[ipct]->ProjectionDerivative(p,ino0,iparam_s-nparam,dn,dd);
        return;
      }
      else{
        break;
      }
    }
    nparam += nparam0;   
  }
  dn[0] = 0;
  dn[1] = 0;
  dn[2] = 0;
  dd = 0;
}

void CContactTarget3D_Array::Draw(unsigned int idraw) const
{ 

}