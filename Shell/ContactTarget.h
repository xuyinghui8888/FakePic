#if !defined(CONTACT_TARGET_H)
#define CONTACT_TARGET_H

#include <string>
#include <vector>
#include <assert.h>
#include "SpatialHashGrid3d.h"


namespace SDF 
{


class CContactTarget3D
{
public:
  virtual ~CContactTarget3D(){}
	virtual void Draw(unsigned int idraw=0) const = 0;
	virtual double Projection(const double p[3], unsigned int ino, double n[3]) const = 0;
  virtual void ProjectionDerivative(const double p[3], unsigned int ino, unsigned int iparam_s, double dn[3], double& dd) const = 0;
  virtual bool IntersectionPoint(double p[3],  const double org[3], const double dir[3]) const = 0;
  virtual void GetMesh(std::vector<unsigned int>& aTri,
                       std::vector<double>& aXYZ,
                       double elen) const = 0;
  
  ////
  virtual unsigned int NParam() const = 0;
  virtual std::vector<double> GetParamAry() const = 0;
  virtual void SetParameter(unsigned int idim, double param) = 0;
  virtual void SetParameterAry(const std::vector<double>& ap){
    unsigned int np=this->NParam();
    for(unsigned int ip=0;ip<np;ip++){ SetParameter(ip,ap[ip]); }
  }
  virtual void PrecomputeSensitivity(int iparam) = 0;
  virtual void PrecomputeSpacialHash(unsigned int nNode) = 0;
  virtual void GetParamRefWidthName(std::vector<double>& param_ref, std::vector<double>& param_width, std::vector<std::string>& param_name) const = 0;
  virtual bool IsParamGood() const = 0;
public:
  double Projection(double px, double py, double pz, double n[3]) const
  {
    const double p0[3] = {px,py,pz};
    return this->Projection(p0,0,n);
  }
};

class CContactTarget3D_Array : public CContactTarget3D
{
public:
  CContactTarget3D_Array(){
    trans[0] = 0; 
    trans[1] = 0;        
    trans[2] = 0; 
  }
  ~CContactTarget3D_Array()
  {
    for(unsigned int ipct=0;ipct<apCT.size();ipct++){ delete apCT[ipct]; }
    apCT.clear();
  }

	virtual void Draw(unsigned int idraw) const;
	virtual double Projection(const double p[3], unsigned int ino, double n[3]) const;
  virtual void ProjectionDerivative(const double p[3], unsigned int ino, unsigned int iparam_s, double dn[3], double& dd) const;
  virtual bool IntersectionPoint
  (double p[3], 
  const double org[3], const double dir[3]) const{ return false; }
  virtual void GetMesh(std::vector<unsigned int>& aTri,
                       std::vector<double>& aXYZ,
                       double elen) const {};
  virtual unsigned int NParam() const{
    unsigned int np = 0;
    for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
      np += apCT[ipct]->NParam();
    }
    return np;
  }
  virtual void PrecomputeSensitivity(int iparam){}
  virtual void PrecomputeSpacialHash(unsigned int nNode){}
  ////
  std::vector<double> GetParamAry() const{
    std::vector<double> aParam;
    for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
      std::vector<double> ap0 = apCT[ipct]->GetParamAry();
      for(unsigned int ip=0;ip<ap0.size();ip++){ aParam.push_back(ap0[ip]); }
    }
    return aParam;
  }
  void SetParameter(unsigned int idim, double param){
    unsigned int n=0;
    for(unsigned int ipct=0;ipct<apCT.size();ipct++){ 
      unsigned int n0 = apCT[ipct]->NParam();
      if( idim >= n && idim < n+n0 ){
        apCT[ipct]->SetParameter(idim-n,param);
      }
      n += n0;
    }
  }
  void GetParamRefWidthName(std::vector<double>& param_ref, 
                            std::vector<double>& param_width,
                            std::vector<std::string>& param_name) const 
  {  
    param_ref.clear();
    param_width.clear();
    param_name.clear();
    for(unsigned int ipct=0;ipct<apCT.size();ipct++){       
      std::vector<double> r0,w0;
      std::vector<std::string> s0;
      apCT[ipct]->GetParamRefWidthName(r0,w0,s0);
      assert( r0.size() == w0.size() );
      assert( s0.size() == r0.size() );
      for(unsigned int ip=0;ip<r0.size();ip++){ 
        param_ref.push_back(r0[ip]);
        param_width.push_back(w0[ip]);
        param_name.push_back(s0[ip]);
      }
    }
  }
  bool IsParamGood() const { return true; }
public:
  std::vector<CContactTarget3D*> apCT;
  double trans[3];
};


}




#endif
