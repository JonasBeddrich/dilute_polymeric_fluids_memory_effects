#pragma once

#include "FokkerPlanckSettings.hpp"
#include "mfem.hpp"

namespace mfem {
namespace fokker {

class FPOperator : public TimeDependentOperator {
 public:
  FPOperator(int ndofs);

  virtual void SetParameters(VectorCoefficient &space_adv_coeff,
                             MatrixArrayCoefficient &conf_adv_coeff) {}
  
  virtual void GeneralizedImplicitSolve(const double dtn, 
                                        const Vector &phin, 
                                        const Vector &inFn1, 
                                        Vector  &dphi_dt) {} 

  virtual ~FPOperator() {}
};
}  // namespace fokker
}  // namespace mfem