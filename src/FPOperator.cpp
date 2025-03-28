#include "FPOperator.hpp"

using namespace mfem;
using namespace fokker;

FPOperator::FPOperator(int ndofs) : TimeDependentOperator(ndofs) {}