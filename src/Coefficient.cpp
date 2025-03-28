#include "Coefficient.hpp"

using namespace mfem;

ElementTransformation *RefinedToCoarse(Mesh &coarse_mesh,
                                       const ElementTransformation &T,
                                       const IntegrationPoint &ip,
                                       IntegrationPoint &coarse_ip) {
  Mesh &fine_mesh = *T.mesh;
  // Get the element transformation of the coarse element containing the
  // fine element.
  int fine_element = T.ElementNo;
  const CoarseFineTransformations &cf = fine_mesh.GetRefinementTransforms();
  int coarse_element = cf.embeddings[fine_element].parent;
  ElementTransformation *coarse_T =
      coarse_mesh.GetElementTransformation(coarse_element);
  // Transform the integration point from fine element coordinates to coarse
  // element coordinates.
  Geometry::Type geom = T.GetGeometryType();
  IntegrationPointTransformation fine_to_coarse;
  IsoparametricTransformation &emb_tr = fine_to_coarse.Transf;
  emb_tr.SetIdentityTransformation(geom);
  emb_tr.SetPointMat(
      cf.point_matrices[geom](cf.embeddings[fine_element].matrix));
  fine_to_coarse.Transform(ip, coarse_ip);
  coarse_T->SetIntPoint(&coarse_ip);
  return coarse_T;
}

void GradientVectorGridFunctionCoefficient::Eval(DenseMatrix &K,
                                                 ElementTransformation &T,
                                                 const IntegrationPoint &ip) {
  Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
  if (T.mesh->GetNE() == gf_mesh->GetNE()) {
    GridFunc->GetVectorGradient(T, K);
  } else {
    IntegrationPoint coarse_ip;
    ElementTransformation *coarse_T =
        RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
    GridFunc->GetVectorGradient(*coarse_T, K);
  }
}

double MatrixEntryCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip) {
  DenseMatrix K;
  coeff->Eval(K, T, ip);
  return K(row, col);
}

void DivergenceMatrixGridFunctionCoefficient::Eval(Vector &v,
                                                   ElementTransformation &T,
                                                   const IntegrationPoint &ip) {
  Mesh *gf_mesh = GridFunc[0][0]->FESpace()->GetMesh();
  const int dim = GridFunc.NumRows();
  v.SetSize(dim);
  v = 0.0;

  Vector tmp;
  bool equal = T.mesh->GetNE() == gf_mesh->GetNE();
  IntegrationPoint coarse_ip;
  ElementTransformation *actual_T =
      equal ? &T : RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < GridFunc.NumCols(); ++j) {
      GridFunc[i][j]->GetGradient(*actual_T, tmp);
      v[i] += tmp[j];
    }
  }
}
