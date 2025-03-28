#pragma once

#include <utility>
#include "mfem.hpp"

namespace mfem
{
    /// Matrix coefficient defined as the gradient of a vector GridFunction
    class GradientVectorGridFunctionCoefficient : public MatrixCoefficient
    {
    protected:
        const GridFunction *GridFunc;

    public:
        GradientVectorGridFunctionCoefficient(const GridFunction *gf)
            : MatrixCoefficient(gf ? gf->VectorDim() : 0), GridFunc(gf) {}

        /// Set the vector grid function.
        void SetGridFunction(const GridFunction *gf)
        {
            GridFunc = gf;
            height = width = gf ? gf->VectorDim() : 0;
        }

        /// Get the vector GridFunction.
        const GridFunction *GetGridFunction() const { return GridFunc; }

        /// Evaluate the matrix gradient coefficient at @a ip.
        virtual void Eval(DenseMatrix &K,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) override;

        virtual ~GradientVectorGridFunctionCoefficient() {}
    };

    /// Scalar Coefficient defined as one entry of a MatrixCoefficient
    class MatrixEntryCoefficient : public Coefficient
    {
    protected:
        MatrixCoefficient *coeff;
        int row, col;

    public:
        MatrixEntryCoefficient(MatrixCoefficient *M, int i, int j)
            : Coefficient(), coeff(M), row(i), col(j) {}

        /// Set the matrix coefficient.
        void SetMatrixCoefficient(MatrixCoefficient *M) { coeff = M; }

        /// Set the matrix indices.
        void SetIndices(int i, int j)
        {
            row = i;
            col = j;
        }

        /// Get the matrix coefficient.
        const MatrixCoefficient *GetMatrixCoefficient() const { return coeff; }

        /// Get the matrix indices.
        std::pair<int, int> GetIndices() const { return {row, col}; }

        /// Evaluate the matrix entry coefficient at @a ip.
        virtual double Eval(ElementTransformation &T,
                            const IntegrationPoint &ip) override;

        virtual ~MatrixEntryCoefficient() {}
    };

    /// VectorCoefficient defined as divergence of a MatrixCoefficient
    class DivergenceMatrixGridFunctionCoefficient : public VectorCoefficient
    {
    protected:
        Array2D<ParGridFunction *> GridFunc;

    public:
        DivergenceMatrixGridFunctionCoefficient(Array2D<ParGridFunction *> gf)
            : VectorCoefficient(gf.NumRows()), GridFunc(gf) {}

        /// Set the matrix of GridFunctions.
        void SetMatrixCoefficient(Array2D<ParGridFunction *> gf)
        {
            MFEM_ASSERT(vdim == gf.NumRows(), "Dimensions do not match.");
            GridFunc = gf;
        }

        /// Get the matrix of GridFunctions.
        const Array2D<ParGridFunction *> GetMatrixCoefficient() const { return GridFunc; }

        /// Evaluate the matrix divergence coefficient at @a ip.
        virtual void Eval(Vector &v,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) override;

        virtual ~DivergenceMatrixGridFunctionCoefficient() {}
    };
}