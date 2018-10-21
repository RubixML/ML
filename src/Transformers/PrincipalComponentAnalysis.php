<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;
use RuntimeException;

/**
 * Principal Component Analyis
 *
 * Principal Component Analysis or *PCA* is a dimensionality reduction technique that
 * aims to transform the feature space by the *k* principal components that explain
 * the most variance of the data where *k* is the dimensionality of the output
 * specified by the user. PCA is used to compress high dimensional samples down to
 * lower dimensions such that would retain as much of the information within the data
 * as possible.
 * 
 * References:
 * [1] H. Abdi et al. (2010). Principal Component Analysis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PrincipalComponentAnalysis implements Transformer, Stateful
{
    /**
     * The target number of dimensions to project onto.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The eigenvalues computed during fitting.
     * 
     * @var array|null
     */
    protected $eigenvalues;

    /**
     * The matrix of eigenvectors computed at fitting.
     *
     * @var \Rubix\Tensor\Matrix|null
    */
    protected $eigenvectors;

    /**
     * The amount of variance that is preserved by the transformation.
     * 
     * @var int|float|null
     */
    protected $explainedVar;

    /**
     * The centers (means) of the input feature columns.
     * 
     * @var array|null
     */
    protected $means;

    /**
     * @param  int  $dimensions
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot project onto less than'
                . ' 1 dimension.');
        }

        $this->dimensions = $dimensions;
    }

    /**
     * Return the fraction of explained variance that has been preserved by
     * the transformation.
     * 
     * @return int|float|null
     */
    public function explainedVar()
    {
        return $this->explainedVar;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $xT = Matrix::build($dataset->samples())->transpose();

        list($eigenvalues, $eigenvectors) = $xT->covariance()->eig();

        $total = array_sum($eigenvalues);

        $eigenvectors = $eigenvectors->asArray();
        
        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        $eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
        $eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);

        $eigenvectors = Matrix::quick($eigenvectors)
            ->transpose();

        $this->eigenvalues = $eigenvalues;
        $this->eigenvectors = $eigenvectors;

        $this->explainedVar = array_sum($eigenvalues)
            / ($total ?: self::EPSILON);

        $this->means = $xT->mean()->transpose();
    }

    /**
     * Transform the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->means) or is_null($this->eigenvectors)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = Matrix::build($samples)
            ->subtract($this->means)
            ->matmul($this->eigenvectors)
            ->asArray();
    }
}
