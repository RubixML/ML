<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gaussian Random Projector
 *
 * A Random Projector is a dimensionality reducer based on the
 * Johnson-Lindenstrauss lemma that uses a random matrix to project a feature
 * vector onto a user-specified number of dimensions. It is faster than most
 * non-randomized dimensionality reduction techniques and offers similar
 * performance. This version uses a random matrix sampled from a Gaussian
 * distribution.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianRandomProjector implements Transformer, Stateful
{
    /**
     * The target number of dimensions.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The random matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
    */
    protected $r;

    /**
     * Calculate the minimum number of dimensions for n total samples with a
     * given maximum distortion using the Johnson-Lindenstrauss lemma.
     *
     * @param  int  $n
     * @param  float  $maxDistortion
     * @return int
     */
    public static function minDimensions(int $n, float $maxDistortion = 0.1) : int
    {
        return (int) round(4. * log($n)
            / ($maxDistortion ** 2 / 2. - $maxDistortion ** 3 / 3.));
    }

    /**
     * @param  int  $dimensions
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot project onto less than'
                . " 1 dimension, $dimensions given.");
        }

        $this->dimensions = $dimensions;
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
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $this->r = Matrix::gaussian($dataset->numColumns(), $this->dimensions);
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->r)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = Matrix::quick($samples)->matmul($this->r)->asArray();
    }
}
