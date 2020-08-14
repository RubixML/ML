<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

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
class GaussianRandomProjector implements Transformer, Stateful, Stringable
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
     * @var \Tensor\Matrix|null
     */
    protected $r;

    /**
     * Estimate the minimum dimensionality needed to satisfy a *max distortion* constraint with *n*
     * samples using the Johnson-Lindenstrauss lemma.
     *
     * @param int $n
     * @param float $maxDistortion
     * @throws \InvalidArgumentException
     * @return int
     */
    public static function minDimensions(int $n, float $maxDistortion = 0.5) : int
    {
        if ($n < 0) {
            throw new InvalidArgumentException('Number of samples'
                . " must be be greater than 0, $n given.");
        }

        if ($maxDistortion <= 0.0) {
            throw new InvalidArgumentException('Max distortion must be'
                . " greater than 0, $maxDistortion given.");
        }

        $denominator = $maxDistortion ** 2 / 2.0 - $maxDistortion ** 3 / 3.0;

        return (int) round(4.0 * log($n) / $denominator);
    }

    /**
     * @param int $dimensions
     * @throws \InvalidArgumentException
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        $this->dimensions = $dimensions;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->r);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->r = Matrix::gaussian($dataset->numColumns(), $this->dimensions);
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->r) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = Matrix::quick($samples)
            ->matmul($this->r)
            ->asArray();
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Gaussian Random Projector (dimensions: {$this->dimensions})";
    }
}
