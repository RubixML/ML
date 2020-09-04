<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;

use function count;

/**
 * Sparse Random Projector
 *
 * A *database-friendly* random projector that samples its random projection matrix from a
 * sparse probabilistic approximation of the Gaussian distribution.
 *
 * References:
 * [1] D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss
 * with binary coins.
 * [2] P. Li at al. (2006). Very Sparse Random Projections.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SparseRandomProjector extends GaussianRandomProjector
{
    /**
     * The amount of sparsity in the random projection matrix.
     *
     * @var float
     */
    protected $sparsity;

    /**
     * @param int $dimensions
     * @param float $sparsity
     * @throws \InvalidArgumentException
     */
    public function __construct(int $dimensions, float $sparsity = 3.0)
    {
        if ($sparsity < 1.0) {
            throw new InvalidArgumentException('Sparsity must be'
                . " greater than 1, $sparsity given.");
        }

        parent::__construct($dimensions);

        $this->sparsity = $sparsity;
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

        $n = $dataset->numColumns();

        $max = getrandmax();

        $distribution = [
            [sqrt($this->sparsity), 1.0 / (2.0 * $this->sparsity)],
            [0.0, 1.0 - (1.0 / $this->sparsity)],
            [-sqrt($this->sparsity), 1.0 / (2.0 * $this->sparsity)],
        ];

        $r = [];

        while (count($r) < $n) {
            $row = [];

            while (count($row) < $this->dimensions) {
                $delta = rand() / $max;

                foreach ($distribution as [$value, $probability]) {
                    $delta -= $probability;

                    if ($delta <= 0.0) {
                        $row[] = $value;

                        break 1;
                    }
                }
            }

            $r[] = $row;
        }

        $this->r = Matrix::quick($r);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Sparse Random Projector (dimensions: {$this->dimensions}, sparsity: {$this->sparsity})";
    }
}
