<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function is_null;

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
    use AutotrackRevisions;

    /**
     * The decimal representation of the fraction two thirds.
     *
     * @var float
     */
    protected const TWO_THIRDS = 2.0 / 3.0;

    /**
     * The proportion of zero to non-zero elements in the random projection matrix.
     *
     * @var float|null
     */
    protected ?float $sparsity;

    /**
     * @param int $dimensions
     * @param float|null $sparsity
     * @throws InvalidArgumentException
     */
    public function __construct(int $dimensions, ?float $sparsity = self::TWO_THIRDS)
    {
        if ($sparsity < 0.0 or $sparsity > 1.0) {
            throw new InvalidArgumentException('Sparsity must be'
                . " between 0 and 1, $sparsity given.");
        }

        parent::__construct($dimensions);

        $this->sparsity = $sparsity;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $n = $dataset->numFeatures();

        if (is_null($this->sparsity)) {
            $density = 1.0 / (1.0 + sqrt($n));
        } else {
            $density = 1.0 - $this->sparsity;
        }

        $dHat = sqrt(1.0 / $density);

        $distribution = [
            [-$dHat, 0.5 * $density],
            [0.0, 1.0 - $density],
            [$dHat, 0.5 * $density],
        ];

        $max = getrandmax();

        $r = [];

        while (count($r) < $n) {
            $row = [];

            while (count($row) < $this->dimensions) {
                $delta = rand() / $max;

                foreach ($distribution as [$value, $probability]) {
                    $delta -= $probability;

                    if ($delta <= 0.0) {
                        $row[] = $value;

                        break;
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Sparse Random Projector (dimensions: {$this->dimensions}, sparsity: {$this->sparsity})";
    }
}
