<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\SquaredEuclidean;
use Rubix\ML\Kernels\Distance\Symmetric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

use function max;
use function count;
use function asort;
use function array_keys;
use function array_slice;

/**
 * Trustworthiness
 *
 * Trustworthiness is a manifold-learning quality metric that measures how well
 * a low-dimensional embedding preserves the local neighborhood structure of
 * the original feature space. For every sample, its nearest neighbors in the
 * embedded space are inspected; any that were far relatives in the input space
 * are penalized in proportion to how far they actually are. Perfect preservation
 * of local structure yields a score of 1.0; a degraded embedding lowers the
 * score.
 *
 * References:
 * [1] J. Venna and S. Kaski. (2001). Neighborhood Preservation in Nonlinear
 *     Projection Methods: An Experimental Study.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Trustworthiness implements Stringable
{
    /**
     * The number of nearest neighbors considered when scoring local structure preservation.
     *
     * @var int
     */
    protected int $k;

    /**
     * The symmetric distance kernel used to measure dissimilarity in the input space.
     *
     * @var Symmetric
     */
    protected Symmetric $inputKernel;

    /**
     * The symmetric distance kernel used to measure dissimilarity in the embedded
     * space.
     *
     * @var Symmetric
     */
    protected Symmetric $embeddingKernel;

    /**
     * @param int $k
     * @param ?Symmetric $inputKernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 5, ?Symmetric $inputKernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Number of neighbors must be'
                . " greater than 0, $k given.");
        }

        $this->k = $k;
        $this->inputKernel = $inputKernel ?? new Euclidean();
        $this->embeddingKernel = new SquaredEuclidean();
    }

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return Tuple<float,float>
     */
    public function range() : Tuple
    {
        return new Tuple(0.0, 1.0);
    }

    /**
     * Trustworthiness scores point clouds (embeddings) directly rather than
     * estimator predictions, so it is compatible with no estimator type.
     *
     * @internal
     *
     * @return list<EstimatorType>
     */
    public function compatibility() : array
    {
        return [];
    }

    /**
     * Score the extent to which the embedded space retains the local
     * neighborhood structure of the original space.
     *
     * @param list<list<string|int|float>> $samples
     * @param list<list<float>> $embedded
     * @throws InvalidArgumentException
     * @return float
     */
    public function score(array $samples, array $embedded) : float
    {
        $n = count($samples);

        if ($n !== count($embedded)) {
            throw new InvalidArgumentException('Number of samples and embedded'
                . " points must match, $n samples vs " . count($embedded) . ' given.');
        }

        if ($n < 2) {
            return 1.0;
        }

        if (2 * $this->k >= $n) {
            throw new InvalidArgumentException("Number of neighbors ($this->k) must be"
                . ' less than n / 2 (' . ($n / 2) . ') or the'
                . ' Trustworthiness could fall below 0.');
        }

        $total = 0.0;

        for ($i = 0; $i < $n; ++$i) {
            $embeddedRow = $this->distancesFrom($embedded, $i, $this->embeddingKernel);

            asort($embeddedRow);

            $neighbors = array_slice(array_keys($embeddedRow), 0, $this->k);

            $inputRow = $this->distancesFrom($samples, $i, $this->inputKernel);

            foreach ($neighbors as $j) {
                $rank = 1;

                foreach ($inputRow as $l => $distance) {
                    if ($distance < $inputRow[$j] or ($distance == $inputRow[$j] and $l < $j)) {
                        ++$rank;
                    }
                }

                $total += max(0, $rank - $this->k);
            }
        }

        $denominator = $n * $this->k * (2 * $n - 3 * $this->k - 1);

        return 1.0 - (2.0 * $total) / $denominator;
    }

    /**
     * Compute the distance from sample $i to every other sample, keyed by index
     * with $i itself omitted. Because $i is absent, no INF diagonal sentinel is
     * needed — a point can never be its own neighbor — and the keys are emitted
     * in ascending index order so that a stable sort reproduces the index-order
     * tie-break.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @param int $index
     * @param Symmetric $kernel
     * @return array<int, float>
     */
    protected function distancesFrom(array $samples, int $index, Symmetric $kernel) : array
    {
        $row = [];

        $sample = $samples[$index];

        for ($i = 0; $i < count($samples); ++$i) {
            if ($i === $index) {
                continue;
            }

            $row[$i] = $kernel->compute($sample, $samples[$i]);
        }

        return $row;
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
        return 'Trustworthiness';
    }
}
