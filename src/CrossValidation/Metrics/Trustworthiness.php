<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Symmetric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

use function max;
use function count;
use function asort;
use function is_infinite;

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
     * The number of nearest neighbors considered when scoring local
     * structure preservation.
     *
     * @var int
     */
    protected int $k;

    /**
     * The symmetric distance kernel used to measure dissimilarity in the input
     * space. Symmetry allows the pairwise distance matrix to be computed by
     * only evaluating the upper triangle and mirroring the result.
     *
     * @var Symmetric
     */
    protected Symmetric $kernel;

    /**
     * @param int $k
     * @param ?Symmetric $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 5, ?Symmetric $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Number of neighbors must be'
                . " greater than 0, $k given.");
        }

        $this->k = $k;
        $this->kernel = $kernel ?? new Euclidean();
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

        $ranks = $this->inputRanks($samples);

        $distances = $this->pairwiseDistances($embedded, [$this, 'euclidean']);

        $total = 0.0;

        for ($i = 0; $i < $n; ++$i) {
            foreach ($this->kNearest($distances, $i, $this->k) as $j) {
                $total += max(0, $ranks[$i][$j] - $this->k);
            }
        }

        $denominator = $n * $this->k * (2 * $n - 3 * $this->k - 1);

        return 1.0 - (2.0 * $total) / $denominator;
    }

    /**
     * For each sample, assign a 1-based rank (1 = closest) to every other
     * sample by input-space distance.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @return array<int, array<int,int>>
     */
    protected function inputRanks(array $samples) : array
    {
        $n = count($samples);

        $distances = $this->pairwiseDistances($samples);

        $ranks = [];

        for ($i = 0; $i < $n; ++$i) {
            $row = $distances[$i];

            asort($row);

            $rank = 1;

            foreach ($row as $j => $_distance) {
                if (is_infinite($_distance)) {
                    break;
                }

                $ranks[$i][$j] = $rank;

                ++$rank;
            }
        }

        return $ranks;
    }

    /**
     * Compute the full n x n pairwise distance matrix between every pair of
     * sample points, seeding the diagonal so that each point sorts last and can
     * never be selected as its own neighbor.
     *
     * Only the upper triangle is computed and mirrored, halving the number of
     * distance evaluations.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @return array<int, array<int, float>>
     */
    protected function pairwiseDistances(array $samples) : array
    {
        $n = count($samples);

        $matrix = [];

        for ($i = 0; $i < $n; ++$i) {
            $matrix[$i][$i] = INF;
        }

        for ($i = 0; $i < $n; ++$i) {
            for ($j = $i + 1; $j < $n; ++$j) {
                $distance = $this->kernel->compute($samples[$i], $samples[$j]);

                $matrix[$i][$j] = $distance;
                $matrix[$j][$i] = $distance;
            }
        }

        return $matrix;
    }

    /**
     * Return the indices of the k closest points to point $i, excluding $i
     * itself (whose distance is seeded high enough to sort last).
     *
     * @internal
     *
     * @param array<int, array<int, float>> $matrix
     * @param int $i
     * @param int $k
     * @return list<int>
     */
    protected function kNearest(array $matrix, int $i, int $k) : array
    {
        $row = $matrix[$i];

        asort($row);

        $neighbors = [];

        foreach ($row as $j => $distance) {
            if (is_infinite($distance)) {
                break;
            }

            $neighbors[] = $j;

            if (count($neighbors) === $k) {
                break;
            }
        }

        return $neighbors;
    }

    /**
     * Return the squared Euclidean distance between two points.
     *
     * @internal
     *
     * @param list<float> $a
     * @param list<float> $b
     * @return float
     */
    protected function euclidean(array $a, array $b) : float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $delta = $value - $b[$i];

            $distance += $delta * $delta;
        }

        return $distance;
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
