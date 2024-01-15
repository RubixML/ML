<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function max;

/**
 * Vantage Point
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VantagePoint implements Hypersphere, HasBinaryChildren
{
    use HasBinaryChildrenTrait;

    /**
     * The center or multivariate mean of the centroid.
     *
     * @var list<string|int|float>
     */
    protected $center;

    /**
     * The radius of the centroid.
     *
     * @var float
     */
    protected $radius;

    /**
     * The left and right splits of the training data.
     *
     * @var array{Labeled,Labeled}|null
     */
    protected $subsets;

    /**
     * Factory method to build a hypersphere by splitting the dataset into left and right clusters.
     *
     * @param Labeled $dataset
     * @param Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
    {
        $center = [];

        foreach ($dataset->features() as $column => $values) {
            if ($dataset->featureType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $threshold = Stats::median($distances);

        $samples = $dataset->samples();
        $labels = $dataset->labels();

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        foreach ($distances as $i => $distance) {
            if ($distance <= $threshold) {
                $leftSamples[] = $samples[$i];
                $leftLabels[] = $labels[$i];
            } else {
                $rightSamples[] = $samples[$i];
                $rightLabels[] = $labels[$i];
            }
        }

        $radius = max($distances) ?: 0.0;

        return new self($center, $radius, [
            Labeled::quick($leftSamples, $leftLabels),
            Labeled::quick($rightSamples, $rightLabels),
        ]);
    }

    /**
     * @param list<string|int|float> $center
     * @param float $radius
     * @param array{Labeled,Labeled} $subsets
     */
    public function __construct(array $center, float $radius, array $subsets)
    {
        $this->center = $center;
        $this->radius = $radius;
        $this->subsets = $subsets;
    }

    /**
     * Return the center vector.
     *
     * @return list<string|int|float>
     */
    public function center() : array
    {
        return $this->center;
    }

    /**
     * Return the radius of the centroid.
     *
     * @return float
     */
    public function radius() : float
    {
        return $this->radius;
    }

    /**
     * Return the left and right subsets of the training data.
     *
     * @throws RuntimeException
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function subsets() : array
    {
        if (!isset($this->subsets)) {
            throw new RuntimeException('Subsets property does not exist.');
        }

        return $this->subsets;
    }

    /**
     * Does the hypersphere reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool
    {
        return $this->radius === 0.0;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->subsets);
    }
}
