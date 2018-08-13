<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * Mean Shift
 *
 * A hierarchical clustering algorithm that uses peak finding to locate the
 * local maxima (centroids) of a training set given by a radius constraint.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanShift implements Estimator, Persistable
{
    /**
     * The bandwidth of the radial basis function kernel. i.e. The maximum
     * distance between two points to be considered neighbors.
     *
     * @var float
     */
    protected $radius;

    /**
     * The distance kernel to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The minimum change in the centroids necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var array
     */
    protected $centroids = [
        //
    ];

    /**
     * The amount of centroid shift during each epoch of training.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  float  $radius
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @param  float  $minChange
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius, Distance $kernel = null, float $minChange = 1e-4,
                                int $epochs = PHP_INT_MAX)
    {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be greater than'
                . ' 0');
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Threshold cannot be set to less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->radius = $radius;
        $this->kernel = $kernel;
        $this->minChange = $minChange;
        $this->epochs = $epochs;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return array
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the amount of centroid shift at each epoch of training.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->centroids = $previous = $dataset->samples();

        $denominator = 2.0 * $this->radius ** 2;

        $this->steps = [];

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($this->centroids as $i => &$centroid1) {
                foreach ($dataset as $sample) {
                    $distance = $this->kernel->compute($sample, $centroid1);

                    if ($distance < $this->radius) {
                        foreach ($sample as $column => $feature) {
                            $weight = exp(-($distance ** 2 / $denominator));

                            $centroid1[$column] = ($weight * $feature) / $weight;
                        }
                    }
                }

                foreach ($this->centroids as $j => $centroid2) {
                    if ($i === $j) {
                        continue 1;
                    }

                    $distance = $this->kernel->compute($centroid1, $centroid2);

                    if ($distance < $this->radius) {
                        unset($this->centroids[$j]);
                    }
                }
            }

            $shift = $this->calculateCentroidShift($previous);

            $this->steps[] = $shift;

            if ($shift < $this->minChange) {
                break 1;
            }

            $previous = $this->centroids;
        }

        $this->centroids = array_values($this->centroids);
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->assignLabel($sample);
        }

        return $predictions;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @return int
     */
    protected function assignLabel(array $sample) : int
    {
        $bestDistance = INF;
        $bestLabel = -1;

        foreach ($this->centroids as $label => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $bestDistance) {
                $bestDistance = $distance;
                $bestLabel = $label;
            }
        }

        return (int) $bestLabel;
    }

    /**
     * Calculate the magnitude (l1) of a centroid shift from the previous epoch.
     *
     * @param  array  $previous
     * @return float
     */
    protected function calculateCentroidShift(array $previous) : float
    {
        $shift = 0.0;

        foreach ($this->centroids as $i => $centroid) {
            foreach ($centroid as $j => $mean) {
                $shift += abs($previous[$i][$j] - $mean);
            }
        }

        return $shift;
    }
}
