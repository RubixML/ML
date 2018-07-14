<?php

namespace Rubix\ML\Clusterers;

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
class MeanShift implements Clusterer, Persistable
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
        if ($radius <= 0) {
            throw new InvalidArgumentException('Radius must be greater than'
                . ' 0');
        }

        if ($minChange < 0) {
            throw new InvalidArgumentException('Threshold cannot be set to less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($kernel)) {
            $kernel = new Euclidean();
        }

        $this->radius = $radius;
        $this->kernel = $kernel;
        $this->minChange = $minChange;
        $this->epochs = $epochs;
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->centroids = $previous = $dataset->samples();

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($this->centroids as $i => &$centroid1) {
                foreach ($dataset as $sample) {
                    $distance = $this->kernel->compute($sample, $centroid1);

                    if ($distance <= $this->radius) {
                        foreach ($sample as $column => $feature) {
                            $weight = exp(-(($distance ** 2)
                                / (2 * $this->radius ** 2)));

                            $centroid1[$column] = ($weight * $feature)
                                / $weight;
                        }
                    }
                }

                foreach ($this->centroids as $j => $centroid2) {
                    if ($i !== $j) {
                        $distance = $this->kernel->compute($centroid1, $centroid2);

                        if ($distance < $this->radius) {
                            unset($this->centroids[$j]);
                        }
                    }
                }
            }

            $shift = $this->calculateShift($previous);

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
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->label($sample);
        }

        return $predictions;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @return int
     */
    protected function label(array $sample) : int
    {
        $best = ['distance' => INF, 'label' => -1];

        foreach ($this->centroids as $label => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best['distance'] = $distance;
                $best['label'] = (int) $label;
            }
        }

        return $best['label'];
    }

    /**
     * Calculate the magnitude of a centroid shift from the previous epoch.
     *
     * @param  array  $previous
     * @return float
     */
    protected function calculateShift(array $previous) : float
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
