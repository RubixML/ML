<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\LoggerAware;
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
 * References:
 * [1] M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms
 * for Clustering.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanShift implements Learner, Verbose, Persistable
{
    use LoggerAware;

    /**
     * The bandwidth of the radial basis function kernel. i.e. The maximum
     * distance between two points to be considered neighbors.
     *
     * @var float
     */
    protected $radius;

    /**
     * The precomputed denominator of the weight calculation.
     *
     * @var float
     */
    protected $delta;

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
     * @param  int  $epochs
     * @param  float  $minChange
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius, ?Distance $kernel = null, int $epochs = 100,
                                float $minChange = 1e-4)
    {
        if ($radius <= 0.) {
            throw new InvalidArgumentException("Cluster radius must be"
                . " greater than 0, $radius given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException("Estimator must train for at"
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException("Minimum change cannot be less"
                . " than 0, $minChange given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->radius = $radius;
        $this->delta = 2. * $radius ** 2;
        $this->kernel = $kernel;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
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
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($this->logger) $this->logger->info('Training started');

        $this->centroids = $previous = $dataset->samples();

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($this->centroids as $i => &$centroid) {
                foreach ($dataset as $sample) {
                    $distance = $this->kernel->compute($sample, $centroid);

                    if ($distance > $this->radius) {
                        continue 1;
                    }

                    foreach ($centroid as $column => &$mean) {
                        $weight = exp(-($distance ** 2 / $this->delta));

                        $mean = ($weight * $sample[$column]) / $weight;
                    }
                }

                foreach ($this->centroids as $j => $neighbor) {
                    if ($i === $j) {
                        continue 1;
                    }

                    $distance = $this->kernel->compute($centroid, $neighbor);

                    if ($distance < $this->radius) {
                        unset($this->centroids[$j]);
                    }
                }
            }

            $shift = $this->centroidShift($previous);

            $this->steps[] = $shift;

            if ($this->logger) $this->logger->info("Epoch $epoch"
                . " completed, shift: $shift");

            if ($shift < $this->minChange) {
                break 1;
            }

            $previous = $this->centroids;
        }

        $this->centroids = array_values($this->centroids);

        if ($this->logger) $this->logger->info("Training complete");
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->assignCluster($sample);
        }

        return $predictions;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @return int
     */
    protected function assignCluster(array $sample) : int
    {
        $bestDistance = INF;
        $bestCluster = -1;

        foreach ($this->centroids as $cluster => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $bestDistance) {
                $bestDistance = $distance;
                $bestCluster = $cluster;
            }
        }

        return (int) $bestCluster;
    }

    /**
     * Calculate the magnitude (l1) of a centroid shift from the previous epoch.
     *
     * @param  array  $previous
     * @return float
     */
    protected function centroidShift(array $previous) : float
    {
        $shift = 0.;

        foreach ($this->centroids as $cluster => $centroid) {
            $prevCluster = $previous[$cluster];

            foreach ($centroid as $column => $mean) {
                $shift += abs($prevCluster[$column] - $mean);
            }
        }

        return $shift;
    }
}
