<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\Tensor\Matrix;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
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
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the centroids necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The cluster centroid seeder.
     *
     * @var \Rubix\ML\Clusterers\Seeders\Seeder|null
     */
    protected $seeder;

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
     * @param float $radius
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius,
        ?Distance $kernel = null,
        int $epochs = 100,
        float $minChange = 1e-4,
        ?Seeder $seeder = null
    ) {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Cluster radius must be'
                . " greater than 0, $radius given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->radius = $radius;
        $this->delta = 2. * $radius ** 2;
        $this->kernel = $kernel ?? new Euclidean();
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->seeder = $seeder;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->centroids);
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
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner initialized w/ '
                . Params::stringify([
                    'radius' => $this->radius,
                    'kernel' => $this->kernel,
                    'epochs' => $this->epochs,
                    'min_change' => $this->minChange,
                    'seeder' => $this->seeder,
                ]));
        }

        $samples = $dataset->samples();

        if ($this->seeder) {
            $k = (int) round(0.5 * $dataset->numRows());

            $centroids = $this->seeder->seed($dataset, $k);
        } else {
            $centroids = $samples;
        }

        $previous = $centroids;

        $this->steps = [];
 
        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($centroids as $i => &$centroid) {
                [$neighbors, $distances] = $this->range($centroid, $samples, $this->radius);

                $step = Matrix::quick($neighbors)
                    ->transpose()
                    ->mean()
                    ->asArray();

                $mu2 = Stats::mean($distances) ** 2;

                $weight = exp(-$mu2 / $this->delta);

                foreach ($centroid as $column => &$mean) {
                    $mean = ($weight * $step[$column]) / $weight;
                }

                foreach ($centroids as $j => $neighbor) {
                    if ($i === $j) {
                        continue 1;
                    }

                    $distance = $this->kernel->compute($centroid, $neighbor);

                    if ($distance < $this->radius) {
                        unset($centroids[$j]);
                    }
                }
            }

            $shift = $this->centroidShift($centroids, $previous);

            $this->steps[] = $shift;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch complete, shift=$shift");
            }

            if (is_nan($shift)) {
                break 1;
            }

            if ($shift < $this->minChange) {
                break 1;
            }

            $previous = $centroids;
        }

        $this->centroids = array_values($centroids);

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'assign'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param array $sample
     * @return int
     */
    protected function assign(array $sample) : int
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
     * The the neighbors within range of a sample.
     *
     * @param array $sample
     * @param array $samples
     * @param float $radius
     * @return array
     */
    protected function range(array $sample, array $samples, float $radius) : array
    {
        $neighbors = $distances = [];

        foreach ($samples as $neighbor) {
            $distance = $this->kernel->compute($sample, $neighbor);

            if ($distance <= $this->radius) {
                $neighbors[] = $neighbor;
                $distances[] = $distance;
            }
        }

        return [$neighbors, $distances];
    }

    /**
     * Calculate the magnitude (l1) of centroid shift from the previous epoch.
     *
     * @param array $current
     * @param array $previous
     * @return float
     */
    protected function centroidShift(array $current, array $previous) : float
    {
        $shift = 0.;

        foreach ($current as $cluster => $centroid) {
            $prevCentroid = $previous[$cluster];

            foreach ($centroid as $column => $mean) {
                $shift += abs($prevCentroid[$column] - $mean);
            }
        }

        return $shift;
    }
}
