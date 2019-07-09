<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Local Outlier Factor
 *
 * Local Outlier Factor (LOF) measures the local deviation of density
 * of a given sample with respect to its k nearest neighbors. As such,
 * LOF only considers the local region of a sample thus enabling it to
 * detect anomalies within individual clusters of data.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local
 * Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LocalOutlierFactor implements Estimator, Learner, Online, Ranking, Persistable
{
    protected const DEFAULT_THRESHOLD = 1.5;
    
    /**
     * The number of nearest neighbors to consider a local region.
     *
     * @var int
     */
    protected $k;

    /**
     * The percentage of outliers that are assumed to be present in the
     * training set.
     *
     * @var float|null
     */
    protected $contamination;

    /**
     * The distance kernel to use when computing the distances between two
     * data points.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The training samples.
     *
     * @var array[]
     */
    protected $samples = [
        //
    ];

    /**
     * The precomputed k distances between each training sample and its kth
     * nearest neighbor.
     *
     * @var float[]
     */
    protected $kdistances = [
        //
    ];
 
    /**
     * The precomputed local reachability densities of the training set.
     *
     * @var float[]
     */
    protected $lrds = [
        //
    ];

    /**
     * The local outlier factor threshold used by the decision function.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * @param int $k
     * @param float|null $contamination
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 20, ?float $contamination = null, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to form a local region, $k given.");
        }

        if (isset($contamination) and ($contamination < 0. or $contamination > 0.5)) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->k = $k;
        $this->contamination = $contamination;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::ANOMALY_DETECTOR;
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
        return $this->threshold and $this->samples;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        $this->samples = $this->kdistances = $this->lrds = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->samples = array_merge($this->samples, $dataset->samples());

        $distances = array_map([self::class, 'localRegion'], $this->samples);

        $this->kdistances = array_map('end', $distances);

        $this->lrds = array_map([self::class, 'localReachabilityDensity'], $distances);

        if ($this->contamination) {
            $lofs = array_map([self::class, 'localOutlierFactor'], $this->samples);

            $threshold = Stats::percentile($lofs, 100. - (100. * $this->contamination));
        } else {
            $threshold = self::DEFAULT_THRESHOLD;
        }

        $this->threshold = $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function rank(Dataset $dataset) : array
    {
        if (empty($this->samples)) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }
        
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'localOutlierFactor'], $dataset->samples());
    }

    /**
     * Calculate the local outlier factor of a given sample.
     *
     * @param array $sample
     * @throws \RuntimeException
     * @return float
     */
    protected function localOutlierFactor(array $sample) : float
    {
        if (empty($this->lrds)) {
            throw new RuntimeException('Local reachability distances have'
                . ' not been computed, must train estimator first.');
        }

        $distances = $this->localRegion($sample);

        $lrd = $this->localReachabilityDensity($distances);

        $lrds = array_intersect_key($this->lrds, $distances);

        $ratios = [];

        foreach ($lrds as $lHat) {
            $ratios[] = $lHat / $lrd;
        }

        return Stats::mean($ratios);
    }

    /**
     * Calculate the local reachability density of a sample given its
     * distances to its k nearest neighbors.
     *
     * @param array $distances
     * @throws \RuntimeException
     * @return float
     */
    protected function localReachabilityDensity(array $distances) : float
    {
        if (empty($this->kdistances)) {
            throw new RuntimeException('K distances have not been computed,'
                . ' must train estimator first.');
        }

        $kdistances = array_intersect_key($this->kdistances, $distances);

        $rds = array_map('max', $distances, $kdistances);

        return 1. / (Stats::mean($rds) ?: EPSILON);
    }

    /**
     * Find the K nearest neighbors to the given sample vector using the
     * brute force method.
     *
     * @param array $sample
     * @throws \RuntimeException
     * @return array
     */
    protected function localRegion(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_slice($distances, 0, $this->k, true);
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->threshold ? 1 : 0;
    }
}
