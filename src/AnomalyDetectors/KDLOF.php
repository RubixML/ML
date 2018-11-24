<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\KDTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d LOF
 *
 * A K-d tree accelerated version of Local Outlier Factor (LOF). Unlike brute
 * force LOF, this estimator cannot be partially trained.
 * 
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local
 * Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDLOF extends KDTree implements Learner, Persistable
{
    const THRESHOLD = 1.5;

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
     * @var float
     */
    protected $contamination;

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
     * The local outlier factor offset used by the decision function.
     * 
     * @var float|null
     */
    protected $offset;

    /**
     * @param  int  $k
     * @param  float  $contamination
     * @param  int  $maxLeafSize
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 20, float $contamination = 0.1, int $maxLeafSize = 20,
                                ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to form a local region, $k given.");
        }

        if ($contamination < 0.) {
            throw new InvalidArgumentException('Contamination cannot be less'
                . " than 0, $contamination given.");
        }

        if ($k > $maxLeafSize) {
            throw new InvalidArgumentException('K cannot be larger than the max'
                . " leaf size, $k given but $maxLeafSize allowed.");
        }

        $this->k = $k;
        $this->contamination = $contamination;

        parent::__construct($maxLeafSize, $kernel);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::DETECTOR;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This estimator only works'
                . ' with continuous features.');
        }

        $this->kdistances = $this->lrds = [];

        $this->grow($dataset);

        $distances = [];

        foreach ($dataset as $sample) {
            list($dHat) = $this->neighbors($sample, $this->k);

            $distances[] = $dHat;
            $this->kdistances[] = end($dHat);
        }

        foreach ($distances as $row) {
            $this->lrds[] = $this->localReachabilityDensity($row);
        }

        $lofs = [];

        foreach ($dataset as $sample) {
            $lofs[] = $this->localOutlierFactor($sample);
        }

        $shift = Stats::percentile($lofs, 100. * $this->contamination);
        
        $this->offset = self::THRESHOLD + $shift;
    }

    /**
     * Make predictions from a dataset.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $lof = $this->localOutlierFactor($sample);

            $predictions[] = $lof > $this->offset ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Calculate the local outlier factor of a given sample.
     * 
     * @param  array  $sample
     * @throws \RuntimeException
     * @return float
     */
    protected function localOutlierFactor(array $sample) : float
    {
        if (empty($this->lrds)) {
            throw new RuntimeException('Local reachability distances have'
                . ' not been computed, must train estimator first.');
        }

        list($distances) = $this->neighbors($sample, $this->k);

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
     * @param  array  $distances
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

        $mean = Stats::mean($rds);

        return 1. / ($mean ?: self::EPSILON);
    }
}
