<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * Local Outlier Factor
 *
 * The Local Outlier Factor (LOF) algorithm only considers the local region of
 * a sample, set by the k parameter. A density estimate for each neighbor is
 * computed by measuring the radius of the cluster centroid that the point and
 * its neighbors form. The LOF is the ratio of the sample over the median radius
 * of the local region.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LocalOutlierFactor implements Detector, Probabilistic, Online, Persistable
{
    /**
     * The number of nearest neighbors to consider a local region.
     *
     * @var int
     */
    protected $k;

    /**
     * The number of nearest neighbors sampled to estimate the density of a
     * centroid.
     *
     * @var int
     */
    protected $neighbors;

    /**
     * The threshold outlier facter. Factor is a value between 0 and 1 where
     * greater than 0.5 signifies outlier territory.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The distance kernel to use when computing the distances between two
     * data points.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The memoized coordinate vectors of the training data.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * @param  int  $k
     * @param  int  $neighbors
     * @param  float  $threshold
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 10, int $neighbors = 20, float $threshold = 0.5,
                                Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to form a local region.');
        }

        if ($neighbors < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to estimate the density of a centroid.');
        }

        if (!isset($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->neighbors = $neighbors;
        $this->threshold = $threshold;
        $this->kernel = $kernel;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->samples = [];

        $this->partial($dataset);
    }

    /**
     * Store the sample and outcome arrays. No other work to be done as this is
     * a lazy learning algorithm.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->samples = array_merge($this->samples, $dataset->samples());
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probability) {
            $predictions[] = $probability > $this->threshold ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Return a probability estimate based on the density of the sample over the
     * median density of the local region.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probablities = [];

        foreach ($dataset as $sample) {
            $radii = [];

            foreach ($this->findLocalRegion($sample) as $neighbor) {
                $radii[] = $this->calculateRadius($neighbor);
            }

            $radius = $this->calculateRadius($sample);

            $probablities[] = 2.0 ** -(Average::median($radii)
                / ($radius + self::EPSILON));
        }

        return $probablities;
    }

    /**
     * Find the K nearest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findLocalRegion(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $index => $neighbor) {
            $distances[$index] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_intersect_key($this->samples,
            array_slice($distances, 0, $this->k, true));
    }

    /**
     * Calculate the radius of a cluster centroid.
     *
     * @param  array  $sample
     * @return float
     */
    protected function calculateRadius(array $sample) : float
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        sort($distances);

        return $distances[$this->neighbors - 1] ?? end($distances);
    }
}
