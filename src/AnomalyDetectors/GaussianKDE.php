<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Ranking;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;

/**
 * Gaussian KDE
 *
 * The Gaussian Kernel Density Estimator is able to spot outliers by computing a
 * probability density function over the features assuming they are independent
 * and normally (Gaussian) distributed. Assigning low probability density
 * translates to a high anomaly score. The final anomaly score is given as the
 * negative log likelihood of a sample being an outlier.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianKDE implements Estimator, Learner, Online, Ranking, Persistable
{
    use PredictsSingle;

    /**
     * The minimum negative log likelihood score necessary to flag an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The prior log probability of an anomaly.
     *
     * @var float
     */
    protected $logPrior;

    /**
     * The precomputed means of each feature column of the training set.
     *
     * @var float[]
     */
    protected $means = [
        //
    ];

    /**
     * The precomputed variances of each feature column of the training set.
     *
     * @var float[]
     */
    protected $variances = [
        //
    ];

    /**
     * The number of samples that have passed through training so far.
     *
     * @var int
     */
    protected $n = 0;

    /**
     * @param float $threshold
     * @param float $contamination
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5, float $contamination = 0.1)
    {
        if ($threshold <= 0.) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        if ($contamination < 0. or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->threshold = $threshold;
        $this->logPrior = log($contamination);
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
        return $this->means and $this->variances;
    }

    /**
     * Return the column means computed from the training set.
     *
     * @return array
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the column variances computed from the training set.
     *
     * @return array
     */
    public function variances() : array
    {
        return $this->variances;
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

        $this->means = $this->variances = [];

        foreach ($dataset->columns() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = $mean;
            $this->variances[$column] = $variance ?: EPSILON;
        }

        $this->n = $dataset->numRows();
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->means or !$this->variances) {
            $this->train($dataset);

            return;
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $n = $dataset->numRows();

        foreach ($dataset->columns() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $oldWeight = $this->n;
            $oldMean = $this->means[$column];
            $oldVariance = $this->variances[$column];

            $this->means[$column] = (($n * $mean)
                + ($oldWeight * $oldMean))
                / ($oldWeight + $n);

            $vHat = ($oldWeight
                * $oldVariance + ($n * $variance)
                + ($oldWeight / ($n * ($oldWeight + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($oldWeight + $n);

            $this->variances[$column] = $vHat ?: EPSILON;
        }

        $this->n += $n;
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
     * @return array
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances) {
            throw new RuntimeException('The estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'logLikelihood'], $dataset->samples());
    }

    /**
     * Calculate the negative log likelihood of a sample being an outlier.
     *
     * @param array $sample
     * @return float
     */
    protected function logLikelihood(array $sample) : float
    {
        $likelihood = $this->logPrior;

        foreach ($sample as $column => $value) {
            $mean = $this->means[$column];
            $variance = $this->variances[$column];

            $pdf = 0.5 * log(TWO_PI * $variance);
            $pdf += 0.5 * (($value - $mean) ** 2) / $variance;

            $likelihood += $pdf;
        }

        return $likelihood;
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
