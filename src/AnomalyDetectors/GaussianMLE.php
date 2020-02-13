<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Ranking;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;

/**
 * Gaussian MLE
 *
 * The Gaussian Maximum Likelihood Estimator (MLE) is able to spot outliers by computing
 * a probability density function (PDF) over the features assuming they are independently
 * and normally (Gaussian) distributed. Assigning low probability density to a sample
 * translates to a high anomaly score.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMLE implements Estimator, Learner, Online, Ranking, Persistable
{
    use PredictsSingle, RankSingle;

    /**
     * The proportion of outliers that are assumed to be present in the
     * training set.
     *
     * @var float
     */
    protected $contamination;

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
     * The minimum log likelihood score necessary to flag an anomaly.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * The number of samples that have passed through training so far.
     *
     * @var int
     */
    protected $n = 0;

    /**
     * @param float $contamination
     * @throws \InvalidArgumentException
     */
    public function __construct(float $contamination = 0.1)
    {
        if ($contamination < 0.0 or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->contamination = $contamination;
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'contamination' => $this->contamination,
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
     * @return float[]
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the column variances computed from the training set.
     *
     * @return float[]
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $this->means = $this->variances = [];

        foreach ($dataset->columns() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = $mean;
            $this->variances[$column] = $variance ?: EPSILON;
        }

        $lls = array_map([self::class, 'logLikelihood'], $dataset->samples());

        $this->threshold = Stats::percentile($lls, 100.0 * (1.0 - $this->contamination));

        $this->n = $dataset->numRows();
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->means or !$this->variances or !$this->threshold) {
            $this->train($dataset);

            return;
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

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

        $lls = array_map([self::class, 'logLikelihood'], $dataset->samples());

        $threshold = Stats::percentile($lls, 100.0 * (1.0 - $this->contamination));

        $weight = $n / $this->n;

        $this->threshold = (1.0 - $weight) * $this->threshold + $weight * $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'logLikelihood'], $dataset->samples());
    }

    /**
     * Calculate the log likelihood of a sample being an outlier.
     *
     * @param (int|float)[] $sample
     * @return float
     */
    protected function logLikelihood(array $sample) : float
    {
        $likelihood = 0.0;

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
