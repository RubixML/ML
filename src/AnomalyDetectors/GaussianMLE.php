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
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Other\Traits\ScoresSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function Rubix\ML\warn_deprecated;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;

/**
 * Gaussian MLE
 *
 * The Gaussian Maximum Likelihood Estimator (MLE) is able to spot outliers by computing
 * a probability density function (PDF) over the features assuming they are independently
 * and normally (Gaussian) distributed. Samples that are assigned low probability density
 * are more likely to be outliers.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMLE implements Estimator, Learner, Online, Ranking, Persistable, Stringable
{
    use PredictsSingle, ScoresSingle;

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
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
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
     * @internal
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
        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            SamplesAreCompatibleWithEstimator::with($dataset, $this),
        ]);

        $this->means = $this->variances = [];

        foreach ($dataset->columns() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = $mean;
            $this->variances[$column] = $variance ?: EPSILON;
        }

        $lls = array_map([$this, 'logLikelihood'], $dataset->samples());

        $this->threshold = Stats::quantile($lls, 1.0 - $this->contamination);

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

        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            DatasetHasDimensionality::with($dataset, count($this->means)),
            SamplesAreCompatibleWithEstimator::with($dataset, $this),
        ]);

        $n = $dataset->numRows();

        foreach ($dataset->columns() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $oldMean = $this->means[$column];
            $oldVariance = $this->variances[$column];

            $muHat = (($this->n * $oldMean) + ($n * $mean))
                / ($this->n + $n);

            $vHat = ($this->n * $oldVariance + ($n * $variance)
                + ($this->n / ($n * ($this->n + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($this->n + $n);

            $this->means[$column] = $muHat;
            $this->variances[$column] = $vHat ?: EPSILON;
        }

        $this->n += $n;

        $lls = array_map([$this, 'logLikelihood'], $dataset->samples());

        $threshold = Stats::quantile($lls, 1.0 - $this->contamination);

        $weight = $n / $this->n;

        $this->threshold = (1.0 - $weight) * $this->threshold + $weight * $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'decide'], $this->score($dataset));
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->means))->check();

        return array_map([$this, 'logLikelihood'], $dataset->samples());
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<float>
     */
    public function rank(Dataset $dataset) : array
    {
        warn_deprecated('Rank() is deprecated, use score() instead.');

        return $this->score($dataset);
    }

    /**
     * Calculate the log likelihood of a sample being an outlier.
     *
     * @param list<int|float> $sample
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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian MLE (' . Params::stringify($this->params()) . ')';
    }
}
