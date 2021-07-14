<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use const Rubix\ML\TWO_PI;

/**
 * Gaussian MLE
 *
 * The Gaussian Maximum Likelihood Estimator (MLE) is able to spot outliers by computing
 * a probability density function (PDF) over the features assuming they are independently
 * and normally (Gaussian) distributed. Samples that are assigned low probability density
 * are more likely to be outliers.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing
 * Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMLE implements Estimator, Learner, Online, Scoring, Persistable
{
    use AutotrackRevisions;

    /**
     * The proportion of outliers that are assumed to be present in the training set.
     *
     * @var float
     */
    protected float $contamination;

    /**
     * The amount of epsilon smoothing added to the variance of each feature.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * The precomputed means of each feature column of the training set.
     *
     * @var float[]
     */
    protected array $means = [
        //
    ];

    /**
     * The precomputed variances of each feature column of the training set.
     *
     * @var float[]
     */
    protected array $variances = [
        //
    ];

    /**
     * A small portion of variance to add for smoothing.
     *
     * @var float|null
     */
    protected ?float $epsilon = null;

    /**
     * The number of samples that have passed through training so far.
     *
     * @var int
     */
    protected int $n = 0;

    /**
     * The minimum log likelihood score necessary to flag an anomaly.
     *
     * @var float|null
     */
    protected ?float $threshold = null;

    /**
     * @param float $contamination
     * @param float $smoothing
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $contamination = 0.1, float $smoothing = 1e-9)
    {
        if ($contamination < 0.0 or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->contamination = $contamination;
        $this->smoothing = $smoothing;
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
            'smoothing' => $this->smoothing,
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
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $this->means = $this->variances = [];

        foreach ($dataset->features() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $this->means[$column] = $mean;
            $this->variances[$column] = $variance;
        }

        $epsilon = max($this->smoothing * max($this->variances), CPU::epsilon());

        foreach ($this->variances as &$variance) {
            $variance += $epsilon;
        }

        $lls = array_map([$this, 'logLikelihood'], $dataset->samples());

        $this->threshold = Stats::quantile($lls, 1.0 - $this->contamination);

        $this->epsilon = $epsilon;

        $this->n = $dataset->numSamples();
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

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, count($this->means)),
        ])->check();

        $n = $dataset->numSamples();

        foreach ($dataset->features() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);

            $oldMean = $this->means[$column];
            $oldVariance = $this->variances[$column];

            $oldVariance -= $this->epsilon;

            $this->means[$column] = (($this->n * $oldMean)
                + ($n * $mean)) / ($this->n + $n);

            $this->variances[$column] = ($this->n
                * $oldVariance + ($n * $variance)
                + ($this->n / ($n * ($this->n + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($this->n + $n);
        }

        $epsilon = max($this->smoothing * max($this->variances), CPU::epsilon());

        foreach ($this->variances as &$variance) {
            $variance += $epsilon;
        }

        $this->epsilon = $epsilon;

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
        if (!$this->means or !$this->variances or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->means))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return $this->logLikelihood($sample) > $this->threshold ? 1 : 0;
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian MLE (' . Params::stringify($this->params()) . ')';
    }
}
