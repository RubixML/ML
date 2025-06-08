<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
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
final class GaussianMLE implements Estimator, Learner, Online, Scoring, Persistable
{
    use AutotrackRevisions;

    /**
     * The proportion of outliers that are assumed to be present in the training set.
     */
    protected float $contamination;

    /**
     * The amount of epsilon smoothing added to the variance of each feature.
     */
    protected float $smoothing;

    /**
     * The precomputed means of each feature column of the training set.
     * @var array<float>
     */
    protected array $means = [];

    /**
     * The precomputed variances of each feature column of the training set.
     * @var array<float>
     */
    protected array $variances = [];

    /**
     * A small portion of variance to add for smoothing.
     */
    protected ?float $epsilon = null;

    /**
     * The number of samples that have passed through training so far.
     */
    protected int $n = 0;

    /**
     * The minimum log likelihood score necessary to flag an anomaly.
     */
    protected ?float $threshold = null;

    /**
     * @param float $contamination
     * @param float $smoothing
     * @throws InvalidArgumentException
     */
    public function __construct(float $contamination = 0.1, float $smoothing = 1e-9)
    {
        if ($contamination < 0.0 || $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be between 0 and 0.5, ' . $contamination . ' given.');
        }

        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be greater than 0, ' . $smoothing . ' given.');
        }

        $this->contamination = $contamination;
        $this->smoothing = $smoothing;
    }

    /**
     * Return the estimator type.
     *
     * @internal
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
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return [DataType::continuous()];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return array<string, mixed>
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
     */
    public function trained() : bool
    {
        return !empty($this->means) && !empty($this->variances);
    }

    /**
     * Return the column means computed from the training set.
     *
     * @return array<float>
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the column variances computed from the training set.
     *
     * @return array<float>
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Train the learner with a dataset.
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

        $this->updateVariancesWithSmoothing();
        $this->calculateThreshold($dataset);
        $this->n = $dataset->numSamples();
    }

    /**
     * Perform a partial train on the learner.
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->trained()) {
            $this->train($dataset);
            return;
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, count($this->means)),
        ])->check();

        $n = $dataset->numSamples();
        $weight = $this->n + $n;

        foreach ($dataset->features() as $column => $values) {
            [$mean, $variance] = Stats::meanVar($values);
            $oldMean = $this->means[$column];
            $oldVariance = $this->variances[$column] - $this->epsilon;

            $this->means[$column] = (($this->n * $oldMean) + ($n * $mean)) / $weight;
            
            $this->variances[$column] = ($this->n * $oldVariance + ($n * $variance)
                + ($this->n / ($n * $weight)) * ($n * $oldMean - $n * $mean) ** 2) / $weight;
        }

        $this->updateVariancesWithSmoothing();
        $this->updateThreshold($dataset, $n, $weight);
        $this->n = $weight;
    }

    /**
     * Make predictions from a dataset.
     *
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained() || $this->threshold === null) {
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
     */
    public function predictSample(array $sample) : int
    {
        return $this->logLikelihood($sample) > $this->threshold ? 1 : 0;
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @throws RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->means))->check();

        return array_map([$this, 'logLikelihood'], $dataset->samples());
    }

    /**
     * Calculate the log likelihood of a sample being an outlier.
     *
     * @param list<int|float> $sample
     */
    protected function logLikelihood(array $sample) : float
    {
        $likelihood = 0.0;

        foreach ($sample as $column => $value) {
            $mean = $this->means[$column];
            $variance = $this->variances[$column];
            
            $likelihood += 0.5 * (log(TWO_PI * $variance) + (($value - $mean) ** 2) / $variance;
        }

        return $likelihood;
    }

    /**
     * Update variances with smoothing epsilon.
     */
    private function updateVariancesWithSmoothing() : void
    {
        $this->epsilon = max($this->smoothing * max($this->variances), CPU::epsilon());
        
        foreach ($this->variances as &$variance) {
            $variance += $this->epsilon;
        }
    }

    /**
     * Calculate the anomaly threshold based on training data.
     */
    private function calculateThreshold(Dataset $dataset) : void
    {
        $lls = array_map([$this, 'logLikelihood'], $dataset->samples());
        $this->threshold = Stats::quantile($lls, 1.0 - $this->contamination);
    }

    /**
     * Update the anomaly threshold during partial training.
     */
    private function updateThreshold(Dataset $dataset, int $n, int $weight) : void
    {
        $lls = array_map([$this, 'logLikelihood'], $dataset->samples());
        $threshold = Stats::quantile($lls, 1.0 - $this->contamination);
        $proportion = $n / $weight;
        $this->threshold = $proportion * $threshold + (1.0 - $proportion) * $this->threshold;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     */
    public function __toString() : string
    {
        return 'Gaussian MLE (' . Params::stringify($this->params()) . ')';
    }
}
