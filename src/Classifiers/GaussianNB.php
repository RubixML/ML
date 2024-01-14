<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function is_null;
use function log;
use function exp;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\LOG_EPSILON;

/**
 * Gaussian Naive Bayes
 *
 * Gaussian Naive Bayes is a version of the Naive Bayes classifier for continuous features. It places
 * a probability density function over the input features on a class basis and uses Bayes' Theorem to
 * derive the class probabilities. In addition to feature independence, Gaussian NB comes with the
 * additional assumption that all features are normally (Gaussian) distributed.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample
 * Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianNB implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * The class prior log probabilities.
     *
     * @var float[]|null
     */
    protected ?array $logPriors = null;

    /**
     * Should we compute the prior probabilities from the training set?
     *
     * @var bool
     */
    protected bool $fitPriors;

    /**
     * The amount of epsilon smoothing added to the variance of each feature.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var float[]
     */
    protected array $weights = [
        //
    ];

    /**
     * The means of each feature of the training set conditioned on a class basis.
     *
     * @var array<list<float>>
     */
    protected array $means = [
        //
    ];

    /**
     * The variances of each feature of the training set conditioned by class.
     *
     * @var array<list<float>>
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
     * @param float[]|null $priors
     * @param float $smoothing
     * @throws InvalidArgumentException
     */
    public function __construct(?array $priors = null, float $smoothing = 1e-9)
    {
        $logPriors = [];

        if ($priors) {
            $total = array_sum($priors);

            if ($total == 0) {
                throw new InvalidArgumentException('Total class prior'
                    . ' probability cannot be equal to 0.');
            }

            foreach ($priors as $class => $prior) {
                if ($prior < 0) {
                    throw new InvalidArgumentException('Prior probability'
                        . " must be greater than 0, $prior given.");
                }

                $logPriors[$class] = log($prior / $total);
            }
        }

        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->logPriors = $logPriors;
        $this->fitPriors = is_null($priors);
        $this->smoothing = $smoothing;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
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
            'priors' => $this->fitPriors ? null : $this->priors(),
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
     * Return the class prior probabilities.
     *
     * @return float[]|null
     */
    public function priors() : ?array
    {
        return $this->logPriors ? array_map('exp', $this->logPriors) : null;
    }

    /**
     * Return the running means of each feature column of the training data by class.
     *
     * @return array<list<float>>|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the running variances of each feature column of the training data by class.
     *
     * @return array<list<float>>|null
     */
    public function variances() : ?array
    {
        return $this->variances;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->means = $this->variances = $this->weights = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $maxVariance = 0.0;

        foreach ($dataset->stratifyByLabel() as $class => $stratum) {
            if (isset($this->means[$class])) {
                $oldMeans = $this->means[$class];
                $oldVariances = $this->variances[$class];
                $oldWeight = $this->weights[$class];

                $n = $stratum->numSamples();

                $weight = $oldWeight + $n;

                $means = $variances = [];

                foreach ($stratum->features() as $column => $values) {
                    $oldMean = $oldMeans[$column];
                    $oldVariance = $oldVariances[$column];

                    $oldVariance -= $this->epsilon;

                    [$mean, $variance] = Stats::meanVar($values);

                    $means[] = (($n * $mean)
                        + ($oldWeight * $oldMean))
                        / $weight;

                    $variances[] = ($oldWeight
                        * $oldVariance + ($n * $variance)
                        + ($oldWeight / ($n * $weight))
                        * ($n * $oldMean - $n * $mean) ** 2)
                        / $weight;
                }
            } else {
                $means = $variances = [];

                foreach ($stratum->features() as $values) {
                    [$mean, $variance] = Stats::meanVar($values);

                    $means[] = $mean;
                    $variances[] = $variance;
                }

                $weight = $stratum->numSamples();
            }

            $maxVariance = max($maxVariance, ...$variances);

            $this->means[$class] = $means;
            $this->variances[$class] = $variances;
            $this->weights[$class] = $weight;
        }

        $epsilon = max($this->smoothing * $maxVariance, CPU::epsilon());

        foreach ($this->variances as &$variances) {
            foreach ($variances as &$variance) {
                $variance += $epsilon;
            }
        }

        if ($this->fitPriors) {
            $total = array_sum($this->weights);

            foreach ($this->weights as $class => $weight) {
                $this->logPriors[$class] = log($weight / $total);
            }
        }

        $this->epsilon = $epsilon;
    }

    /**
     * Calculate the likelihood of the sample being a member of a class and choose the class with the highest likelihood as the prediction.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->means)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param (int|float)[] $sample
     * @return string
     */
    public function predictSample(array $sample) : string
    {
        return argmax($this->jointLogLikelihood($sample));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->means)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @internal
     *
     * @param (int|float)[] $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $jll = $this->jointLogLikelihood($sample);

        $total = logsumexp($jll);

        $dist = [];

        foreach ($jll as $class => $likelihood) {
            $dist[$class] = exp($likelihood - $total);
        }

        return $dist;
    }

    /**
     * Calculate the joint log likelihood of a sample being a member of each class.
     *
     * @param list<int|float> $sample
     * @return array<string,float>
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihoods = [];

        foreach ($this->means as $class => $means) {
            $variances = $this->variances[$class];

            $likelihood = $this->logPriors[$class] ?? LOG_EPSILON;

            foreach ($sample as $column => $value) {
                $mean = $means[$column];
                $variance = $variances[$column];

                $pdf = -0.5 * log(TWO_PI * $variance);
                $pdf -= 0.5 * (($value - $mean) ** 2) / $variance;

                $likelihood += $pdf;
            }

            $likelihoods[$class] = $likelihood;
        }

        return $likelihoods;
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
        return 'Gaussian NB (' . Params::stringify($this->params()) . ')';
    }
}
