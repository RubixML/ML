<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\AutotrackRevisions;
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

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;
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
    protected $logPriors;

    /**
     * Should we compute the prior probabilities from the training set?
     *
     * @var bool
     */
    protected $fitPriors;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var float[]
     */
    protected $weights = [
        //
    ];

    /**
     * The means of each feature of the training set conditioned on a class basis.
     *
     * @var array[]
     */
    protected $means = [
        //
    ];

    /**
     * The variances of each feature of the training set conditioned by class.
     *
     * @var array[]
     */
    protected $variances = [
        //
    ];

    /**
     * @param (int|float)[]|null $priors
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(?array $priors = null)
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

        $this->logPriors = $logPriors;
        $this->fitPriors = is_null($priors);
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
     * @return array[]|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the running variances of each feature column of the training data by class.
     *
     * @return array[]|null
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

        foreach ($dataset->stratify() as $class => $stratum) {
            if (isset($this->means[$class])) {
                $oldMeans = $this->means[$class];
                $oldVariances = $this->variances[$class];
                $oldWeight = $this->weights[$class];

                $n = $stratum->numRows();

                $means = $variances = [];

                foreach ($stratum->columns() as $column => $values) {
                    [$mean, $variance] = Stats::meanVar($values);

                    $means[] = (($n * $mean)
                        + ($oldWeight * $oldMeans[$column]))
                        / ($oldWeight + $n);

                    $vHat = ($oldWeight
                        * $oldVariances[$column] + ($n * $variance)
                        + ($oldWeight / ($n * ($oldWeight + $n)))
                        * ($n * $oldMeans[$column] - $n * $mean) ** 2)
                        / ($oldWeight + $n);

                    $variances[] = $vHat ?: EPSILON;
                }

                $weight = $oldWeight + $n;
            } else {
                $means = $variances = [];

                foreach ($stratum->columns() as $values) {
                    [$mean, $variance] = Stats::meanVar($values);

                    $means[] = $mean;
                    $variances[] = $variance ?: EPSILON;
                }

                $weight = $stratum->numRows();
            }

            $this->means[$class] = $means;
            $this->variances[$class] = $variances;
            $this->weights[$class] = $weight;
        }

        if ($this->fitPriors) {
            $total = array_sum($this->weights);

            foreach ($this->weights as $class => $weight) {
                $this->logPriors[$class] = log($weight / $total);
            }
        }
    }

    /**
     * Calculate the likelihood of the sample being a member of a class and
     * choose the class with the highest likelihood as the prediction.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float[]>
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
     * @return float[]
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
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian NB (' . Params::stringify($this->params()) . ')';
    }
}
