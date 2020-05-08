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
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

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
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianNB implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use PredictsSingle, ProbaSingle;
    
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
     * The precomputed means of each feature column of the training set.
     *
     * @var array[]
     */
    protected $means = [
        //
    ];

    /**
     * The precomputed variances of each feature column of the training set.
     *
     * @var array[]
     */
    protected $variances = [
        //
    ];

    /**
     * @param (int|float)[]|null $priors
     * @throws \InvalidArgumentException
     */
    public function __construct(?array $priors = null)
    {
        if ($priors) {
            $total = array_sum($priors) ?: EPSILON;

            foreach ($priors as &$prior) {
                $prior = log($prior / $total);
            }
        }

        $this->logPriors = $priors;
        $this->fitPriors = is_null($priors);
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that the model is compatible with.
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
            'priors' => $this->priors(),
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
     * Return the running mean of each feature column of the training data.
     *
     * @return array[]|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the running variances of each feature column of the training data.
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $classes = $dataset->possibleOutcomes();

        $this->weights = array_fill_keys($classes, 0.0);

        $this->means = $this->variances = array_fill_keys($classes, []);

        foreach ($dataset->stratify() as $class => $stratum) {
            $means = $variances = [];

            foreach ($stratum->columns() as $values) {
                [$mean, $variance] = Stats::meanVar($values);

                $means[] = $mean;
                $variances[] = $variance ?: EPSILON;
            }

            $this->means[$class] = $means;
            $this->variances[$class] = $variances;

            $this->weights[$class] += $stratum->numRows();
        }

        if ($this->fitPriors) {
            $this->logPriors = [];

            $total = array_sum($this->weights) ?: EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->logPriors[$class] = log($weight / $total);
            }
        }
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->weights or !$this->means or !$this->variances) {
            $this->train($dataset);

            return;
        }

        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        foreach ($dataset->stratify() as $class => $stratum) {
            $means = $oldMeans = $this->means[$class] ?? [];
            $variances = $oldVariances = $this->variances[$class] ?? [];

            $oldWeight = $this->weights[$class] ?? 0;

            $n = $stratum->numRows();

            foreach ($stratum->columns() as $column => $values) {
                [$mean, $variance] = Stats::meanVar($values);

                $means[$column] = (($n * $mean)
                    + ($oldWeight * $oldMeans[$column]))
                    / ($oldWeight + $n);

                $vHat = ($oldWeight
                    * $oldVariances[$column] + ($n * $variance)
                    + ($oldWeight / ($n * ($oldWeight + $n)))
                    * ($n * $oldMeans[$column] - $n * $mean) ** 2)
                    / ($oldWeight + $n);

                $variances[$column] = $vHat ?: EPSILON;
            }

            $this->means[$class] = $means;
            $this->variances[$class] = $variances;
            
            $this->weights[$class] = $oldWeight + $n;
        }

        if ($this->fitPriors) {
            $total = array_sum($this->weights) ?: EPSILON;

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
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $jll = array_map([self::class, 'jointLogLikelihood'], $dataset->samples());

        return array_map('Rubix\ML\argmax', $jll);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->means or !$this->variances) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset->samples() as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $total = logsumexp($jll);

            $dist = [];

            foreach ($jll as $class => $likelihood) {
                $dist[$class] = exp($likelihood - $total);
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }

    /**
     * Calculate the joint log likelihood of a sample being a member of each class.
     *
     * @param (int|float)[] $sample
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
}
