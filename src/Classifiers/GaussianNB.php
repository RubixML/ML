<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\logsumexp;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;
use const Rubix\ML\LOG_EPSILON;

/**
 * Gaussian Naive Bayes
 *
 * A variate of the Naive Bayes classifier that uses a probability density
 * function over continuous features which are assumed to be normally
 * distributed.
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
    /**
     * The class prior probabilities.
     *
     * @var array|null
     */
    protected $priors;

    /**
     * Should we compute the prior probabilities from the training set?
     *
     * @var bool
     */
    protected $fitPriors;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
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
     * The possible class labels.
     *
     * @var (int|string)[]
     */
    protected $classes = [
        //
    ];

    /**
     * @param array|null $priors
     * @throws \InvalidArgumentException
     */
    public function __construct(?array $priors = null)
    {
        if ($priors) {
            foreach ($priors as $weight) {
                if (!is_int($weight) and !is_float($weight)) {
                    throw new InvalidArgumentException('Weight must be'
                        . ' an integer or float, ' . gettype($weight)
                        . ' found.');
                }
            }

            $total = array_sum($priors) ?: EPSILON;

            if ($total != 1) {
                foreach ($priors as &$weight) {
                    $weight = log($weight / $total);
                }
            }
        }

        $this->priors = $priors;
        $this->fitPriors = is_null($priors);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
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
     * Return the class prior probabilities.
     *
     * @return float[]
     */
    public function priors() : array
    {
        $priors = [];

        if ($this->priors) {
            $total = logsumexp($this->priors);

            foreach ($this->priors as $class => $weight) {
                $priors[$class] = exp($weight - $total);
            }
        }

        return $priors;
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
     * Compute the necessary statistics to estimate a probability density for
     * each feature column.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $classes = $dataset->possibleOutcomes();

        $this->classes = $classes;
        $this->weights = array_fill_keys($classes, 0);

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
            $this->priors = [];

            $total = array_sum($this->weights) ?: EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->priors[$class] = log($weight / $total);
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
        if (empty($this->weights) or empty($this->means) or empty($this->variances)) {
            $this->train($dataset);

            return;
        }

        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        foreach ($dataset->stratify() as $class => $stratum) {
            $means = $this->means[$class];
            $variances = $this->variances[$class];

            $oldWeight = $this->weights[$class];
            $oldMeans = $this->means[$class];
            $oldVariances = $this->variances[$class];

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
            
            $this->weights[$class] += $n;
        }

        if ($this->fitPriors) {
            $total = array_sum($this->weights) ?: EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->priors[$class] = log($weight / $total);
            }
        }
    }

    /**
     * Calculate the likelihood of the sample being a member of a class and
     * choose the class with the highest likelihood as the prediction.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->means) or empty($this->variances)) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $jll = array_map([self::class, 'jointLogLikelihood'], $dataset->samples());

        return array_map('Rubix\ML\argmax', $jll);
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->means) or empty($this->variances)) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $probabilities = [];

        foreach ($dataset as $sample) {
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
     * @param array $sample
     * @return array
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihoods = [];

        foreach ($this->means as $class => $means) {
            $variances = $this->variances[$class];

            $likelihood = $this->priors[$class] ?? LOG_EPSILON;

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
