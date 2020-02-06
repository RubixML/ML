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
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\logsumexp;
use function count;

use const Rubix\ML\EPSILON;
use const Rubix\ML\LOG_EPSILON;

/**
 * Naive Bayes
 *
 * Probability-based classifier that uses Bayes' Theorem and the strong assumption that
 * all features are independent. In practice, the independence assumption tends to work
 * out despite most features being correlated in the real world. This particular
 * implementation is based on a categorical distribution of input features.
 *
 * > **Note:** Each partial train has the overhead of recomputing the probability
 * mass function for each feature per class. As such, it is better to train with
 * fewer but larger training sets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NaiveBayes implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use PredictsSingle, ProbaSingle;
    
    /**
     * The amount of additive (Laplace) smoothing to apply to the probabilities.
     *
     * @var float
     */
    protected $alpha;

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
     * The count of each feature from the training set used for online
     * probability calculation.
     *
     * @var array[]
     */
    protected $counts = [
        //
    ];

    /**
     * The precomputed negative log probabilities of each feature conditioned on
     * a given class label.
     *
     * @var array[]
     */
    protected $probs = [
        //
    ];

    /**
     * @param float $alpha
     * @param (int|float)[]|null $priors
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0, ?array $priors = null)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha cannot be less'
                . " than 0, $alpha given.");
        }

        if ($priors) {
            $total = array_sum($priors) ?: EPSILON;

            foreach ($priors as &$prior) {
                $prior = log($prior / $total);
            }
        }
        
        $this->alpha = $alpha;
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
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
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
            'alpha' => $this->alpha,
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
        return $this->weights and $this->counts and $this->probs;
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
     * Return the counts for each category per class.
     *
     * @return array[]|null
     */
    public function counts() : ?array
    {
        return $this->counts;
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
                . ' labeled training set.');
        }

        $this->weights = $this->counts = $this->probs = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        foreach ($dataset->stratify() as $class => $stratum) {
            if (isset($this->counts[$class])) {
                $classCounts = $this->counts[$class];
                $classProbs = $this->probs[$class];
            } else {
                $classCounts = $classProbs = array_fill(0, $stratum->numColumns(), []);

                $this->weights[$class] = 0;
            }

            foreach ($stratum->columns() as $column => $values) {
                $columnCounts = $classCounts[$column];

                $counts = array_count_values($values);

                foreach ($counts as $category => $count) {
                    if (isset($columnCounts[$category])) {
                        $columnCounts[$category] += $count;
                    } else {
                        $columnCounts[$category] = $count;
                    }
                }

                $total = (array_sum($columnCounts)
                    + (count($columnCounts) * $this->alpha))
                    ?: EPSILON;

                $probs = [];

                foreach ($columnCounts as $category => $count) {
                    $probs[$category] = log(($count + $this->alpha) / $total);
                }

                $classCounts[$column] = $columnCounts;
                $classProbs[$column] = $probs;
            }

            $this->counts[$class] = $classCounts;
            $this->probs[$class] = $classProbs;

            $this->weights[$class] += $stratum->numRows();
        }

        if ($this->fitPriors) {
            $this->logPriors = [];
            
            $total = (array_sum($this->weights)
                + (count($this->weights) * $this->alpha))
                ?: EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->logPriors[$class] = log(($weight + $this->alpha) / $total);
            }
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->weights or !$this->probs) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $jll = array_map([self::class, 'jointLogLikelihood'], $dataset->samples());

        return array_map('Rubix\ML\argmax', $jll);
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->weights or !$this->probs) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

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
     * @param string[] $sample
     * @return float[]
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihoods = [];

        foreach ($this->probs as $class => $probs) {
            $likelihood = $this->logPriors[$class] ?? LOG_EPSILON;

            foreach ($sample as $column => $value) {
                $likelihood += $probs[$column][$value] ?? LOG_EPSILON;
            }

            $likelihoods[$class] = $likelihood;
        }

        return $likelihoods;
    }
}
