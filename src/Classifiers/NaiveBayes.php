<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Functions\LogSumExp;
use InvalidArgumentException;

/**
 * Naive Bayes
 *
 * Probability-based classifier that uses probabilistic inference to derive the
 * predicted class. The posterior probabilities are calculated using [Bayes'
 * Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). and the naive part
 * relates to the fact that it assumes that all features are independent. In
 * practice, the independent assumption tends to work out most of the time
 * despite most features being correlated in the real world.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NaiveBayes implements Estimator, Online, Probabilistic, Persistable
{
    const LOG_EPSILON = -8;

    /**
     * The amount of additive (Laplace) smoothing to apply to the probabilities.
     *
     * @var float
     */
    protected $alpha;

    /**
     * Should we fit the empirical prior probabilities of each class? If not,
     * then a prior of 1 / possible class outcomes is assumed.
     *
     * @var bool
     */
    protected $priors;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The prior negative log probabilities of each label.
     *
     * @var array
     */
    protected $_priors = [
        //
    ];

    /**
     * The count of each feature from the training set used for online
     * probability calculation.
     *
     * @var array
     */
    protected $counts = [
        //
    ];

    /**
     * The precomputed negative log probabilities of each feature conditioned on
     * a given class label.
     *
     * @var array
     */
    protected $probs = [
        //
    ];

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  float  $alpha
     * @param  bool  $priors
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.0, bool $priors = true)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Smoothing parameter cannot be'
                . ' less than 0.');
        }

        $this->alpha = $alpha;
        $this->priors = $priors;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the class prior log probabilities based on their weight over all
     * training samples.
     *
     * @return array
     */
    public function priors() : array
    {
        return $this->_priors;
    }

    /**
     * Return the log probabilities of each feature given each class label.
     *
     * @return array
     */
    public function probabilities() : array
    {
        return $this->probs;
    }

    /**
     * Compute the probabilities of each feature in the training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $classes = $dataset->possibleOutcomes();

        $this->classes = $classes;

        $this->weights = array_fill_keys($classes, 0);

        $this->_priors = array_fill_keys($classes, log(1./ count($classes)));

        $this->counts = $this->probs = array_fill_keys($classes,
            array_fill(0, $dataset->numColumns(), []));

        $this->partial($dataset);
    }

    /**
     * Compute the rolling counts and probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(Dataset::CONTINUOUS, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' categorical features.');
        }

        if (empty($this->weights) or empty($this->counts)) {
            $this->train($dataset);
        }

        foreach ($dataset->stratify() as $class => $stratum) {
            $classCounts = $this->counts[$class];

            foreach ($stratum->rotate() as $column => $values) {
                $columnCounts = $classCounts[$column];

                foreach (array_count_values($values) as $category => $count) {
                    if (isset($counts[$category])) {
                        $columnCounts[$category] += $count;
                    } else {
                        $columnCounts[$category] = $count;
                    }
                }

                $sigma = array_sum($columnCounts)
                    + (count($columnCounts) * $this->alpha);

                $probs = [];

                foreach ($columnCounts as $category => $count) {
                    $probs[$category] = log(($count + $this->alpha) / $sigma);
                }

                $this->counts[$class][$column] = $columnCounts;
                $this->probs[$class][$column] = $probs;
            }

            $this->weights[$class] += count($stratum);
        }

        if ($this->priors === true) {
            $sigma = array_sum($this->weights)
                + (count($this->weights) * $this->alpha);

            foreach ($this->weights as $class => $weight) {
                $this->_priors[$class] = log(($weight + $this->alpha) / $sigma);
            }
        }
    }

    /**
    * Calculate the likelihood of the sample being a member of a class and
    * chose the class with the highest likelihood score as the prediction.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $predictions[] = Argmax::compute($jll);
        }

        return $predictions;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = [];

        foreach ($dataset as $i => $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $max = LogSumExp::compute($jll);

            foreach ($jll as $class => $likelihood) {
                $probabilities[$i][$class] = exp($likelihood - $max);
            }
        }

        return $probabilities;
    }

    /**
     * Calculate the joint log likelihood of a sample being a member of each class.
     *
     * @param  array  $sample
     * @return array
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihood = [];

        foreach ($this->classes as $class) {
            $probs = $this->probs[$class];

            $score = $this->_priors[$class];

            foreach ($sample as $column => $feature) {
                $score += $probs[$column][$feature] ?? self::LOG_EPSILON;
            }

            $likelihood[$class] = $score;
        }

        return $likelihood;
    }
}
