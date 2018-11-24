<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Functions\LogSumExp;
use InvalidArgumentException;
use RuntimeException;

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
class NaiveBayes implements Online, Probabilistic, Persistable
{
    const LOG_EPSILON = -8;

    /**
     * The amount of additive (Laplace) smoothing to apply to the probabilities.
     *
     * @var float
     */
    protected $alpha;

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
     * @var array|null
     */
    protected $weights;

    /**
     * The count of each feature from the training set used for online
     * probability calculation.
     *
     * @var array|null
     */
    protected $counts;

    /**
     * The precomputed negative log probabilities of each feature conditioned on
     * a given class label.
     *
     * @var array|null
     */
    protected $probs;

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
     * @param  array|null  $priors
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.0, ?array $priors = null)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Alpha cannot be less'
                . " than 0, $alpha given.");
        }

        if (is_array($priors)) {
            $total = 0;
            
            foreach ($priors as $class => $probability) {
                if (!is_string($class)) {
                    throw new InvalidArgumentException('Class label must be a'
                        . ' string, ' . gettype($class) . ' found.');
                }

                if (!is_int($probability) and !is_float($probability)) {
                    throw new InvalidArgumentException('Probability must be'
                        . ' an integer or float, ' . gettype($probability)
                        . ' found.');
                }

                $total += $probability;
            }

            if ($total != 1 and $total != 0) {
                foreach ($priors as &$probability) {
                    $probability /= $total;
                }
            }
        }
        
        $this->alpha = $alpha;
        $this->priors = $priors;
        $this->fitPriors = is_null($priors);
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
     * Return the class prior probabilities.
     *
     * @return array
     */
    public function priors() : array
    {
        $priors = [];

        if (is_array($this->priors)) {
            $max = LogSumExp::compute($this->priors);

            foreach ($this->priors as $class => $probability) {
                $priors[$class] = exp($probability - $max);
            }
        }

        return $priors;
    }

    /**
     * Return the conditional log probabilities of each feature given each class
     * label.
     *
     * @return array|null
     */
    public function probabilities() : ?array
    {
        return $this->probs;
    }

    /**
     * Train the estimator with a dataset.
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

        $this->counts = $this->probs = array_fill_keys($classes,
            array_fill(0, $dataset->numColumns(), []));

        $this->partial($dataset);
    }

    /**
     * Compute the rolling counts and conditional probabilities.
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

        if ($dataset->typeCount(DataFrame::CATEGORICAL) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This estimator only works'
                . ' with categorical features.');
        }

        if (is_null($this->weights) or is_null($this->counts) or is_null($this->probs)) {
            $this->train($dataset);
            return;
        }

        foreach ($dataset->stratify() as $class => $stratum) {
            $classCounts = $this->counts[$class];

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
                    ?: self::EPSILON;

                $probs = [];

                foreach ($columnCounts as $category => $count) {
                    $probs[$category] = log(($count + $this->alpha) / $total);
                }

                $this->counts[$class][$column] = $columnCounts;
                $this->probs[$class][$column] = $probs;
            }

            $this->weights[$class] += $stratum->numRows();
        }

        if ($this->fitPriors) {
            $total = (array_sum($this->weights)
                + (count($this->weights) * $this->alpha))
                ?: self::EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->priors[$class] = log(($weight + $this->alpha) / $total);
            }
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CONTINUOUS, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' categorical features.');
        }

        if (is_null($this->probs)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $predictions[] = Argmax::compute($jll);
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CONTINUOUS, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' categorical features.');
        }

        if (is_null($this->probs)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $total = LogSumExp::compute($jll);

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
     * @param  array  $sample
     * @return array
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihood = [];

        foreach ($this->classes as $class) {
            $score = $this->priors[$class] ?? self::LOG_EPSILON;
            $probs = $this->probs[$class];

            foreach ($sample as $column => $feature) {
                $score += $probs[$column][$feature] ?? self::LOG_EPSILON;
            }

            $likelihood[$class] = $score;
        }

        return $likelihood;
    }
}
