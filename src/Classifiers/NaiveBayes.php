<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
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
 * Probability-based classifier that used probabilistic inference to derive the
 * correct class. The probabilities are calculated using Bayes Rule. The naive
 * part relates to the fact that it assumes that all features are independent,
 * which is rarely the case in the real world but tends to work out in practice
 * for most problems.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NaiveBayes implements Multiclass, Online, Probabilistic, Persistable
{
    /**
     * The amount of additive (Laplace) smoothing to apply to the feature
     * probabilities.
     *
     * @var float
     */
    protected $smoothing;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The precomputed prior log probabilities of each label.
     *
     * @var array
     */
    protected $priors = [
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
     * The precomputed log probabilities of each feature given each class label.
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
     * @param  float  $smoothing
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $smoothing = 1.0)
    {
        if ($smoothing < 0.0) {
            throw new InvalidArgumentException('Smoothing parameter cannot be'
                . ' less than 0.');
        }

        $this->smoothing = $smoothing;
    }

    /**
     * Return the class prior log probabilities based on their weight over all
     * training samples.
     *
     * @return array
     */
    public function priors() : array
    {
        return $this->priors;
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

        $this->classes = $dataset->possibleOutcomes();

        $this->weights = array_fill_keys($this->classes, 0);

        $this->priors = array_fill_keys($this->classes, 0.0);

        $this->counts = $this->probs = array_fill_keys($this->classes,
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

        if (in_array(self::CONTINUOUS, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' categorical features.');
        }

        if (empty($this->classes) or empty($this->priors)) {
            $this->train($dataset);
        }

        foreach ($dataset->stratify() as $class => $samples) {
            foreach ($samples->rotate() as $column => $values) {
                $counts = array_count_values((array) $values);

                foreach ($counts as $value => $count) {
                    if (!isset($this->counts[$class][$column][$value])) {
                        $this->counts[$class][$column][$value] = $count;
                    } else {
                        $this->counts[$class][$column][$value] += $count;
                    }
                }

                $n = count($this->counts[$class][$column]);

                $total = array_sum($this->counts[$class][$column])
                    + ($n * $this->smoothing);

                foreach ($this->counts[$class][$column] as $value => $count) {
                    $probability = ($count + $this->smoothing) / $total;

                    $this->probs[$class][$column][$value] = log($probability);
                }
            }

            $this->weights[$class] += count($samples);
        }

        $total = (count($this->weights) * $this->smoothing)
            + array_sum($this->weights);

        foreach ($this->weights as $class => $weight) {
            $this->priors[$class] = log(($weight + $this->smoothing) / $total);
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
            $jll = $this->computeJointLogLikelihood($sample);

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
            $jll = $this->computeJointLogLikelihood($sample);

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
    protected function computeJointLogLikelihood(array $sample) : array
    {
        $likelihood = [];

        foreach ($this->classes as $class) {
            $score = $this->priors[$class];

            foreach ($sample as $column => $feature) {
                $score += $this->probs[$class][$column][$feature]
                    ?? log(self::EPSILON);
            }

            $likelihood[$class] = $score;
        }

        return $likelihood;
    }
}
