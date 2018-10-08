<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Elastic;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Functions\LogSumExp;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gaussian Naive Bayes
 *
 * A variate of the Naive Bayes classifier that uses a probability density
 * function over continuous features. The distribution of values is assumed to
 * be Gaussian therefore your data might need to be transformed beforehand if
 * it is not normally distributed.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianNB implements Estimator, Elastic, Probabilistic, Persistable
{
    const TWO_PI = 2. * M_PI;

    /**
     * Should we fit the empirical prior probabilities of each class? If not,
     * then a prior of 1 / possible class outcomes is assumed.
     *
     * @var bool
     */
    protected $priors;

    /**
     * A small amount of smoothing to apply to the variance of each gaussian for
     * numerical stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The precomputed prior log probabilities of each label given by their weight.
     *
     * @var array|null
     */
    protected $_priors;

    /**
     * The precomputed means of each feature column of the training set.
     *
     * @var array|null
     */
    protected $means;

    /**
     * The precomputed variances of each feature column of the training set.
     *
     * @var array|null
     */
    protected $variances;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  bool  $priors
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(bool $priors = true, float $epsilon = 1e-8)
    {
        if ($epsilon < 0.) {
            throw new InvalidArgumentException('Smoothing parameter cannot be'
                . ' less than 0.');
        }

        $this->priors = $priors;
        $this->epsilon = $epsilon;
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
     * @return array|null
     */
    public function priors() : ?array
    {
        return $this->_priors;
    }

    /**
     * Return the running mean of each feature column of the training data.
     *
     * @return array|null
     */
    public function means() : ?array
    {
        return $this->means;
    }

    /**
     * Return the running variances of each feature column of the training data.
     *
     * @return array|null
     */
    public function variances() : ?array
    {
        return $this->variances;
    }

    /**
     * Compute the necessary statistics to estimate a probability density for
     * each feature column.
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

        $k = count($classes);

        $this->weights = array_fill_keys($classes, 0);

        $this->_priors = array_fill_keys($classes, log(1. / $k));

        $this->means = $this->variances = array_fill_keys($classes,
            array_fill(0, $dataset->numColumns(), 0.));

        $this->partial($dataset);
    }

    /**
     * Uupdate the rolling means and variances of each feature column using an
     * online updating algorithm.
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

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if (empty($this->weights) or empty($this->means) or empty($this->variances)) {
            $this->train($dataset);
        }

        foreach ($dataset->stratify() as $class => $stratum) {
            $oldMeans = $this->means[$class];
            $oldVariances = $this->variances[$class];
            $oldWeight = $this->weights[$class];

            $n = $stratum->numRows();

            foreach ($stratum->rotate() as $column => $values) {
                list($mean, $variance) = Stats::meanVar($values);

                $this->means[$class][$column] = (($n * $mean)
                    + ($oldWeight * $oldMeans[$column]))
                    / ($oldWeight + $n);

                $this->variances[$class][$column] = ($oldWeight
                    * $oldVariances[$column] + ($n * $variance)
                    + ($oldWeight / ($n * ($oldWeight + $n)))
                    * ($n * $oldMeans[$column] - $n * $mean) ** 2)
                    / ($oldWeight + $n);
            }

            $this->weights[$class] += $n;
        }

        if ($this->priors === true) {
            $total = array_sum($this->weights) ?: self::EPSILON;

            foreach ($this->weights as $class => $weight) {
                $this->_priors[$class] = log($weight / $total);
            }
        }
    }

    /**
    * Calculate the likelihood of the sample being a member of a class and
    * choose the class with the highest likelihood as the prediction.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->means) or is_null($this->variances)) {
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
     * Calculate the probabilities of each class from the joint log likelihood
     * of a sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->means) or is_null($this->variances)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

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
            $means = $this->means[$class];
            $variances = $this->variances[$class];

            $score = $this->_priors[$class];

            foreach ($sample as $column => $feature) {
                $mean = $means[$column];
                $variance = $variances[$column] + $this->epsilon;

                $pdf = -0.5 * log(self::TWO_PI * $variance);
                $pdf -= 0.5 * (($feature - $mean) ** 2) / $variance;

                $score += $pdf;
            }

            $likelihood[$class] = $score;
        }

        return $likelihood;
    }
}
