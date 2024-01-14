<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
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
use function array_count_values;
use function array_sum;
use function count;
use function log;

use const Rubix\ML\LOG_EPSILON;

/**
 * Naive Bayes
 *
 * Categorical Naive Bayes is a probability-based classifier that uses counting and Bayes' Theorem
 * to derive the probabilities of a class given a sample of categorical features. The term *naive*
 * refers to the fact that Naive Bayes treats each feature as if it was independent of the others
 * even though this is usually not the case in real life.
 *
 * > **Note:** Each partial train has the overhead of recomputing the probability mass function for
 * each feature per class. As such, it is better to train with fewer but larger training sets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NaiveBayes implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * The class prior log probabilities.
     *
     * @var array<string,float>|null
     */
    protected ?array $logPriors = null;

    /**
     * Should we compute the prior probabilities from the training set?
     *
     * @var bool
     */
    protected bool $fitPriors;

    /**
     * The amount of Laplace smoothing added to the probabilities.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array<string,int>
     */
    protected array $classCounts = [
        //
    ];

    /**
     * The count of each category from the training set on a class basis.
     *
     * @var array<string,list<array<int<0,max>>>>
     */
    protected array $counts = [
        //
    ];

    /**
     * The precomputed negative log likelihoods of each feature conditioned on a particular class label.
     *
     * @var array<string,list<float[]>>
     */
    protected array $probs = [
        //
    ];

    /**
     * @param float[]|null $priors
     * @param float $smoothing
     * @throws InvalidArgumentException
     */
    public function __construct(?array $priors = null, float $smoothing = 1.0)
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
            DataType::categorical(),
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
        return $this->classCounts and $this->counts and $this->probs;
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
     * Return the counts for each category on a per class basis.
     *
     * @return array<list<array<int<0,max>>>>>|null
     */
    public function counts() : ?array
    {
        return $this->counts;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->classCounts = $this->counts = $this->probs = [];

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

        foreach ($dataset->stratifyByLabel() as $class => $stratum) {
            if (isset($this->counts[$class])) {
                $classCounts = $this->counts[$class];
                $classProbs = $this->probs[$class];
            } else {
                $classCounts = $classProbs = array_fill(0, $stratum->numFeatures(), []);

                $this->classCounts[$class] = 0;
            }

            foreach ($stratum->features() as $column => $values) {
                $columnCounts = $classCounts[$column];

                $counts = array_count_values($values);

                foreach ($counts as $category => $count) {
                    if (isset($columnCounts[$category])) {
                        $columnCounts[$category] += $count;
                    } else {
                        $columnCounts[$category] = $count;
                    }
                }

                $total = array_sum($columnCounts) + $this->smoothing * count($columnCounts);

                $probs = [];

                foreach ($columnCounts as $category => $count) {
                    $probs[$category] = log(($count + $this->smoothing) / $total);
                }

                $classCounts[$column] = $columnCounts;
                $classProbs[$column] = $probs;
            }

            $this->counts[$class] = $classCounts;
            $this->probs[$class] = $classProbs;

            $this->classCounts[$class] += $stratum->numSamples();
        }

        if ($this->fitPriors) {
            $total = array_sum($this->classCounts);

            foreach ($this->classCounts as $class => $weight) {
                $this->logPriors[$class] = log($weight / $total);
            }
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->classCounts or !$this->probs) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->probs)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<string> $sample
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
        if (!$this->classCounts or !$this->probs) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->probs)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @internal
     *
     * @param list<string> $sample
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
     * @param list<string> $sample
     * @return array<string,float>
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

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Naive Bayes (' . Params::stringify($this->params()) . ')';
    }
}
