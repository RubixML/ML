<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Backends\Tasks\Proba;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\array_transpose;
use function array_combine;
use function array_keys;
use function array_map;
use function array_sum;

/**
 * One Vs Rest
 *
 * One Vs Rest is an ensemble learner that trains a binary classifier to predict a particular class
 * vs every other class for every possible class. The final class prediction is the class whose
 * binary classifier returned the highest probability. One of the features of One Vs Rest is that
 * it allows you to build a multiclass classifier out of an ensemble of otherwise binary classifiers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneVsRest implements Estimator, Learner, Probabilistic, Parallel, Persistable
{
    use AutotrackRevisions, Multiprocessing;

    /**
     * The base classifier.
     *
     * @var Learner
     */
    protected Learner $base;

    /**
     * A map of each class to its binary classifier.
     *
     * @var array<\Rubix\ML\Learner>
     */
    protected array $classifiers = [
        //
    ];

    /**
     * The dimensionality of the training set.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * @param Learner $base
     * @throws InvalidArgumentException
     */
    public function __construct(Learner $base)
    {
        if (!$base->type()->isClassifier()) {
            throw new InvalidArgumentException('Base Learner must be'
                . ' a classifier.');
        }

        if (!$base instanceof Probabilistic) {
            throw new InvalidArgumentException('Base classifier must'
                . ' implement the Probabilistic interface.');
        }

        $this->base = $base;
        $this->backend = new Serial();
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
        return $this->base->compatibility();
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
            'base' => $this->base,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->classifiers);
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $classes = $dataset->possibleOutcomes();

        $this->backend->flush();

        foreach ($classes as $class) {
            $estimator = clone $this->base;
            $subset = clone $dataset;

            $binarize = function ($label) use ($class) {
                return $label === $class ? 'y' : 'n';
            };

            $subset->transformLabels($binarize);

            $task = new TrainLearner($estimator, $subset);

            $this->backend->enqueue($task);
        }

        $classifiers = $this->backend->process();

        $classifiers = array_combine($classes, $classifiers) ?: [];

        $this->classifiers = $classifiers;

        $this->featureCount = $dataset->numFeatures();
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
        return array_map('\Rubix\ML\argmax', $this->proba($dataset));
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
        if (!$this->classifiers or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $this->backend->flush();

        /** @var Probabilistic $estimator */
        foreach ($this->classifiers as $estimator) {
            $task = new Proba($estimator, $dataset);

            $this->backend->enqueue($task);
        }

        $aggregate = $this->backend->process();

        $aggregate = array_transpose($aggregate);

        $classes = array_keys($this->classifiers);

        $probabilities = [];

        foreach ($aggregate as $votes) {
            $dist = [];

            foreach ($votes as $j => $proba) {
                $dist[$classes[$j]] = $proba['y'];
            }

            $total = array_sum($dist);

            foreach ($dist as &$probability) {
                $probability /= $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
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
        return 'One Vs Rest (' . Params::stringify($this->params()) . ')';
    }
}
