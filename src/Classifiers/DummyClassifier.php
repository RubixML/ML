<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Other\Strategies\Prior;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function count;

/**
 * Dummy Classifier
 *
 * A classifier that uses a user-defined Guessing Strategy to make predictions.
 * Dummy Classifier is useful to provide a sanity check and to compare
 * performance with an actual classifier.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DummyClassifier implements Estimator, Learner, Persistable, Stringable
{
    use PredictsSingle;

    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $strategy;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param \Rubix\ML\Other\Strategies\Categorical|null $strategy
     */
    public function __construct(?Categorical $strategy = null)
    {
        $this->strategy = $strategy ?? new Prior();
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
        return DataType::all();
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
            'strategy' => $this->strategy,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->featureCount);
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

        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            LabelsAreCompatibleWithLearner::with($dataset, $this),
        ]);

        $this->strategy->fit($dataset->labels());

        $this->featureCount = $dataset->numColumns();
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $m = $dataset->numRows();

        $predictions = [];

        while (count($predictions) < $m) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Dummy Classifier (' . Params::stringify($this->params()) . ')';
    }
}
