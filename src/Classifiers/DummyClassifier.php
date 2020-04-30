<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Strategies\Prior;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use InvalidArgumentException;
use RuntimeException;

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
class DummyClassifier implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $strategy;

    /**
     * Has the learner been trained?
     *
     * @var bool
     */
    protected $trained;

    /**
     * @param \Rubix\ML\Other\Strategies\Categorical|null $strategy
     */
    public function __construct(?Categorical $strategy = null)
    {
        $this->strategy = $strategy ?? new Prior();
        $this->trained = false;
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
        return DataType::all();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
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
        return $this->trained;
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
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->strategy->fit($dataset->labels());

        $this->trained = true;
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained) {
            throw new RuntimeException('Estimator has not been trained.');
        }
        
        $n = $dataset->numRows();

        $predictions = [];

        while (count($predictions) < $n) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }
}
