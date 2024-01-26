<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\AnomalyDetectors\Scoring;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Pipeline
 *
 * Pipeline is a meta-estimator capable of transforming an input dataset by applying a
 * series of Transformer *middleware*. Under the hood, Pipeline will automatically fit the
 * training set and transform any Dataset object supplied as an argument to one of the base
 * estimator's methods before hitting the method context. With *elastic* mode enabled,
 * Pipeline will update the fitting of Elastic transformers during partial training.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements Online, Probabilistic, Scoring, Persistable, EstimatorWrapper
{
    use AutotrackRevisions;

    /**
     * A list of transformers to be applied in series.
     *
     * @var list<\Rubix\ML\Transformers\Transformer>
     */
    protected array $transformers = [
        //
    ];

    /**
     * An instance of a base estimator to receive the transformed data.
     *
     * @var Estimator
     */
    protected Estimator $base;

    /**
     * Should we update the elastic transformers during partial train?
     *
     * @var bool
     */
    protected bool $elastic;

    /**
     * @param \Rubix\ML\Transformers\Transformer[] $transformers
     * @param Estimator $base
     * @param bool $elastic
     * @throws InvalidArgumentException
     */
    public function __construct(array $transformers, Estimator $base, bool $elastic = true)
    {
        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Transformer must'
                    . ' implement the Transformer interface.');
            }
        }

        $this->transformers = array_values($transformers);
        $this->base = $base;
        $this->elastic = $elastic;
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
        return $this->base->type();
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
            'transformers' => $this->transformers,
            'estimator' => $this->base,
            'elastic' => $this->elastic,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->base instanceof Learner
            ? $this->base->trained()
            : true;
    }

    /**
     * Return the base estimator instance.
     *
     * @return Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                $transformer->fit($dataset);
            }

            $dataset->apply($transformer);
        }

        if ($this->base instanceof Learner) {
            $this->base->train($dataset);
        }
    }

    /**
     * Perform a partial train.
     *
     * @param Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if ($this->elastic) {
            foreach ($this->transformers as $transformer) {
                if ($transformer instanceof Elastic) {
                    $transformer->update($dataset);
                }

                $dataset->apply($transformer);
            }
        } else {
            $this->preprocess($dataset);
        }

        if ($this->base instanceof Online) {
            $this->base->partial($dataset);
        }
    }

    /**
     * Preprocess the dataset and return predictions from the estimator.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->preprocess($dataset);

        return $this->base->predict($dataset);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->preprocess($dataset);

        if (!$this->base instanceof Probabilistic) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Probabilistic interface.');
        }

        return $this->base->proba($dataset);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return float[]
     */
    public function score(Dataset $dataset) : array
    {
        $this->preprocess($dataset);

        if (!$this->base instanceof Scoring) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Scoring interface.');
        }

        return $this->base->score($dataset);
    }

    /**
     * Apply the transformer stack to a dataset.
     *
     * @param Dataset $dataset
     */
    protected function preprocess(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $dataset->apply($transformer);
        }
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        foreach ($arguments as $argument) {
            if ($argument instanceof Dataset) {
                $this->preprocess($argument);
            }
        }

        return $this->base->$name(...$arguments);
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
        return 'Pipeline (' . Params::stringify($this->params()) . ')';
    }
}
