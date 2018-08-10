<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

/**
 * Persistent Model
 *
 * It is possible to persist a model to disk by wrapping the Estimator instance
 * in a Persistent Model meta-Estimator. The Persistent Model class gives the
 * Estimator two additional methods save() and restore() that serialize and
 * unserialize to and from disk. In order to be persisted the Estimator must
 * implement the Persistable interface.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PersistentModel implements MetaEstimator
{
    /**
     * The reflector instance of the base estimator.
     *
     * @var \ReflectionClass
     */
    protected $reflector;

    /**
     * The underlying persistable estimator instance.
     *
     * @var \Rubix\ML\Persistable
     */
    protected $model;

    /**
     * Factory method to restore the model from a pickled object file at path.
     *
     * @param  string  $path
     * @throws \RuntimeException
     * @return self
     */
    public static function restore(string $path) : self
    {
        if (!file_exists($path) or !is_readable($path)) {
            throw new RuntimeException('File ' . basename($path) . ' cannot be'
                . ' opened. Check path and permissions.');
        }

        $model = unserialize(file_get_contents($path) ?: '');

        if (!$model instanceof Persistable) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return new self($model);
    }

    /**
     * @param  \Rubix\ML\Persistable  $model
     * @return void
     */
    public function __construct(Persistable $model)
    {
        $this->model = $model;
        $this->reflector = new ReflectionClass($model);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->model->type();
    }

    /**
     * Return the instance of the base estimator.
     *
     * @return \Rubix\ML\Persistable
     */
    public function estimator() : Persistable
    {
        return $this->model;
    }

    /**
     * Train the underlying estimator.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->model->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->model->predict($dataset);
    }

    /**
     * Save the estimator to a file given the path. File contains a pickled PHP
     * object that has been serialized.
     *
     * @param  string|null  $path
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function save(?string $path = null) : void
    {
        if (is_null($path)) {
            $path = strtolower($this->reflector->getShortName())
                . '-' . (string) time() . '.model';
        }

        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        $success = file_put_contents($path, serialize($this->model), LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to serialize object to storage.');
        }
    }

    /**
     * Allow methods to be called on the model from the wrapper.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->model->$name(...$arguments);
    }
}
