<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

class PersistentModel
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
     * Return the instance of the base estimator.
     *
     * @return \Rubix\ML\Persistable
     */
    public function estimator() : Persistable
    {
        return $this->model;
    }

    /**
     * Save the estimator to a file given the path. File contains a pickled PHP
     * object that has been serialized.
     *
     * @param  string  $path
     * @throws \InvalidArgumentException
     * @return bool
     */
    public function save(string $path = '') : bool
    {
        if (empty($path)) {
            $path = strtolower($this->reflector->getShortName())
                . '-' . (string) time() . '.model';
        }

        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        return file_put_contents($path, serialize($this->model), LOCK_EX)
            ? true : false;
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
