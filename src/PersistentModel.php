<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;

class PersistentModel
{
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
                . ' opened. Check path and file permissions.');
        }

        $model = unserialize(file_get_contents($path));

        if (!$model) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        if (!$model instanceof Persistable) {
            throw new RuntimeException('Reconstituted object is not a valid'
                . ' persistent model. ' . get_class($model) . ' found.');
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
    }

    /**
     * Save the estimator to a file given the path. File contains a pickled PHP
     * object that has been serialized.
     *
     * @param  string  $path
     * @throws \InvalidArgumentException
     * @return void
     */
    public function save(string $path = '') : bool
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writeable. Check path and file permissions.');
        }

        if (empty($path)) {
            $path = strtolower(get_class($this->model)) . '_'
                . (string) time() . '.model';
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
