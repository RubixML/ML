<?php

namespace Rubix\Engine\Estimators\Wrappers;

use Rubix\Engine\Estimators\Persistable;
use InvalidArgumentException;
use RuntimeException;

class PersistentModel extends Wrapper
{
    /**
     * Factory method to restore the model from a pickled model file given a path.
     *
     * @param  string  $path
     * @throws \RuntimeException
     * @return self
     */
    public static function restore(string $path) : self
    {
        if (!file_exists($path) || !is_readable($path)) {
            throw new RuntimeException('File ' . basename($path) . ' cannot be'
                . ' opened. Check path and file permissions.');
        }

        $estimator = unserialize(file_get_contents($path));

        if ($estimator === false) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        if (!$estimator instanceof Estimator) {
            throw new RuntimeException('Reconstituted object is not a valid'
                . ' estimator. ' . get_class($estimator) . ' found.');
        }

        return new self($estimator);
    }

    /**
     * @param  \Rubix\Engine\Estimators\Persistable  $estimator
     * @return void
     */
    public function __construct(Persistable $estimator)
    {
        parent::__construct($estimator);
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
                . 'writeable. Check path and file permissions.');
        }

        if (empty($path)) {
            $path = strtolower(get_class($this->estimator)) . '_'
                . (string) time() . '.model';
        }

        return file_put_contents($path, serialize($this->estimator),
            LOCK_EX) ? true : false;
    }
}
