<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use InvalidArgumentException;
use SplObjectStorage;
use RuntimeException;

class Snapshot extends SplObjectStorage
{
    /**
     * Factory method to restore the snapshot from a file.
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

        $snapshot = unserialize(file_get_contents($path) ?: '');

        if (!$snapshot instanceof Snapshot) {
            throw new RuntimeException('Snapshot could not be reconstituted.');
        }

        return $snapshot;
    }

    /**
     * @param  array  $layers
     * @return void
     */
    public function __construct(array $layers = [])
    {
        foreach ($layers as $layer) {
            if ($layer instanceof Parametric) {
                $this->attach($layer, clone $layer->weights());
            }
        }
    }

    /**
     * Save the snapshot to a file.
     *
     * @param  string  $path
     * @throws \InvalidArgumentException
     * @return bool
     */
    public function save(string $path = '') : bool
    {
        if (empty($path)) {
            $path = (string) time() . '.snapshot';
        }

        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        return file_put_contents($path, serialize($this), LOCK_EX)
            ? true : false;
    }
}
