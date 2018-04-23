<?php

namespace Rubix\Engine\Persisters;

interface Persister
{
    /**
     * Save the model. Return true on success and false on error.
     *
     * @return bool
     */
    public function save(Persistable $model) : bool;

    /**
     * Load object from persistence. Returns the stored object or null if
     * either it cannot be found or error.
     *
     * @return \Rubix\Engine\Persisters\Persistable
     */
    public function load() : Persistable;
}
