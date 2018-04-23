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
     * Restore the model from persistence. Returns the stored object of null if
     * cannot be found or error.
     *
     * @return \Rubix\Engine\Persisters\Persistable
     */
    public function restore() : Persistable;
}
