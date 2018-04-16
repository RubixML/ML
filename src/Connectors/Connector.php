<?php

namespace Rubix\Engine\Connectors;

interface Connector
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
     * @return \Rubix\Engine\Connectors\Persistable
     */
    public function restore() : Persistable;
}
