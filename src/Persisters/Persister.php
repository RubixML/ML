<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;

interface Persister
{
    /**
     * Save the persistable model.
     *
     * @param \Rubix\ML\Persistable $persistable
     */
    public function save(Persistable $persistable) : void;

    /**
     * Load the last model that was saved.
     *
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable;
}
