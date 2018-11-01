<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;

interface Persister
{
    /**
     * Save the persitable model.
     *
     * @param  \Rubix\ML\Persistable  $persistable
     * @return void
     */
    public function save(Persistable $persistable) : void;

    /**
     * Restore the persistable model.
     *
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable;
}
