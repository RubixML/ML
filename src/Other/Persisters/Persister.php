<?php

namespace Rubix\ML\Other\Persisters;

use Rubix\ML\Persistable;

interface Persister
{
    /**
     * Restore the persistable object.
     *
     * @return \Rubix\ML\Persistable
     */
    public function restore() : Persistable;

    /**
     * Save the persitable object.
     *
     * @param  \Rubix\ML\Persistable  $persistable
     * @return void
     */
    public function save(Persistable $persistable) : void;
}
