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
     * Load the last saved model or load from backup by order of most recent.
     * 
     * @param  int  $ordinal
     * @return \Rubix\ML\Persistable
     */
    public function load(int $ordinal = 0) : Persistable;
}
