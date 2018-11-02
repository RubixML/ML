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
     * Load a model given a version number where 0 is the last model saved.
     * 
     * @param  int  $version
     * @return \Rubix\ML\Persistable
     */
    public function load(int $version = 0) : Persistable;
}
