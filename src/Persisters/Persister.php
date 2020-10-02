<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use Stringable;

interface Persister extends Stringable
{
    /**
     * Save the persistable object.
     *
     * @param \Rubix\ML\Persistable $persistable
     */
    public function save(Persistable $persistable) : void;

    /**
     * Load the last saved persistable instance.
     *
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable;
}
