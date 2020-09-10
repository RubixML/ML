<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;

interface Persister
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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
