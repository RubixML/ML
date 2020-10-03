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
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function save(Persistable $persistable) : void;

    /**
     * Load the last saved persistable instance.
     *
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable;
}
