<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use Stringable;

/**
 * Persister
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
