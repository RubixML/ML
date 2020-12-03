<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Specification
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class Specification
{
    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    abstract public function check() : void;

    /**
     * Does the specification pass?
     *
     * @return bool
     */
    public function passes() : bool
    {
        try {
            $this->check();

            return true;
        } catch (InvalidArgumentException $exception) {
            return false;
        }
    }

    /**
     * Does the specification fail?
     *
     * @return bool
     */
    public function fails() : bool
    {
        return !$this->passes();
    }
}
