<?php

namespace Rubix\ML\Specifications;

use Exception;

abstract class Specification
{
    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Exception
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
        } catch (Exception $e) {
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
