<?php

namespace Rubix\ML;

/**
 * Persistable
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Persistable
{
    /**
     * Return the revision number of the class.
     *
     * @return string
     */
    public function revision() : string;
}
