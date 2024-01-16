<?php

namespace Rubix\ML;

/**
 * Wrapper
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Wrapper extends Estimator
{
    /**
     * Return the base estimator instance.
     *
     * @return Estimator
     */
    public function base(): Estimator;
}