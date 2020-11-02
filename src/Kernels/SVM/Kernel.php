<?php

namespace Rubix\ML\Kernels\SVM;

/**
 * Kernel
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Kernel
{
    /**
     * Return the options for the libsvm runtime.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function options() : array;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
