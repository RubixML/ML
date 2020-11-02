<?php

namespace Rubix\ML\Kernels\SVM;

use Rubix\ML\Exceptions\RuntimeException;
use svm;

/**
 * Linear
 *
 * A simple linear kernel computed by the dot product.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Linear implements Kernel
{
    /**
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function __construct()
    {
        if (!extension_loaded('svm')) {
            throw new RuntimeException('SVM extension is not loaded, check'
                . ' PHP configuration.');
        }
    }

    /**
     * Return the options for the libsvm runtime.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function options() : array
    {
        return [
            svm::OPT_KERNEL_TYPE => svm::KERNEL_LINEAR,
        ];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Linear';
    }
}
