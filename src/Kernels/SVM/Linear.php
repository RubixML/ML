<?php

namespace Rubix\ML\Kernels\SVM;

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
     * Return the options for the libsvm runtime.
     * 
     * @return array
     */
    public function options() : array
    {
        return [
            svm::OPT_KERNEL_TYPE => svm::KERNEL_LINEAR,
        ];
    }
}