<?php

namespace Rubix\ML\Kernels\SVM;

use RuntimeException;
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
     * @throws \RuntimeException
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
     * @return mixed[]
     */
    public function options() : array
    {
        return [
            svm::OPT_KERNEL_TYPE => svm::KERNEL_LINEAR,
        ];
    }
}
