<?php

namespace Rubix\ML\Kernels\SVM;

use svm;

/**
 * Linear
 * 
 * Non linear radias basis function computes the distance from a centroid
 * or origin.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBF implements Kernel
{
    /**
     * The kernel coefficient.
     * 
     * @var float|null
     */
    protected $gamma;

    /**
     * @param  float|null  $gamma
     * @return void
     */
    public function __construct(?float $gamma = null)
    {
        $this->gamma = $gamma;
    }

    /**
     * Return the options for the libsvm runtime.
     * 
     * @return array
     */
    public function options() : array
    {
        return [
            svm::OPT_KERNEL_TYPE => svm::KERNEL_RBF,
            svm::OPT_GAMMA => $this->gamma,
        ];
    }
}