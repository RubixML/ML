<?php

namespace Rubix\ML\Kernels\SVM;

use svm;

/**
 * Sigmoidal
 * 
 * S shaped nonliearity kernel.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sigmoidal implements Kernel
{
    /**
     * The kernel coefficient.
     * 
     * @var float|null
     */
    protected $gamma;

    /**
     * The independent term.
     * 
     * @var float
     */
    protected $coef0;

    /**
     * @param  float  $gamma
     * @param  float  $coef0
     * @return void
     */
    public function __construct(?float $gamma = null, float $coef0 = 0.)
    {
        $this->gamma = $gamma;
        $this->coef0 = $coef0;
    }

    /**
     * Return the options for the libsvm runtime.
     * 
     * @return array
     */
    public function options() : array
    {
        return [
            svm::OPT_KERNEL_TYPE => svm::KERNEL_SIGMOID,
            svm::OPT_GAMMA => $this->gamma,
            svm::OPT_COEF_ZERO => $this->coef0,
        ];
    }
}