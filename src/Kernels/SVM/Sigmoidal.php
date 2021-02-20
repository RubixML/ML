<?php

namespace Rubix\ML\Kernels\SVM;

use Rubix\ML\Specifications\ExtensionIsLoaded;
use svm;

/**
 * Sigmoidal
 *
 * S shaped nonlinearity kernel.
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
     * @param float $gamma
     * @param float $coef0
     */
    public function __construct(?float $gamma = null, float $coef0 = 0.0)
    {
        ExtensionIsLoaded::with('svm')->check();

        $this->gamma = $gamma;
        $this->coef0 = $coef0;
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
            svm::OPT_KERNEL_TYPE => svm::KERNEL_SIGMOID,
            svm::OPT_GAMMA => $this->gamma,
            svm::OPT_COEF_ZERO => $this->coef0,
        ];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Sigmoidal (gamma: {$this->gamma}, coef0: {$this->coef0})";
    }
}
