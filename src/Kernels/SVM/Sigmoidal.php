<?php

namespace Rubix\ML\Kernels\SVM;

use RuntimeException;
use Stringable;
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
class Sigmoidal implements Kernel, Stringable
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
     * @throws \RuntimeException
     */
    public function __construct(?float $gamma = null, float $coef0 = 0.0)
    {
        if (!extension_loaded('svm')) {
            throw new RuntimeException('SVM extension is not loaded, check'
                . ' PHP configuration.');
        }

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
