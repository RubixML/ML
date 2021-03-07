<?php

namespace Rubix\ML\Kernels\SVM;

use Rubix\ML\Specifications\ExtensionIsLoaded;
use svm;

/**
 * RBF
 *
 * Non-linear radial basis function computes the distance from a centroid or origin.
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
     * @param float|null $gamma
     */
    public function __construct(?float $gamma = null)
    {
        ExtensionIsLoaded::with('svm')->check();

        $this->gamma = $gamma;
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
            svm::OPT_KERNEL_TYPE => svm::KERNEL_RBF,
            svm::OPT_GAMMA => $this->gamma,
        ];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "RBF (gamma: {$this->gamma})";
    }
}
