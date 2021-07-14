<?php

namespace Rubix\ML\Kernels\SVM;

use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
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
    protected ?float $gamma;

    /**
     * @param float|null $gamma
     */
    public function __construct(?float $gamma = null)
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('svm'),
            new ExtensionMinimumVersion('svm', '0.2.0'),
        ])->check();

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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "RBF (gamma: {$this->gamma})";
    }
}
