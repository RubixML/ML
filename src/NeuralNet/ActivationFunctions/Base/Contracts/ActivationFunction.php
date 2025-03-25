<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts;

use NumPower;
use Stringable;

/**
 * Activation Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
interface ActivationFunction extends Stringable
{
    /**
     * Compute the activation.
     *
     * @param NumPower $input Input matrix
     * @return NumPower Output matrix
     */
    public function activate(NumPower $input) : NumPower;

    /**
     * Calculate the derivative of the activation.
     *
     * @param NumPower $input Input matrix
     * @return NumPower Direvative matrix
     */
    public function differentiate(NumPower $input) : NumPower;
}
