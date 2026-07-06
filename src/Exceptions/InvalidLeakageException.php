<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU\Exceptions;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Invalid `Leakage` parameter for LeakyReLU Activation function
 */
class InvalidLeakageException extends InvalidArgumentException
{
}
