<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU\Exceptions;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Invalid `alpha` parameter for ELU Activation function
 */
class InvalidLeakageException extends InvalidArgumentException
{
}
