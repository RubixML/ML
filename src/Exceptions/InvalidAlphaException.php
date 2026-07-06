<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ELU\Exceptions;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Invalid `alpha` parameter for ELU Activation function
 */
class InvalidAlphaException extends InvalidArgumentException
{
}
