<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\HuberLoss\Exceptions;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Invalid `alpha` parameter for HuberLoss Cost function
 */
class InvalidAlphaException extends InvalidArgumentException
{
}
