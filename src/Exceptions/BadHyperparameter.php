<?php

namespace Rubix\ML\Exceptions;

class BadHyperparameter extends InvalidArgumentException
{
    /**
     * @param string $message
     */
    public function __construct(string $message)
    {
        parent::__construct("Bad hyper-parameter: $message");
    }
}
