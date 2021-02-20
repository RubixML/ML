<?php

namespace Rubix\ML\Exceptions;

class MissingExtension extends RuntimeException
{
    public function __construct(string $name)
    {
        parent::__construct("The $name extension is not installed, check PHP configuration.");
    }
}
