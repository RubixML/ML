<?php

namespace Rubix\ML\Exceptions;

class ClassRevisionMismatch extends RuntimeException
{
    public function __construct()
    {
        parent::__construct('Persistable serialized with incompatible class definition.');
    }
}
