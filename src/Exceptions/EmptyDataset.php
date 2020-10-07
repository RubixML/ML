<?php

namespace Rubix\ML\Exceptions;

class EmptyDataset extends InvalidArgumentException
{
    public function __construct()
    {
        parent::__construct('Dataset must contain at least 1 sample.');
    }
}
